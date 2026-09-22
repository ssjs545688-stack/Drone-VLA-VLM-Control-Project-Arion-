from pathlib import Path
from datetime import datetime
import yaml,torch,rclpy,re
from rclpy.node import Node
from std_msgs.msg import String
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from guide_interfaces.srv import GuideLLM
from ament_index_python.packages import get_package_share_directory
from llm_drone_control.schema import DEFINE_SCHEMA,SYSTEM_PROMPT

class LLMService(Node):
    def __init__(self):
        super().__init__("llm_service")

        # 로그
        log_dir=Path.cwd()/"llm_logs"
        log_dir.mkdir(parents=True,exist_ok=True)
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path=log_dir/f"llm_log_{timestamp}.log"
        self.get_logger().info(f"📁 로그 파일 생성 경로: {self.log_file_path}")

        # yaml 로드
        config_path=Path(get_package_share_directory("llm_drone_control"))/"config"/"model.yaml"
        with open(config_path,"r",encoding="utf-8") as f:
            config=yaml.safe_load(f) or {}

        # 파라미터 설정
        model_config=config.get("model",{})
        model_path=Path(model_config.get("path","~/Drone-VLA-VLM-Control-Project-Arion-/models/qwen3-1.7B-drone-int4-awq")).expanduser()
        self.max_new_tokens=model_config.get("max_new_tokens",64)
        self.do_sample=model_config.get("do_sample",False)
        torch_dtype=model_config.get("torch_dtype","auto")
        device_map=model_config.get("device_map","auto")

        if not model_path.exists():
            raise FileNotFoundError(f"모델 경로가 존재하지 않습니다: {model_path}")

        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(
            model_path,local_files_only=True,torch_dtype=torch_dtype,device_map=device_map
        )
        self.model=PeftModel.from_pretrained(self.model,lora_path,local_files_only=True)
        self.model.eval()

        # warning 제거
        self.model.generation_config.do_sample=False
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.model.generation_config.top_k=None

        # pub & sub & service
        self.response_pub=self.create_publisher(String,"llm_response",10)
        self.srv=self.create_service(GuideLLM,"llm",self.callback)
        self.voice_sub=self.create_subscription(String,"/voice_command",self.voice_command_callback,10)

        self.get_logger().info("==========================================")
        self.get_logger().info("Qwen3-1.7B + LoRA LLM Service Ready")
        self.get_logger().info("GuideLLM Service : /llm")
        self.get_logger().info("Smartphone Topic : /voice_command")
        self.get_logger().info("LLM Response     : /llm_response")
        self.get_logger().info("==========================================")

    def callback(self,request,response):
        response.response=self.process_prompt(request.prompt)
        return response

    def voice_command_callback(self,msg):
        prompt=msg.data.strip()
        if not prompt:
            self.get_logger().warn("⚠️ 스마트폰에서 빈 음성 명령이 들어왔습니다.")
            return

        self.get_logger().info(f"📱 스마트폰 음성 명령 수신: {prompt}")
        result=self.process_prompt(prompt)
        self.get_logger().info(f"📤 스마트폰 명령 LLM 처리 완료: {result}")

    def process_prompt(self,prompt):
        self.get_logger().info(f"🤖 LLM 입력: {prompt}")

        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt}
        ]

        inputs=self.tokenizer.apply_chat_template(
            messages,tools=DEFINE_SCHEMA,add_generation_prompt=True,
            enable_thinking=False,return_tensors="pt",return_dict=True
        ).to(self.model.device)

        with torch.inference_mode():
            outputs=self.model.generate(
                **inputs,max_new_tokens=self.max_new_tokens,do_sample=self.do_sample
            )

        result=self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        matches=re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>",result,re.DOTALL)
        cleaned=[f"<tool_call>{re.sub(r'\\s+',' ',m.strip())}</tool_call>" for m in matches]
        response_text="\n".join(cleaned) if cleaned else result.strip()

        msg=String()
        msg.data=response_text
        self.response_pub.publish(msg)

        self.get_logger().info(f"질문: {prompt}")
        self.get_logger().info(f"답변: {response_text}")

        log_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry=f"[{log_timestamp}]\n[질문] {prompt}\n[답변] {response_text}\n{'='*60}\n"

        try:
            with open(self.log_file_path,"a",encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            self.get_logger().error(f"로그 파일 저장 중 오류 발생: {e}")

        return response_text


def main():
    rclpy.init()
    node=LLMService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":
    main()