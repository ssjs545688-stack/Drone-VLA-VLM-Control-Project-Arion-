```python
from pathlib import Path
from datetime import datetime
import time,yaml,torch,rclpy,re
from rclpy.node import Node
from std_msgs.msg import String
from transformers import AutoTokenizer,AutoModelForCausalLM
from guide_interfaces.srv import GuideLLM
from ament_index_python.packages import get_package_share_directory
from llm_drone_control.schema import DEFINE_SCHEMA,SYSTEM_PROMPT

class LLMService(Node):
    def __init__(self):
        super().__init__("llm_service")

        log_dir=Path.cwd()/"llm_logs"
        log_dir.mkdir(parents=True,exist_ok=True)
        self.log_file_path=log_dir/f"llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.get_logger().info(f"📁 로그 파일 생성 경로: {self.log_file_path}")

        config_path=Path(get_package_share_directory("llm_drone_control"))/"config"/"model.yaml"
        with open(config_path,"r",encoding="utf-8") as f:
            config=yaml.safe_load(f) or {}

        model_config=config.get("model",{})
        model_path=Path(model_config.get(
            "path",
            "~/Drone-VLA-VLM-Control-Project-Arion-/models/qwen3-1.7B-drone-int4-awq"
        )).expanduser()

        self.max_new_tokens=model_config.get("max_new_tokens",64)
        self.do_sample=model_config.get("do_sample",False)

        if not model_path.exists():
            raise FileNotFoundError(f"모델 경로가 존재하지 않습니다: {model_path}")

        self.tokenizer=AutoTokenizer.from_pretrained(
            model_path,local_files_only=True
        )

        self.model=AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
        )

        self.model.eval()
        self.model.generation_config.do_sample=False
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.model.generation_config.top_k=None

        self.response_pub=self.create_publisher(String,"llm_response",10)
        self.srv=self.create_service(GuideLLM,"llm",self.callback)
        self.voice_sub=self.create_subscription(
            String,"/voice_command",self.voice_command_callback,10
        )

        self.get_logger().info("==========================================")
        self.get_logger().info("Qwen3-1.7B INT4 AWQ LLM Service Ready")
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
        self.process_prompt(prompt)

    def process_prompt(self,prompt):
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt},
        ]

        inputs=self.tokenizer.apply_chat_template(
            messages,
            tools=DEFINE_SCHEMA,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )

        inputs={k:v.to(self.model.device) for k,v in inputs.items()}
        input_tokens=inputs["input_ids"].shape[-1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time=time.perf_counter()

        with torch.inference_mode():
            outputs=self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time=time.perf_counter()

        generation_time=end_time-start_time
        output_tokens=outputs.shape[-1]-input_tokens
        tokens_per_sec=output_tokens/generation_time if generation_time>0 else 0.0

        result=self.tokenizer.decode(
            outputs[0][input_tokens:],
            skip_special_tokens=True,
        )

        matches=re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            result,
            re.DOTALL,
        )

        cleaned=[]
        for match in matches:
            match=re.sub(r"\s+"," ",match.strip())
            cleaned.append(f"<tool_call>{match}</tool_call>")

        response_text="\n".join(cleaned) if cleaned else result.strip()

        msg=String()
        msg.data=response_text
        self.response_pub.publish(msg)

        log_entry=(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
            f"💬 [질문] {prompt}\n"
            f"🤖 [답변] {response_text}\n"
            f"📊 [입력 토큰] {input_tokens}\n"
            f"📊 [생성 토큰] {output_tokens}\n"
            f"📊 [생성 시간] {generation_time:.4f}초\n"
            f"📊 [생성 속도] {tokens_per_sec:.2f} tokens/s\n"
            f"{'='*60}\n"
        )

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