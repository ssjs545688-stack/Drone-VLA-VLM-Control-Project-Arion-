#!/usr/bin/env python3
from pathlib import Path
import yaml,torch,rclpy
import re
from rclpy.node import Node
from std_msgs.msg import String
from transformers import AutoTokenizer,AutoModelForCausalLM
from guide_interfaces.srv import GuideLLM
from ament_index_python.packages import get_package_share_directory
from llm_drone_control.schema import DEFINE_SCHEMA,SYSTEM_PROMPT

class LLMService(Node):
    def __init__(self):
        super().__init__("llm_service")

        config_path=Path(get_package_share_directory("llm_drone_control"))/"config"/"model.yaml"
        with open(config_path,"r",encoding="utf-8") as f:
            config=yaml.safe_load(f) or {}

        model_config=config.get("model",{})

        model_path=Path(model_config.get("path","~/Drone-VLA-VLM-Control-Project-Arion-/models/Qwen3-0.6B")).expanduser()
        max_new_tokens=model_config.get("max_new_tokens",256)
        do_sample=model_config.get("do_sample",False)
        torch_dtype=model_config.get("torch_dtype","auto")
        device_map=model_config.get("device_map","auto")

        if not model_path.exists():
            raise FileNotFoundError(f"모델 경로가 존재하지 않습니다: {model_path}")

        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=torch_dtype,device_map=device_map)

        self.max_new_tokens=max_new_tokens
        self.do_sample=do_sample

        # publiser & service
        self.response_pub=self.create_publisher(String,"llm_response",10)
        self.srv=self.create_service(GuideLLM,"llm",self.callback)

        self.get_logger().info("Qwen3-0.6B LLM Service Ready")

    def callback(self,request,response):
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":request.prompt}
        ]

        inputs=self.tokenizer.apply_chat_template(
            messages,
            tools=DEFINE_SCHEMA,
            add_generation_prompt=True,
            enable_thinking=False,  # <think> 태그 비활성화
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)

        with torch.inference_mode():
            outputs=self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=None,
                top_p=None,
                top_k=None
            )

        result=self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True)
        matches=re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>",result,re.DOTALL)
        response.response="\n".join(f"<tool_call>{m}</tool_call>" for m in matches)

        msg=String()
        msg.data=response.response
        self.response_pub.publish(msg)

        self.get_logger().info(f"질문: {request.prompt}")
        self.get_logger().info(f"답변: {response.response}")
        return response

def main():
    rclpy.init()
    node=LLMService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()