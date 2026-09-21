#!/usr/bin/env python3
from pathlib import Path
import torch,rclpy  # PyTorch와 ROS 2 Python 라이브러리
from rclpy.node import Node # ROS 2 Node 클래스 가져오기
from transformers import AutoTokenizer,AutoModelForCausalLM # 토크나이저와 언어 모델 클래스
from guide_interfaces.srv import GuideLLM   # ROS 2 Service 인터페이스 가져오기

class LLMService(Node):
    def __init__(self):
        super().__init__("llm_service")
        model_path=Path.home()/"Drone-VLA-VLM-Control-Project-Arion-"/"models"/"Qwen3-1.7B"

        # Qwen3 토크나이저 & llm 모델 로드
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype="auto",device_map="auto")

        # llm ROS 2 Service 서버 생성
        self.srv=self.create_service(GuideLLM,"llm",self.callback)

        self.get_logger().info("Qwen3-1.7B LLM Service Ready")

    def callback(self,request,response):
        # 문자열(request.prompt)을 LLM이 사용하는 chat message 형식으로 변환
        messages=[{"role":"user","content":request.prompt}]
        # 메시지를 Qwen chat 형식으로 변환하고 토큰화한 뒤 모델 장치로 이동
        inputs=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(self.model.device)

        # LLM 추론 수행
        with torch.no_grad():
            outputs=self.model.generate(inputs,max_new_tokens=256,do_sample=False)

        # 생성된 토큰을 문자열로 디코딩하여 Service Response에 저장 > 차후 px4_msg 형태로
        response.response=self.tokenizer.decode(outputs[0][inputs.shape[-1]:],skip_special_tokens=True)

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