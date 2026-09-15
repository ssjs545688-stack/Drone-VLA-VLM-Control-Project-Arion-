#!/usr/bin/env python3
import torch,rclpy #파이토치, ros 라이브러리 들어감
from rclpy.node import Node #노드 클래스 주입
from transformers import AutoTokenizer,AutoModelForCausalLM #AI 로컬 돌리는 라이브러리
from guide_interfaces.srv import GuideLLM #인터페이스

class LLMService(Node):
    def __init__(self):
        super().__init__("llm_service") #노드 이름 지정
        model_path="/home/hkit/Drone-VLA-VLM-Control-Project-Arion-/models/Qwen3-0.6B" #모델 경로
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        #모델 호출
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype="auto",device_map="auto")
        #여기서 토치 라이브러리 물린걸로 봐서 온디바이스 할 때 이걸로 gpu 물어야 함
        self.srv=self.create_service(GuideLLM,"llm",self.callback)
        #서비스 서버 만들기
        self.get_logger().info("Qwen3-0.6B LLM Service Ready")
        #로그 찍음

    def callback(self,request,response):
        messages=[{"role":"user","content":request.prompt}]
        #리퀘스트 날라오면 이변수에 저장
        inputs=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(self.model.device)
        #메세지가 변환
        with torch.no_grad():
            outputs=self.model.generate(inputs,max_new_tokens=256,do_sample=False)
        response.response=self.tokenizer.decode(outputs[0][inputs.shape[-1]:],skip_special_tokens=True)
        # 응답 생성에서 이걸 px4_msg 형태로 응답하게
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