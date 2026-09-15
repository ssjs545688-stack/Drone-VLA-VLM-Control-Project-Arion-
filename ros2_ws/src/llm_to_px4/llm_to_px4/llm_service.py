#!/usr/bin/env python3
from pathlib import Path
import torch,rclpy  # PyTorch와 ROS 2 Python 라이브러리
from rclpy.node import Node # ROS 2 Node 클래스 가져오기
from transformers import AutoTokenizer,AutoModelForCausalLM # 토크나이저와 언어 모델 클래스
from guide_interfaces.srv import GuideLLM   # ROS 2 Service 인터페이스 가져오기

SYSTEM_PROMPT="""너는 드론 제어 명령 변환기다.
사용자의 입력 문장을 해석하여 반드시 아래 4가지 표준 명령 포맷 중 하나로만 출력하라.

[출력 포맷]
- TAKEOFF <고도>
- LAND
- MOVE <x> <y> <z>
- HOVER

[좌표 및 단위]
- 모든 거리는 meter(m) 단위이며 숫자 뒤에 단위를 붙이지 않는다.
- TAKEOFF의 고도는 현재 위치 기준 상대 고도다.
- MOVE의 x,y,z는 현재 위치 기준 상대 좌표다.
- x는 전후 방향이며 +x는 앞으로, -x는 뒤로 이동한다.
- y는 좌우 방향이며 +y는 오른쪽, -y는 왼쪽으로 이동한다.
- z는 상하 방향이며 +z는 위로, -z는 아래로 이동한다.

[제약 사항]
1. 출력은 반드시 지정된 명령 포맷 중 하나만 사용한다.
2. 설명, 인삿말, 문장, 마크다운, 코드블록을 절대 출력하지 않는다.
3. 모든 수치는 정수 또는 소수점 숫자로만 출력하며 단위 기호를 붙이지 않는다.
4. 사용자의 명령을 명확하게 해석할 수 없는 경우 HOVER를 출력한다.
5. 출력 앞뒤로 공백을 포함하지 않고 오직 한 줄로만 출력한다.

[예시]
"드론을 10m로 이륙시켜" → TAKEOFF 10
"이제 착륙해" → LAND
"앞으로 2m 이동해" → MOVE 2 0 0
"오른쪽으로 3m 이동해" → MOVE 0 3 0
"위로 1m 올라가" → MOVE 0 0 1
"드론 멈춰" → HOVER
"""

class LLMService(Node):
    def __init__(self):
        super().__init__("llm_service")
        model_path=Path.home()/"Drone-VLA-VLM-Control-Project-Arion-"/"models"/"Qwen3-0.6B"

        # Qwen3 토크나이저 & llm 모델 로드
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype="auto",device_map="auto")

        # llm ROS 2 Service 서버 생성
        self.srv=self.create_service(GuideLLM,"llm",self.callback)

        self.get_logger().info("Qwen3-0.6B LLM Service Ready")

    def callback(self,request,response):
        # 문자열(request.prompt)을 LLM이 사용하는 chat message 형식으로 변환
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":request.prompt}
        ]
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