#!/usr/bin/env python3
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 모델 경로 설정
model_path = Path.home() / "Drone-VLA-VLM-Control-Project-Arion-" / "models" / "Qwen3-0.6B"

print("모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, local_files_only=True, torch_dtype="auto", device_map="auto"
)
print("모델 로딩 완료!\n")

# 2. 이륙(takeoff), 착륙(land), 이동(move), 복귀(return_to_launch) 함수 스키마 정의
define_schema = [
    {
        "type": "function",
        "function": {
            "name": "takeoff",
            "description": "이륙: 드론을 지정한 목표 고도로 수직 이륙시킵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "altitude": {
                        "type": "number",
                        "description": "이륙 목표 고도 (단위: 미터)"
                    }
                },
                "required": ["altitude"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "land",
            "description": "착륙: 드론을 현재 위치에서 지면으로 안전하게 착륙시킵니다.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move",
            "description": "이동: 현재 위치를 기준으로 드론을 전후, 좌우, 상하 방향으로 상대 이동시킵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "forward": {
                        "type": "number",
                        "description": "앞(+) 또는 뒤(-)로 이동할 상대 거리 (단위: 미터, 예: 앞으로 2m 이동 시 2.0, 뒤로 1m 이동 시 -1.0)"
                    },
                    "right": {
                        "type": "number",
                        "description": "오른쪽(+) 또는 왼쪽(-)으로 이동할 상대 거리 (단위: 미터, 예: 오른쪽으로 1.5m 이동 시 1.5, 왼쪽 이동 시 음수)"
                    },
                    "up": {
                        "type": "number",
                        "description": "상승(+) 또는 하강(-)할 상대 고도 변화량 (단위: 미터, 예: 위로 1m 상승 시 1.0, 1m 하강 시 -1.0)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "return_along_path",
            "description": "복귀: 이륙 위치로부터 현재 위치까지 이동해 온 비행 경로를 그대로 역순으로 되짚어가며(Backtrack) 최초 이륙 위치(출발지/홈)로 안전하게 복귀합니다.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

# 3. 테스트할 사용자 프롬프트 목록 (4가지 기능 검증)
test_prompts = [
    "드론 고도 2m로 띄워줘",
    "앞으로 3미터 전진해",
    "오른쪽으로 1.5미터 가줘",
    "위로 1미터 더 올라가",
    "왔던 길 그대로 되돌아가줘",
    "이동했던 경로 그대로 출발지로 복귀해",
    "홈 위치로 왔던 경로 역순으로 돌아와",
    "이제 바닥으로 안전하게 착륙해"
]

# 4. 추론 테스트 실행
for prompt in test_prompts:
    print("=" * 50)
    print(f"[사용자 명령]: {prompt}")
    
    messages = [{"role": "user", "content": prompt}]
    
    # define_schema 스키마를 tokenizer 템플릿에 주입
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=define_schema,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=128, do_sample=False)

    result = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    print(f"[LLM 출력 결과]:\n{result.strip()}\n")
