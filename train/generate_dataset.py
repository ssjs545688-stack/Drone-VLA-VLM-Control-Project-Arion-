#!/usr/bin/env python3
"""
Qwen3-0.6B 드론 제어 Tool Calling 파인튜닝용 학습 데이터셋(train.jsonl) 자동 생성 스크립트
"""

import json
import random
from pathlib import Path

# 1. 자연어 명령 패턴 정의 (다양한 한국어 표현 조합)
TAKEOFF_PHRASES = [
    "드론 고도 {alt}m로 띄워줘",
    "{alt}미터 높이로 이륙해",
    "지면에서 {alt}m 상승해서 떠라",
    "{alt}m 높이로 이륙 시도해줘",
    "드론 높이 {alt}m까지 수직 이륙해라",
    "고도 {alt}m로 올려줘"
]

LAND_PHRASES = [
    "이제 바닥으로 안전하게 착륙해",
    "지금 위치에 착륙해줘",
    "착륙 시도해",
    "지면으로 착륙해라",
    "비행 종료하고 착륙해",
    "드론 바닥에 착륙시켜줘"
]

RETURN_PHRASES = [
    "왔던 길 그대로 되돌아가줘",
    "이동했던 경로 그대로 출발지로 복귀해",
    "홈 위치로 왔던 경로 역순으로 돌아와",
    "출발지로 복귀해줘",
    "복귀 경로 따라 원점으로 돌아가",
    "처음 이륙 위치로 되돌아가라"
]

# 이동 관련 개별 표현 패턴
MOVE_FORWARD = ["앞으로 {dist}미터 전진해", "{dist}m 앞으로 가줘", "전방으로 {dist}m 이동해"]
MOVE_BACK = ["뒤로 {dist}m 물러나 줘", "{dist}미터 후진해", "후방으로 {dist}m 이동"]
MOVE_RIGHT = ["오른쪽으로 {dist}미터 가줘", "우측으로 {dist}m 이동해", "우측 {dist}m 가라"]
MOVE_LEFT = ["왼쪽으로 {dist}m 이동해", "좌측으로 {dist}미터 가줘", "좌측 {dist}m 가라"]
MOVE_UP = ["위로 {dist}미터 더 올라가", "상승 {dist}m 해줘", "고도 {dist}m 높여라"]
MOVE_DOWN = ["아래로 {dist}m 내려가", "하강 {dist}미터 해줘", "고도 {dist}m 낮춰라"]

num_rep = 50  # 반복 횟수 설정 (N*5개의 샘플 생성)

def make_sample(user_cmd: str, func_name: str, args: dict) -> dict:
    """하나의 사용자 명령과 Tool Call을 ChatML 메시지 구조로 묶는다."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "너는 드론 제어 AI야. 사용자의 자연어 명령을 분석하여 적절한 함수(Tool Call)를 호출해."
            },
            {
                "role": "user",
                "content": user_cmd
            },
            {
                "role": "assistant",
                "content": f"<tool_call>\n{json.dumps({'name': func_name, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"
            }
        ]
    }


def generate_dataset(num_repeats: int = num_rep, output_file: str = "train.jsonl"):
    """
    자연어 수식을 조합하여 train.jsonl 데이터 생성
    (비율 -> move : land : takeoff : return_along_path = 2 : 1 : 1 : 1)
    """
    dataset = []

    # move 명령 생성용 다양한 방향 패턴 함수 리스트
    move_generators = [
        lambda d: (random.choice(MOVE_FORWARD).format(dist=d), {"forward": d}),
        lambda d: (random.choice(MOVE_BACK).format(dist=d), {"forward": -d}),
        lambda d: (random.choice(MOVE_RIGHT).format(dist=d), {"right": d}),
        lambda d: (random.choice(MOVE_LEFT).format(dist=d), {"right": -d}),
        lambda d: (random.choice(MOVE_UP).format(dist=d), {"up": d}),
        lambda d: (random.choice(MOVE_DOWN).format(dist=d), {"up": -d}),
    ]

    for _ in range(num_repeats):
        # 1. 이륙 (takeoff) - 1개
        alt = round(random.uniform(1.0, 5.0), 1)
        cmd = random.choice(TAKEOFF_PHRASES).format(alt=alt)
        dataset.append(make_sample(cmd, "takeoff", {"altitude": alt}))

        # 2. 착륙 (land) - 1개
        cmd = random.choice(LAND_PHRASES)
        dataset.append(make_sample(cmd, "land", {}))

        # 3. 복귀 (return_along_path) - 1개
        cmd = random.choice(RETURN_PHRASES)
        dataset.append(make_sample(cmd, "return_along_path", {}))

        # 4. 상대 이동 (move) - 방향을 무작위로 선택하여 정확히 2개만 생성
        for _ in range(2):
            dist = round(random.uniform(0.5, 5.0), 1)
            gen = random.choice(move_generators)
            move_cmd, args = gen(dist)
            dataset.append(make_sample(move_cmd, "move", args))

    # 데이터 순서 섞기
    random.shuffle(dataset)

    # 파일 저장
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 총 {len(dataset)}개의 파인튜닝 데이터 생성이 완료되었습니다!")
    print(f"📄 저장 위치: {output_path.resolve()}")


if __name__ == "__main__":
    # num_repeats=50 설정 시 약 550개의 샘플 생성
    generate_dataset(num_repeats=num_rep, output_file="train.jsonl")