#!/usr/bin/env python3
"""
Qwen3-0.6B 드론 제어 Tool Calling 파인튜닝용
train/validation 데이터셋 자동 생성 스크립트
"""

import json
import random
from pathlib import Path
from llm_drone_control.schema import SYSTEM_PROMPT


# 자연어 명령 패턴

# 1. Takeoff
TAKEOFF_PHRASES = [
    "드론 고도 {alt}m로 띄워줘",
    "{alt}미터 높이로 이륙해",
    "지면에서 {alt}m 상승해서 떠라",
    "{alt}m 높이로 이륙해줘",
    "드론 높이 {alt}m까지 수직 이륙해",
    "고도 {alt}m로 이륙해"
]

# 2. Land
LAND_PHRASES = [
    "이제 바닥으로 안전하게 착륙해",
    "지금 위치에 착륙해줘",
    "착륙해",
    "지면으로 착륙해라",
    "비행 종료하고 착륙해",
    "드론을 지면에 착륙시켜줘"
]

# 3. Goto history - previous
GOTO_PREVIOUS_PHRASES = [
    "바로 이전 위치로 돌아가",
    "직전 위치로 복귀해줘",
    "방금 이동하기 전 위치로 돌아가",
    "이전 위치로 되돌아가",
    "직전 명령 시작 위치로 돌아와",
    "바로 전 위치로 복귀해"
]

# 4. Goto history - first
GOTO_FIRST_PHRASES = [
    "처음 위치로 돌아가",
    "최초 출발지로 복귀해줘",
    "처음 이륙했던 위치로 돌아가",
    "원점으로 돌아가",
    "홈 위치로 복귀해",
    "출발했던 위치로 돌아와"
]

# 5. Reverse plan
REVERSE_PLAN_PHRASES = [
    "왔던 길 그대로 되돌아가줘",
    "이동했던 경로를 역순으로 돌아가",
    "지금까지 온 경로를 거꾸로 따라가",
    "방금까지 이동한 경로를 되짚어 돌아가",
    "왔던 경로 그대로 출발 방향으로 돌아가",
    "이동 경로를 역순으로 복귀해"
]

# 6. Move - Forward
MOVE_FORWARD = [
    "앞으로 {dist}미터 전진해",
    "{dist}m 앞으로 가줘",
    "전방으로 {dist}m 이동해"
]

# 7. Move - Backward
MOVE_BACK = [
    "뒤로 {dist}m 물러나 줘",
    "{dist}미터 후진해",
    "후방으로 {dist}m 이동해"
]

# 8. Move - Right
MOVE_RIGHT = [
    "오른쪽으로 {dist}미터 가줘",
    "우측으로 {dist}m 이동해",
    "우측으로 {dist}m 가라"
]

# 9. Move - Left
MOVE_LEFT = [
    "왼쪽으로 {dist}m 이동해",
    "좌측으로 {dist}미터 가줘",
    "좌측으로 {dist}m 가라"
]

# 10. Move - Up
MOVE_UP = [
    "위로 {dist}미터 더 올라가",
    "상승 {dist}m 해줘",
    "고도를 {dist}m 높여라"
]

# 11. Move - Down
MOVE_DOWN = [
    "아래로 {dist}m 내려가",
    "하강 {dist}미터 해줘",
    "고도를 {dist}m 낮춰라"
]


# Dataset 설정
NUM_REPEATS = 50    # 데이터 수 = NUM_REPEATS * 5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
OUTPUT_PATH = "dataset"

# 데이터 형식
def make_sample(user_cmd: str, func_name: str, args: dict) -> dict:
    """하나의 사용자 명령과 Tool Call을 ChatML 메시지 구조로 만든다."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_cmd},
            {
                "role": "assistant",
                "content": f"<tool_call>\n"
                           f"{json.dumps({'name': func_name, 'arguments': args}, ensure_ascii=False)}"
                           f"\n</tool_call>"
            }
        ]
    }

# 데이터 생성
def generate_dataset(num_repeats: int = NUM_REPEATS):
    """자연어 명령과 Tool Call을 조합하여 전체 데이터셋을 생성한다."""
    dataset = []

    move_generators = [
        lambda d: (
            random.choice(MOVE_FORWARD).format(dist=d),
            {"dx": d, "dy": 0.0, "dz": 0.0, "d_yaw": 0.0}
        ),
        lambda d: (
            random.choice(MOVE_BACK).format(dist=d),
            {"dx": -d, "dy": 0.0, "dz": 0.0, "d_yaw": 0.0}
        ),
        lambda d: (
            random.choice(MOVE_RIGHT).format(dist=d),
            {"dx": 0.0, "dy": d, "dz": 0.0, "d_yaw": 0.0}
        ),
        lambda d: (
            random.choice(MOVE_LEFT).format(dist=d),
            {"dx": 0.0, "dy": -d, "dz": 0.0, "d_yaw": 0.0}
        ),
        lambda d: (
            random.choice(MOVE_UP).format(dist=d),
            {"dx": 0.0, "dy": 0.0, "dz": d, "d_yaw": 0.0}
        ),
        lambda d: (
            random.choice(MOVE_DOWN).format(dist=d),
            {"dx": 0.0, "dy": 0.0, "dz": -d, "d_yaw": 0.0}
        ),
    ]

    for _ in range(num_repeats):
        # 1. 이륙
        alt = round(random.uniform(1.0, 5.0), 1)
        cmd = random.choice(TAKEOFF_PHRASES).format(alt=alt)
        dataset.append(
            make_sample(cmd, "takeoff", {"altitude": alt})
        )

        # 2. 착륙
        cmd = random.choice(LAND_PHRASES)
        dataset.append(
            make_sample(cmd, "land", {})
        )

        # 3. 이동 경로 역순 복귀
        cmd = random.choice(REVERSE_PLAN_PHRASES)
        dataset.append(
            make_sample(cmd, "reverse_plan", {})
        )

        # 4. 상대 이동 2개
        for _ in range(2):
            dist = round(random.uniform(0.5, 5.0), 1)
            move_cmd, args = random.choice(move_generators)(dist)
            dataset.append(
                make_sample(move_cmd, "move", args)
            )

    random.shuffle(dataset)
    return dataset

# 데이터 분리
def split_dataset(dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """전체 데이터셋을 train/validation으로 분리한다."""
    if abs(train_ratio + val_ratio - 1.0) > 1e-6:
        raise ValueError("TRAIN_RATIO + VAL_RATIO는 1.0이어야 합니다.")

    split_index = int(len(dataset) * train_ratio)

    train_dataset = dataset[:split_index]
    val_dataset = dataset[split_index:]

    return train_dataset, val_dataset

# 데이터 저장
def save_jsonl(dataset, output_file):
    """데이터셋을 지정된 출력 디렉토리에 JSONL로 저장한다."""
    output_path = Path(OUTPUT_PATH) / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"📄 저장 위치: {output_path.resolve()}")


if __name__ == "__main__":
    # 전체 데이터 생성
    dataset = generate_dataset(NUM_REPEATS)

    # train / validation 분리
    train_dataset, val_dataset = split_dataset(
        dataset,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO
    )

    # 각각 저장
    save_jsonl(train_dataset, "train.jsonl")
    save_jsonl(val_dataset, "val.jsonl")

    print(f"\n✅ 전체 데이터: {len(dataset)}개")
    print(f"📚 학습 데이터: {len(train_dataset)}개 ({TRAIN_RATIO:.0%})")
    print(f"🧪 검증 데이터: {len(val_dataset)}개 ({VAL_RATIO:.0%})")