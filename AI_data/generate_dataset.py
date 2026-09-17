#!/usr/bin/env python3
"""
Qwen3-0.6B 드론 제어 Tool Calling 파인튜닝용
train/validation/test 데이터셋 자동 생성 스크립트
"""

import json
import random
from pathlib import Path
from llm_drone_control.schema import SYSTEM_PROMPT


# ============================================================
# Train 문장 패턴
# ============================================================

TRAIN_TAKEOFF=[
    "드론 고도 {alt}m로 띄워줘",
    "{alt}미터 높이로 이륙해",
    "지면에서 {alt}m 상승해서 떠라",
    "{alt}m 높이로 이륙해줘",
    "드론 높이 {alt}m까지 수직 이륙해",
    "고도 {alt}m로 이륙해"
]

TRAIN_LAND=[
    "이제 바닥으로 안전하게 착륙해",
    "지금 위치에 착륙해줘",
    "착륙해",
    "지면으로 착륙해라",
    "비행 종료하고 착륙해",
    "드론을 지면에 착륙시켜줘"
]

TRAIN_GOTO_PREVIOUS=[
    "바로 이전 위치로 돌아가",
    "직전 위치로 복귀해줘",
    "방금 이동하기 전 위치로 돌아가",
    "이전 위치로 되돌아가",
    "직전 위치로 돌아와",
    "바로 전 위치로 복귀해"
]

TRAIN_GOTO_FIRST=[
    "처음 위치로 돌아가",
    "최초 출발지로 복귀해줘",
    "처음 이륙했던 위치로 돌아가",
    "원점으로 돌아가",
    "홈 위치로 복귀해",
    "출발했던 위치로 돌아와"
]

TRAIN_REVERSE=[
    "왔던 길 그대로 되돌아가줘",
    "이동했던 경로를 역순으로 돌아가",
    "지금까지 온 경로를 거꾸로 따라가",
    "방금까지 이동한 경로를 되짚어 돌아가",
    "왔던 경로 그대로 출발 방향으로 돌아가",
    "이동 경로를 역순으로 복귀해"
]

TRAIN_FORWARD=[
    "앞으로 {dist}미터 전진해",
    "{dist}m 앞으로 가줘",
    "전방으로 {dist}m 이동해"
]

TRAIN_BACK=[
    "뒤로 {dist}m 물러나 줘",
    "{dist}미터 후진해",
    "후방으로 {dist}m 이동해"
]

TRAIN_RIGHT=[
    "오른쪽으로 {dist}미터 가줘",
    "우측으로 {dist}m 이동해",
    "우측으로 {dist}m 가라"
]

TRAIN_LEFT=[
    "왼쪽으로 {dist}m 이동해",
    "좌측으로 {dist}미터 가줘",
    "좌측으로 {dist}m 가라"
]

TRAIN_UP=[
    "위로 {dist}미터 더 올라가",
    "{dist}m 상승 해줘",
    "고도를 {dist}m 높여라"
]

TRAIN_DOWN=[
    "아래로 {dist}m 내려가",
    "{dist}미터 하강 해줘",
    "고도를 {dist}m 낮춰라"
]

TRAIN_CW=[
    "오른쪽으로 {angle}도 회전해",
    "시계 방향으로 {angle}도 돌아",
    "오른쪽으로 {angle}도 돌려줘"
]

TRAIN_CCW=[
    "왼쪽으로 {angle}도 회전해",
    "반시계 방향으로 {angle}도 돌아",
    "왼쪽으로 {angle}도 돌려줘"
]


# ============================================================
# Validation 문장 패턴
# ============================================================

VAL_TAKEOFF=[
    "{alt}m까지 드론을 띄워",
    "드론을 {alt}미터 고도로 올려",
    "목표 고도를 {alt}m로 해서 이륙해",
    "{alt}m 상공까지 올라가서 이륙 상태를 유지해"
]

VAL_LAND=[
    "현재 자리에서 내려서 착륙해",
    "드론을 지상에 내려줘",
    "지금 있는 곳에서 착륙 절차를 시작해",
    "비행을 끝내고 지면으로 내려가"
]

VAL_GOTO_PREVIOUS=[
    "한 단계 전 위치로 돌아가",
    "방금 전 자리로 되돌아가",
    "직전에 있던 곳으로 이동해",
    "바로 전 위치를 찾아 돌아가"
]

VAL_GOTO_FIRST=[
    "비행을 시작했던 자리로 돌아가",
    "출발 지점으로 되돌아가",
    "처음 시작한 곳으로 복귀해",
    "비행 시작점으로 돌아와"
]

VAL_REVERSE=[
    "지금까지 이동한 순서의 반대로 복귀해",
    "지나온 이동을 반대로 수행해서 돌아가",
    "현재까지의 이동 경로를 거꾸로 되짚어가",
    "이동했던 순서를 뒤집어서 출발점 방향으로 가"
]

VAL_FORWARD=[
    "기체를 앞쪽으로 {dist}m 보내",
    "정면 방향으로 {dist}m 나아가",
    "전방을 향해 {dist}m 움직여",
    "앞 방향으로 {dist}미터 이동해"
]

VAL_BACK=[
    "기체를 뒤쪽으로 {dist}m 보내",
    "후방으로 {dist}m 물러서",
    "뒤쪽 방향으로 {dist}미터 움직여",
    "진행 방향 반대로 {dist}m 이동해"
]

VAL_RIGHT=[
    "기체를 오른편으로 {dist}m 옮겨",
    "오른편으로 {dist}m 움직여",
    "우측 방향으로 {dist}미터 보내",
    "기체를 오른쪽 측면으로 {dist}m 이동해"
]

VAL_LEFT=[
    "기체를 왼편으로 {dist}m 옮겨",
    "왼편으로 {dist}m 움직여",
    "좌측 방향으로 {dist}미터 보내",
    "기체를 왼쪽 측면으로 {dist}m 이동해"
]

VAL_UP=[
    "기체를 {dist}m 위로 올려",
    "현재보다 {dist}m 더 높은 곳으로 가",
    "상대적으로 {dist}m 상승해",
    "드론을 위쪽으로 {dist}m 이동시켜"
]

VAL_DOWN=[
    "기체를 {dist}m 아래로 내려",
    "현재보다 {dist}m 낮은 곳으로 가",
    "상대적으로 {dist}m 하강해",
    "드론을 아래쪽으로 {dist}m 이동시켜"
]

VAL_CW=[
    "기체를 오른쪽으로 {angle}도 틀어",
    "진행 방향을 시계 방향으로 {angle}도 바꿔",
    "우측 방향으로 {angle}도 방향을 전환해",
    "오른쪽으로 {angle}도 방향을 돌려"
]

VAL_CCW=[
    "기체를 왼쪽으로 {angle}도 틀어",
    "진행 방향을 반시계 방향으로 {angle}도 바꿔",
    "좌측 방향으로 {angle}도 방향을 전환해",
    "왼쪽으로 {angle}도 방향을 돌려"
]


# ============================================================
# Test 문장 패턴
# ============================================================

TEST_TAKEOFF=[
    "드론을 {alt}m 상공으로 올려줘",
    "공중으로 {alt}미터까지 띄워",
    "지면에서 {alt}m 높이까지 수직으로 올라가",
    "비행 고도를 {alt}m로 설정하고 떠올라"
]

TEST_LAND=[
    "비행을 멈추고 아래로 내려 착지해",
    "기체를 지면까지 내려보내",
    "현재 좌표에서 지상으로 내려앉아",
    "드론을 안전하게 지상에 내려줘"
]

TEST_GOTO_PREVIOUS=[
    "직전 이동이 시작되기 전 자리로 돌아가",
    "조금 전에 머물렀던 위치로 되돌아가",
    "마지막 이동 이전의 장소를 찾아가",
    "바로 앞서 있던 지점으로 복귀해"
]

TEST_GOTO_FIRST=[
    "비행을 시작한 최초 지점으로 되돌아가",
    "처음 출발했던 장소를 찾아가",
    "맨 처음 있던 위치로 돌아와",
    "비행 시작 위치로 다시 이동해"
]

TEST_REVERSE=[
    "지금까지 지나온 길을 그대로 거꾸로 이동해",
    "현재까지의 이동을 반대 순서로 재현해서 돌아가",
    "지나온 경로를 처음부터 역방향으로 되짚어가",
    "지금까지 움직인 길을 반대로 따라 출발한 곳으로 가"
]

TEST_FORWARD=[
    "기수를 향한 방향으로 {dist}m 더 나아가",
    "드론을 정면 쪽으로 {dist}미터 전진시켜",
    "현재 바라보는 방향을 따라 {dist}m 이동해",
    "기체 앞쪽으로 {dist}m만큼 보내줘"
]

TEST_BACK=[
    "기체를 진행 반대쪽으로 {dist}m 후퇴시켜",
    "현재 방향의 뒤쪽으로 {dist}미터 이동해",
    "드론을 후진시켜 {dist}m 물러나",
    "기체를 뒤편으로 {dist}m 옮겨줘"
]

TEST_RIGHT=[
    "기체의 우현 쪽으로 {dist}m 이동시켜",
    "현재 기체 기준 오른편으로 {dist}미터 옮겨",
    "드론을 우측 측면으로 {dist}m 보내줘",
    "오른쪽 측면 방향으로 {dist}m 이동해"
]

TEST_LEFT=[
    "기체의 좌현 쪽으로 {dist}m 이동시켜",
    "현재 기체 기준 왼편으로 {dist}미터 옮겨",
    "드론을 좌측 측면으로 {dist}m 보내줘",
    "왼쪽 측면 방향으로 {dist}m 이동해"
]

TEST_UP=[
    "드론을 현재보다 {dist}m 높여줘",
    "기체를 상공 방향으로 {dist}미터 이동해",
    "현재 고도에서 {dist}m만큼 더 상승시켜",
    "위쪽으로 {dist}m 위치를 변경해"
]

TEST_DOWN=[
    "드론을 현재보다 {dist}m 낮춰줘",
    "기체를 지면 쪽으로 {dist}미터 이동해",
    "현재 고도에서 {dist}m만큼 내려가",
    "아래 방향으로 {dist}m 위치를 변경해"
]

TEST_CW=[
    "기체의 방향을 오른쪽으로 {angle}도 꺾어",
    "현재 방향에서 시계 방향으로 {angle}도 틀어",
    "기수 방향을 우측으로 {angle}도 전환해",
    "드론의 진행 방향을 오른쪽으로 {angle}도 돌려"
]

TEST_CCW=[
    "기체의 방향을 왼쪽으로 {angle}도 꺾어",
    "현재 방향에서 반시계 방향으로 {angle}도 틀어",
    "기수 방향을 좌측으로 {angle}도 전환해",
    "드론의 진행 방향을 왼쪽으로 {angle}도 돌려"
]


# ============================================================
# 데이터 개수 = NUM * 7
# ============================================================

NUM_TRAIN=800
NUM_VAL=100
NUM_TEST=100
OUTPUT_PATH="dataset"


# ============================================================
# 데이터 형식
# ============================================================

def make_sample(user_cmd:str,func_name:str,args:dict)->dict:
    tool_call=json.dumps(
        {"name":func_name,"arguments":args},
        ensure_ascii=False
    )
    return {
        "messages":[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":user_cmd},
            {"role":"assistant","content":f"<tool_call>{tool_call}</tool_call>"}
        ]
    }


# ============================================================
# Move generator
# ============================================================

def make_move_generators(
    forward,back,right,left,up,down,cw,ccw
):
    return [
        lambda d:(random.choice(forward).format(dist=d),
                  {"dx":d,"dy":0.0,"dz":0.0,"d_yaw":0.0}),
        lambda d:(random.choice(back).format(dist=d),
                  {"dx":-d,"dy":0.0,"dz":0.0,"d_yaw":0.0}),
        lambda d:(random.choice(right).format(dist=d),
                  {"dx":0.0,"dy":-d,"dz":0.0,"d_yaw":0.0}),
        lambda d:(random.choice(left).format(dist=d),
                  {"dx":0.0,"dy":d,"dz":0.0,"d_yaw":0.0}),
        lambda d:(random.choice(up).format(dist=d),
                  {"dx":0.0,"dy":0.0,"dz":d,"d_yaw":0.0}),
        lambda d:(random.choice(down).format(dist=d),
                  {"dx":0.0,"dy":0.0,"dz":-d,"d_yaw":0.0}),
        lambda a:(random.choice(cw).format(angle=a),
                  {"dx":0.0,"dy":0.0,"dz":0.0,"d_yaw":-a}),
        lambda a:(random.choice(ccw).format(angle=a),
                  {"dx":0.0,"dy":0.0,"dz":0.0,"d_yaw":a})
    ]


# ============================================================
# 기본 명령 생성
# ============================================================

def generate_basic_dataset(num_samples,phrases):
    dataset=[]

    move_generators=make_move_generators(
        phrases["forward"],
        phrases["back"],
        phrases["right"],
        phrases["left"],
        phrases["up"],
        phrases["down"],
        phrases["cw"],
        phrases["ccw"]
    )

    for _ in range(num_samples):
        alt=round(random.uniform(1.0,5.0),1)
        dataset.append(make_sample(
            random.choice(phrases["takeoff"]).format(alt=alt),
            "takeoff",
            {"altitude":alt}
        ))

        dataset.append(make_sample(
            random.choice(phrases["land"]),
            "land",
            {}
        ))

        dataset.append(make_sample(
            random.choice(phrases["previous"]),
            "goto_history",
            {"recall":"previous"}
        ))

        dataset.append(make_sample(
            random.choice(phrases["first"]),
            "goto_history",
            {"recall":"first"}
        ))

        dataset.append(make_sample(
            random.choice(phrases["reverse"]),
            "reverse_plan",
            {}
        ))

        for _ in range(2):
            if random.random()<0.25:
                angle=round(random.uniform(15.0,180.0),1)
                move_cmd,args=random.choice(move_generators[6:])(angle)
            else:
                dist=round(random.uniform(0.5,5.0),1)
                move_cmd,args=random.choice(move_generators[:6])(dist)

            dataset.append(make_sample(move_cmd,"move",args))

    random.shuffle(dataset)
    return dataset


# ============================================================
# Phrase 묶음
# ============================================================

TRAIN_PHRASES={
    "takeoff":TRAIN_TAKEOFF,
    "land":TRAIN_LAND,
    "previous":TRAIN_GOTO_PREVIOUS,
    "first":TRAIN_GOTO_FIRST,
    "reverse":TRAIN_REVERSE,
    "forward":TRAIN_FORWARD,
    "back":TRAIN_BACK,
    "right":TRAIN_RIGHT,
    "left":TRAIN_LEFT,
    "up":TRAIN_UP,
    "down":TRAIN_DOWN,
    "cw":TRAIN_CW,
    "ccw":TRAIN_CCW
}

VAL_PHRASES={
    "takeoff":VAL_TAKEOFF,
    "land":VAL_LAND,
    "previous":VAL_GOTO_PREVIOUS,
    "first":VAL_GOTO_FIRST,
    "reverse":VAL_REVERSE,
    "forward":VAL_FORWARD,
    "back":VAL_BACK,
    "right":VAL_RIGHT,
    "left":VAL_LEFT,
    "up":VAL_UP,
    "down":VAL_DOWN,
    "cw":VAL_CW,
    "ccw":VAL_CCW
}

TEST_PHRASES={
    "takeoff":TEST_TAKEOFF,
    "land":TEST_LAND,
    "previous":TEST_GOTO_PREVIOUS,
    "first":TEST_GOTO_FIRST,
    "reverse":TEST_REVERSE,
    "forward":TEST_FORWARD,
    "back":TEST_BACK,
    "right":TEST_RIGHT,
    "left":TEST_LEFT,
    "up":TEST_UP,
    "down":TEST_DOWN,
    "cw":TEST_CW,
    "ccw":TEST_CCW
}


# ============================================================
# 저장
# ============================================================

def save_jsonl(dataset,output_file):
    output_path=Path(OUTPUT_PATH)/output_file
    output_path.parent.mkdir(parents=True,exist_ok=True)

    with open(output_path,"w",encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item,ensure_ascii=False)+"\n")

    print(f"📄 저장 위치: {output_path.resolve()}")


# ============================================================
# 실행
# ============================================================

if __name__=="__main__":
    train_dataset=generate_basic_dataset(NUM_TRAIN,TRAIN_PHRASES)
    val_dataset=generate_basic_dataset(NUM_VAL,VAL_PHRASES)
    test_dataset=generate_basic_dataset(NUM_TEST,TEST_PHRASES)

    save_jsonl(train_dataset,"train.jsonl")
    save_jsonl(val_dataset,"val.jsonl")
    save_jsonl(test_dataset,"test.jsonl")

    print(f"\n✅ Train: {len(train_dataset)}개")
    print(f"🧪 Validation: {len(val_dataset)}개")
    print(f"📝 Test: {len(test_dataset)}개")
    print(f"📦 전체: {len(train_dataset)+len(val_dataset)+len(test_dataset)}개")