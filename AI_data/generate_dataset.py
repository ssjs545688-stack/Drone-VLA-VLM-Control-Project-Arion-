#!/usr/bin/env python3
"""
Qwen3-0.6B 드론 제어 Tool Calling 파인튜닝용
train/validation/test 데이터셋 자동 생성 스크립트
"""

import json
import random
from collections import Counter
from pathlib import Path
from llm_drone_control.schema import SYSTEM_PROMPT


# ============================================================
# 설정값
# ============================================================

# 데이터 수 = NUM_TRAIN * (5 + MOVES_PER_LOOP) + NUM_COMPOUND + NUM_NEGATIVE
NUM_TRAIN = 70
NUM_VAL = 15
NUM_TEST = 30

MOVES_PER_LOOP = 4    # 루프 당 move 수
ROTATE_PROB = 0.3     # move 중 회전 비율

DIST_RANGE_M = (0.3, 8.0)     # 병진 이동 거리(미터 단위로 말할 때) 범위
DIST_RANGE_CM = (20, 500)     # 병진 이동 거리(센티미터 단위로 말할 때) 범위, 정수
ANGLE_RANGE = (5, 180)        # 회전 각도 범위(정수, 도)
ALT_RANGE = (1.0, 5.0)        # 이륙 고도 범위

NUM_COMPOUND = {"train": 30, "val": 6, "test": 6}   # 복합 명령
NUM_NEGATIVE = {"train": 20, "val": 4, "test": 4}   # 엉뚱한 질문

OUTPUT_PATH = "dataset"


# ============================================================
# 한글 수사(sino-Korean) 변환 유틸
# ============================================================

_SINO_DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
_SINO_UNITS = ["", "십", "백", "천"]  # 최대 9999까지. 이 스크립트의 값 범위(<1000)엔 충분.


def _int_to_sino_korean(num: int) -> str:
    if num == 0:
        return "영"
    result = ""
    digits = str(num)
    length = len(digits)
    for i, ch in enumerate(digits):
        d = int(ch)
        unit_idx = length - i - 1
        if d == 0:
            continue
        if d == 1 and unit_idx > 0:
            # "일십", "일백"이 아니라 "십", "백"으로 읽는 관례
            result += _SINO_UNITS[unit_idx] if unit_idx < len(_SINO_UNITS) else str(d)
        else:
            unit = _SINO_UNITS[unit_idx] if unit_idx < len(_SINO_UNITS) else ""
            result += _SINO_DIGITS[d] + unit
    return result


def number_to_korean(num) -> str:
    """3.2 -> '삼점이', 30 -> '삼십', 0.5 -> '영점오' 같은 한글 숫자 읽기로 변환.
    실제 ASR 전사본에 종종 등장하는 '한글 수사' 표기 노이즈를 재현하기 위함."""
    if isinstance(num, int) or float(num) == int(num):
        return _int_to_sino_korean(int(num))
    int_part = int(num)
    dec_str = f"{num:.1f}".split(".")[1]
    int_kor = _int_to_sino_korean(int_part) if int_part != 0 else "영"
    dec_kor = "".join(_SINO_DIGITS[int(d)] if d != "0" else "영" for d in dec_str)
    return f"{int_kor}점{dec_kor}"


# ============================================================
# train/val용 노이즈 주입 (test 전용 노이즈였던 것을 학습 가능하게 이전)
# ============================================================

COMMON_TYPO_MAP = {
    "이동해": "이동해줘",
    "돌아가": "돌아가라",
    "복귀해": "복기해",
    "고도를": "고도을",
    "높여라": "높혀라",
    "낮춰라": "낮춰라",
    "회전해": "회잔해",
    "전진해": "전진해라",
    "후진해": "후진해줘",
}


def inject_typo(text: str, prob: float) -> str:
    if random.random() >= prob:
        return text
    candidates = [k for k in COMMON_TYPO_MAP if k in text]
    if not candidates:
        return text
    key = random.choice(candidates)
    return text.replace(key, COMMON_TYPO_MAP[key], 1)


def inject_spacing_noise(text: str, prob: float) -> str:
    """공백 기준 인접 토큰 두 개를 붙여써서 '붙여쓰기' 오류를 재현."""
    if random.random() >= prob:
        return text
    tokens = text.split(" ")
    if len(tokens) < 2:
        return text
    idx = random.randrange(len(tokens) - 1)
    tokens[idx] = tokens[idx] + tokens[idx + 1]
    del tokens[idx + 1]
    return " ".join(tokens)


def apply_train_val_noise(text: str, typo_prob: float, spacing_prob: float) -> str:
    text = inject_typo(text, typo_prob)
    text = inject_spacing_noise(text, spacing_prob)
    return text


# ============================================================
# 샘플 빌더
# ============================================================

def make_sample(user_cmd: str, func_name: str, args: dict) -> dict:
    """단일 tool_call 샘플"""
    tool_call = json.dumps({"name": func_name, "arguments": args}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_cmd},
            {"role": "assistant", "content": f"<tool_call>{tool_call}</tool_call>"},
        ]
    }


def make_multi_call_sample(user_cmd: str, calls: list) -> dict:
    """복합 명령: calls = [(func_name, args), (func_name, args), ...]
    assistant 응답에 tool_call 블록을 순서대로 이어붙인다."""
    blocks = "\n".join(
        f'<tool_call>{json.dumps({"name": name, "arguments": args}, ensure_ascii=False)}</tool_call>'
        for name, args in calls
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_cmd},
            {"role": "assistant", "content": blocks},
        ]
    }


def make_refusal_sample(user_cmd: str, refusal_text: str) -> dict:
    """스코프 밖 요청: tool_call 없이 평문으로 응답"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_cmd},
            {"role": "assistant", "content": refusal_text},
        ]
    }


# ============================================================
# 방향/좌표 정의 (REP-103 FLU 바디 프레임: x-전방, y-좌측, z-상방)
# ============================================================

TRANSLATE_AXES = {
    "forward": (1, 0, 0),
    "back": (-1, 0, 0),
    "right": (0, -1, 0),
    "left": (0, 1, 0),
    "up": (0, 0, 1),
    "down": (0, 0, -1),
}


def m(*templates):
    """미터 단위로 말하는 템플릿들을 (템플릿, 'm') 튜플로 감싼다."""
    return [(t, "m") for t in templates]


def cm(*templates):
    """센티미터 단위로 말하는 템플릿들을 (템플릿, 'cm') 튜플로 감싼다."""
    return [(t, "cm") for t in templates]


def gen_distance_value(unit: str):
    """unit에 맞는 자연스러운 범위에서 값을 생성하고, 항상 '미터'로 환산한 값도 함께 반환.
    반환: (표시용 원본 값, 실제 인자로 쓸 미터 값)"""
    if unit == "cm":
        raw = int(round(random.uniform(*DIST_RANGE_CM)))
        meters = raw / 100.0
    else:
        raw = round(random.uniform(*DIST_RANGE_M), 1)
        meters = raw
    return raw, meters


def make_translate_sample(direction: str, unit_templates: list):
    template, unit = random.choice(unit_templates)
    raw, meters = gen_distance_value(unit)
    text = template.format(dist=raw, dist_kor=number_to_korean(raw))
    sx, sy, sz = TRANSLATE_AXES[direction]
    args = {"dx": sx * meters, "dy": sy * meters, "dz": sz * meters, "d_yaw": 0.0}
    return text, args


def make_rotate_sample(direction: str, templates: list):
    angle = int(round(random.uniform(*ANGLE_RANGE)))
    template = random.choice(templates)
    # 모든 템플릿에 dist/dist_kor/angle/angle_kor/alt/alt_kor를 전부 넘겨서
    # 어떤 플레이스홀더를 쓰든(super-set kwargs) KeyError가 나지 않게 한다.
    text = template.format(angle=angle, angle_kor=number_to_korean(angle))
    sign = -1 if direction == "cw" else 1  # 시계방향(오른쪽 회전) = 음수 yaw
    args = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "d_yaw": sign * angle}
    return text, args


# ============================================================
# Train 문장 패턴
# ============================================================

TRAIN_TAKEOFF = [
    "드론 고도 {alt}m로 띄워줘",
    "{alt}미터 높이로 이륙해",
    "지면에서 {alt}m 상승해서 떠라",
    "드론 높이 {alt}m까지 수직 이륙해",
    "고도 {alt}m로 이륙해",
    "고도 {alt_kor}미터로 이륙해",       # 한글 수사 노출 (train)
]

TRAIN_LAND = [
    "이제 바닥으로 안전하게 착륙해",
    "지금 위치에 착륙해줘",
    "착륙해",
    "지면으로 착륙해라",
    "착륙하여 비행 종료해",
    "드론을 착륙시켜 지면에 둬",
]

TRAIN_GOTO_PREVIOUS = [
    "바로 이전 위치로 돌아가",
    "직전 위치로 복귀해줘",
    "방금 이동하기 전 위치로 돌아가",
    "이전 위치로 되돌아가",
    "돌아서 직전 위치로 가",
    "바로 전 위치로 복귀해",
]

TRAIN_GOTO_FIRST = [
    "처음 위치로 직선으로 돌아가",
    "복귀해서 최초 출발지로 와",
    "처음 이륙했던 위치로 일직선으로 돌아가",
    "원점으로 바로 돌아가",
    "홈 위치로 복귀해",
    "돌아서 출발했던 위치로 와",
]

TRAIN_REVERSE = [
    "왔던 길 그대로 되돌아가줘",
    "이동했던 경로를 역순으로 돌아가",
    "지금까지 온 경로를 거꾸로 따라가",
    "방금까지 이동한 경로를 되짚어 돌아가",
    "왔던 경로 그대로 출발 방향으로 돌아가",
    "역순으로 이동 경로를 복귀해",
]

TRAIN_FORWARD = m(
    "앞으로 {dist}미터 전진해",
    "{dist}m 앞으로 가줘",
    "전방으로 {dist}m 이동해",
    "지금 바라보는 방향으로 {dist}미터 가라",
    "정면으로 {dist}m만 나아가",
    "앞으로 {dist_kor}미터 가",           # 한글 수사 노출 (train)
) + cm(
    "앞으로 {dist}cm만 가줘",
)

TRAIN_BACK = m(
    "뒤로 {dist}m 물러나 줘",
    "{dist}미터 후진해",
    "후방으로 {dist}m 이동해",
    "{dist}미터 지금 바라보는 반대쪽으로 가라",
    "뒤로 {dist_kor}미터 물러나",          # 한글 수사 노출 (train)
) + cm(
    "뒤로 {dist}cm만 물러나",
)

TRAIN_RIGHT = m(
    "오른쪽으로 {dist}미터 가줘",
    "{dist}m 우측으로 이동해",
    "우측으로 {dist}m 가라",
    "{dist}미터 오른쪽으로 움직여",
    "오른쪽으로 {dist_kor}미터 가",        # 한글 수사 노출 (train)
) + cm(
    "오른쪽으로 {dist}cm만 이동해",
)

TRAIN_LEFT = m(
    "왼쪽으로 {dist}m 이동해",
    "{dist}미터 좌측으로 가줘",
    "좌측으로 {dist}m 가라",
    "{dist}미터 왼쪽으로 움직여",
    "왼쪽으로 {dist_kor}미터 가",          # 한글 수사 노출 (train)
) + cm(
    "왼쪽으로 {dist}cm만 이동해",
)

TRAIN_UP = m(
    "{dist}미터 더 올라가",
    "{dist}m 상승 해줘",
    "고도를 {dist}m 높여라",
    "위쪽으로 {dist}미터 이동해",
    "{dist_kor}미터 더 올라가",           # 한글 수사 노출 (train)
) + cm(
    "{dist}cm만 더 올라가",
)

TRAIN_DOWN = m(
    "아래로 {dist}m 내려가",
    "{dist}미터 하강 해줘",
    "고도를 {dist}m 낮춰라",
    "{dist}미터 밑으로 이동해",
    "아래로 {dist_kor}미터 내려가",        # 한글 수사 노출 (train)
) + cm(
    "{dist}cm만 내려가",
)

TRAIN_CW = [
    "오른쪽으로 {angle}도 회전해",
    "시계 방향으로 {angle}도 돌아",
    "{angle}도 오른쪽으로 돌려줘",
    "우측으로 {angle}도 회전해줘",
    "오른쪽으로 {angle_kor}도 회전해",     # 한글 수사 노출 (train)
]

TRAIN_CCW = [
    "왼쪽으로 {angle}도 회전해",
    "반시계 방향으로 {angle}도 돌아",
    "{angle}도 왼쪽으로 돌려줘",
    "좌측으로 {angle}도 회전해줘",
    "왼쪽으로 {angle_kor}도 회전해",       # 한글 수사 노출 (train)
]


# ============================================================
# Validation 문장 패턴
# ============================================================

VAL_TAKEOFF = [
    "{alt}m까지 드론을 띄워",
    "드론을 {alt}미터 고도로 올려",
    "목표 고도를 {alt}m로 해서 이륙해",
    "{alt}m 상공까지 올라가서 이륙 상태를 유지해",
    "목표 고도 {alt_kor}미터로 이륙해",    # 한글 수사 노출 (val)
]

VAL_LAND = [
    "현재 자리에서 내려서 착륙해",
    "드론을 지상에 내려줘",
    "지금 있는 곳에서 착륙 절차를 시작해",
    "비행을 끝내고 지면으로 내려가",
]

VAL_GOTO_PREVIOUS = [
    "한 단계 전 위치로 돌아가",
    "방금 전 자리로 되돌아가",
    "직전에 있던 곳으로 이동해",
    "바로 전 위치를 찾아 돌아가",
]

VAL_GOTO_FIRST = [
    "비행을 시작했던 자리로 돌아가",
    "출발 지점으로 되돌아가",
    "처음 시작한 곳으로 복귀해",
    "비행 시작점으로 돌아와",
]

VAL_REVERSE = [
    "지금까지 이동한 순서의 반대로 복귀해",
    "지나온 이동을 반대로 수행해서 돌아가",
    "현재까지의 이동 경로를 거꾸로 되짚어가",
    "이동했던 순서를 뒤집어서 출발점 방향으로 가",
]

VAL_FORWARD = m(
    "기체를 앞쪽으로 {dist}m 보내",
    "정면 방향으로 {dist}m 나아가",
    "전방을 향해 {dist}m 움직여",
    "앞 방향으로 {dist}미터 이동해",
    "정면으로 {dist_kor}미터 나아가",      # 한글 수사 노출 (val)
) + cm(
    "앞쪽으로 {dist}cm 옮겨",
)

VAL_BACK = m(
    "기체를 뒤쪽으로 {dist}m 보내",
    "후방으로 {dist}m 물러서",
    "뒤쪽 방향으로 {dist}미터 움직여",
    "진행 방향 반대로 {dist}m 이동해",
    "후방으로 {dist_kor}미터 물러서",      # 한글 수사 노출 (val)
) + cm(
    "뒤쪽으로 {dist}cm 옮겨",
)

VAL_RIGHT = m(
    "기체를 오른편으로 {dist}m 옮겨",
    "오른편으로 {dist}m 움직여",
    "우측 방향으로 {dist}미터 보내",
    "기체를 오른쪽 측면으로 {dist}m 이동해",
    "우측 방향으로 {dist_kor}미터 보내",   # 한글 수사 노출 (val)
) + cm(
    "오른편으로 {dist}cm 옮겨",
)

VAL_LEFT = m(
    "기체를 왼편으로 {dist}m 옮겨",
    "왼편으로 {dist}m 움직여",
    "좌측 방향으로 {dist}미터 보내",
    "기체를 왼쪽 측면으로 {dist}m 이동해",
    "좌측 방향으로 {dist_kor}미터 보내",   # 한글 수사 노출 (val)
) + cm(
    "왼편으로 {dist}cm 옮겨",
)

VAL_UP = m(
    "기체를 {dist}m 위로 올려",
    "현재보다 {dist}m 더 높은 곳으로 가",
    "상대적으로 {dist}m 상승해",
    "드론을 위쪽으로 {dist}m 이동시켜",
    "상대적으로 {dist_kor}미터 상승해",    # 한글 수사 노출 (val)
) + cm(
    "{dist}cm 위로 올려",
)

VAL_DOWN = m(
    "기체를 {dist}m 아래로 내려",
    "현재보다 {dist}m 낮은 곳으로 가",
    "상대적으로 {dist}m 하강해",
    "드론을 아래쪽으로 {dist}m 이동시켜",
    "상대적으로 {dist_kor}미터 하강해",    # 한글 수사 노출 (val)
) + cm(
    "{dist}cm 아래로 내려",
)

VAL_CW = [
    "기체를 오른쪽으로 {angle}도 틀어",
    "진행 방향을 시계 방향으로 {angle}도 바꿔",
    "우측 방향으로 {angle}도 방향을 전환해",
    "오른쪽으로 {angle}도 방향을 돌려",
    "우측 방향으로 {angle_kor}도 방향을 전환해",  # 한글 수사 노출 (val)
]

VAL_CCW = [
    "기체를 왼쪽으로 {angle}도 틀어",
    "진행 방향을 반시계 방향으로 {angle}도 바꿔",
    "좌측 방향으로 {angle}도 방향을 전환해",
    "왼쪽으로 {angle}도 방향을 돌려",
    "좌측 방향으로 {angle_kor}도 방향을 전환해",  # 한글 수사 노출 (val)
]


# ============================================================
# Test 문장 패턴 (오탈자, 띄어쓰기 오류, 한글 수사, 단위 변형 노이즈)
# ============================================================

TEST_TAKEOFF = [
    "드론 {alt}미타높이로 띄어줘",                      # 오탈자 & 붙여쓰기
    "공중으로{alt}m까지띄워라",                          # 띄어쓰기 없음
    "지면에서 {alt_kor}미터 높이까지 수직으로 업",       # 한글 수사 (실제 구현됨)
    "비행고도 {alt}m 설정 후 떠오르기",                  # 변칙적 어미
]

TEST_LAND = [
    "비행 멈추구 아래로 착지해라",
    "기체를지상까지착륙시켜",
    "현위치에서 땅으로 내려앉아라",
    "드론 착류캐줘",
]

TEST_GOTO_PREVIOUS = [
    "직전위치로 빽해줘",
    "조금전에 머물렀던데로 돌아가",
    "방금전 위치 찾아가라",
    "바로 앞서있던 지점 복귀",
]

TEST_GOTO_FIRST = [
    "처음있던데로 빽해",
    "처음출발지점으로 되돌아가줘",
    "맨처음 장소로 컴백해라",
    "비행 시작지점 복귀",
]

TEST_REVERSE = [
    "지나온길 그대로 리버스해줘",
    "지금까지 이동 반대로 재현해서 돌아가",
    "왔던경로 역순으로 빽",
    "지나온 길 거꾸로 되짚어가라",
]

TEST_FORWARD = cm(
    "앞으로 {dist}센치 이동해",
) + m(
    "정면쪽으로 {dist_kor}미터 가라",
    "앞으로{dist}m전진",
) + cm(
    "기수방향 {dist}cm 갓",
)

TEST_BACK = cm(
    "뒤로 {dist}센티미터 후진해",
) + m(
    "후방으로 {dist_kor}미터 물러나",
    "뒤쪽으로{dist}m가줘",
) + cm(
    "진행반대방향 {dist}cm 퇴각",
)

TEST_RIGHT = cm(
    "오른쪽으로 {dist}cm 움직여",
) + m(
    "우측으로 {dist_kor}미터 보내기",
    "오른편으로{dist}미터이동",
) + cm(
    "우현 쪽으로 {dist}cm 틀어서 가라",
)

TEST_LEFT = cm(
    "왼쪽으로 {dist}cm 가줘",
) + m(
    "좌측으로 {dist_kor}미터 이동해",
    "왼편으로{dist}m가라",
) + cm(
    "좌현 방향으로 {dist}cm 이동",
)

TEST_UP = cm(
    "위로 {dist}센티 올려",
) + m(
    "상공으로 {dist_kor}미터 상승해",
) + cm(
    "위쪽으로{dist}cm업",
    "고도 {dist}cm 올려라",
)

TEST_DOWN = cm(
    "아래로 {dist}cm 다운해",
) + m(
    "지면쪽으로 {dist_kor}미터 하강",
) + cm(
    "아래쪽으로{dist}cm내려가",
    "고도 {dist}cm 낮춰",
)

TEST_CW = [
    "시계방향으로 {angle_kor}도 틀어줘",     # 한글 수사 (실제 구현됨)
    "오른쪽으로{angle}도회전",
    "우측으로 {angle}도 꺾어라",
    "시계방향 {angle}도 턴",
]

TEST_CCW = [
    "반시계방향으로 {angle_kor}도 틀어줘",
    "왼쪽으로{angle}도회전",
    "좌측으로 {angle}도 꺾어라",
    "반시계 {angle}도 턴해",
]


# ============================================================
# Phrase 묶음
# ============================================================

TRAIN_PHRASES = {
    "takeoff": TRAIN_TAKEOFF, "land": TRAIN_LAND,
    "previous": TRAIN_GOTO_PREVIOUS, "first": TRAIN_GOTO_FIRST, "reverse": TRAIN_REVERSE,
    "forward": TRAIN_FORWARD, "back": TRAIN_BACK, "right": TRAIN_RIGHT, "left": TRAIN_LEFT,
    "up": TRAIN_UP, "down": TRAIN_DOWN, "cw": TRAIN_CW, "ccw": TRAIN_CCW,
}

VAL_PHRASES = {
    "takeoff": VAL_TAKEOFF, "land": VAL_LAND,
    "previous": VAL_GOTO_PREVIOUS, "first": VAL_GOTO_FIRST, "reverse": VAL_REVERSE,
    "forward": VAL_FORWARD, "back": VAL_BACK, "right": VAL_RIGHT, "left": VAL_LEFT,
    "up": VAL_UP, "down": VAL_DOWN, "cw": VAL_CW, "ccw": VAL_CCW,
}

TEST_PHRASES = {
    "takeoff": TEST_TAKEOFF, "land": TEST_LAND,
    "previous": TEST_GOTO_PREVIOUS, "first": TEST_GOTO_FIRST, "reverse": TEST_REVERSE,
    "forward": TEST_FORWARD, "back": TEST_BACK, "right": TEST_RIGHT, "left": TEST_LEFT,
    "up": TEST_UP, "down": TEST_DOWN, "cw": TEST_CW, "ccw": TEST_CCW,
}


# ============================================================
# 기본 명령 생성 (이착륙/히스토리/이동/회전)
# ============================================================

def generate_basic_dataset(
    num_loops: int,
    phrases: dict,
    typo_prob: float = 0.0,
    spacing_prob: float = 0.0,
) -> list:
    """typo_prob/spacing_prob > 0이면 생성된 문장에 일반화된 오타/붙여쓰기
    노이즈를 후처리로 씌운다. train/val에 사용해 해당 패턴을 실제로
    학습하게 하기 위함 — test는 자체 노이즈 템플릿이 있으므로 보통 0으로 둔다."""
    dataset = []
    translate_dirs = ["forward", "back", "right", "left", "up", "down"]

    def noisy(text: str) -> str:
        return apply_train_val_noise(text, typo_prob, spacing_prob)

    for _ in range(num_loops):
        alt = round(random.uniform(*ALT_RANGE), 1)
        text = random.choice(phrases["takeoff"]).format(alt=alt, alt_kor=number_to_korean(alt))
        dataset.append(make_sample(noisy(text), "takeoff", {"altitude": alt}))

        dataset.append(make_sample(noisy(random.choice(phrases["land"])), "land", {}))
        dataset.append(make_sample(noisy(random.choice(phrases["previous"])), "goto_history", {"recall": "previous"}))
        dataset.append(make_sample(noisy(random.choice(phrases["first"])), "goto_history", {"recall": "first"}))
        dataset.append(make_sample(noisy(random.choice(phrases["reverse"])), "reverse_plan", {}))

        for _ in range(MOVES_PER_LOOP):
            if random.random() < ROTATE_PROB:
                direction = random.choice(["cw", "ccw"])
                text, args = make_rotate_sample(direction, phrases[direction])
            else:
                direction = random.choice(translate_dirs)
                text, args = make_translate_sample(direction, phrases[direction])
            dataset.append(make_sample(noisy(text), "move", args))

    random.shuffle(dataset)
    return dataset


# ============================================================
# 복합 명령 (한 문장에 tool_call 2개)
# ============================================================

COMPOUND_PATTERNS = [
    {
        "vars": ["dist"],
        "template": "{dist}미터 앞으로 가고 착륙해",
        "build": lambda v: [
            ("move", {"dx": v["dist"], "dy": 0.0, "dz": 0.0, "d_yaw": 0.0}),
            ("land", {}),
        ],
    },
    {
        "vars": ["alt", "angle"],
        "template": "고도 {alt}m로 이륙한 다음 오른쪽으로 {angle}도 회전해",
        "build": lambda v: [
            ("takeoff", {"altitude": v["alt"]}),
            ("move", {"dx": 0.0, "dy": 0.0, "dz": 0.0, "d_yaw": -v["angle"]}),
        ],
    },
    {
        "vars": [],
        "template": "처음 위치로 돌아간 후 착륙해",
        "build": lambda v: [
            ("goto_history", {"recall": "first"}),
            ("land", {}),
        ],
    },
    {
        "vars": ["dist", "dist2"],
        "template": "{dist}m 뒤로 물러난 다음 {dist2}m 위로 올라가",
        "build": lambda v: [
            ("move", {"dx": -v["dist"], "dy": 0.0, "dz": 0.0, "d_yaw": 0.0}),
            ("move", {"dx": 0.0, "dy": 0.0, "dz": v["dist2"], "d_yaw": 0.0}),
        ],
    },
    {
        "vars": ["angle", "dist"],
        "template": "왼쪽으로 {angle}도 돌고 {dist}미터 전진해",
        "build": lambda v: [
            ("move", {"dx": 0.0, "dy": 0.0, "dz": 0.0, "d_yaw": v["angle"]}),
            ("move", {"dx": v["dist"], "dy": 0.0, "dz": 0.0, "d_yaw": 0.0}),
        ],
    },
]


def generate_compound_dataset(num_samples: int) -> list:
    dataset = []
    for _ in range(num_samples):
        pattern = random.choice(COMPOUND_PATTERNS)
        values = {}
        for var in pattern["vars"]:
            if var.startswith("dist"):
                values[var] = round(random.uniform(*DIST_RANGE_M), 1)
            elif var.startswith("angle"):
                values[var] = int(round(random.uniform(*ANGLE_RANGE)))
            elif var.startswith("alt"):
                values[var] = round(random.uniform(*ALT_RANGE), 1)
        text = pattern["template"].format(**values)
        calls = pattern["build"](values)
        dataset.append(make_multi_call_sample(text, calls))
    return dataset


# ============================================================
# 스코프 밖 요청 (tool_call 없이 거절)
# ============================================================

OUT_OF_SCOPE_PHRASES = [
    "오늘 날씨 어때?",
    "노래 좀 틀어줘",
    "커피 한 잔 타줘",
    "지금 몇 시야?",
    "옆방 불 좀 꺼줘",
    "심심한데 농담 하나 해줘",
    "주식 시세 좀 알려줘",
    "라면 끓이는 법 알려줘",
]

REFUSAL_TEXT = (
    "죄송하지만 저는 드론 비행 제어(이륙, 착륙, 이동, 회전, 이전/최초 위치 복귀)만 "
    "수행할 수 있어요. 요청하신 내용은 처리할 수 없습니다."
)


def generate_negative_dataset(num_samples: int) -> list:
    dataset = []
    for _ in range(num_samples):
        phrase = random.choice(OUT_OF_SCOPE_PHRASES)
        dataset.append(make_refusal_sample(phrase, REFUSAL_TEXT))
    return dataset


# ============================================================
# 저장 & QA 통계
# ============================================================

def save_jsonl(dataset: list, output_file: str):
    output_path = Path(OUTPUT_PATH) / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"📄 저장 위치: {output_path.resolve()}")


def print_label_distribution(dataset: list, split_name: str):
    """샘플별 assistant 응답에서 어떤 tool이 몇 번 나왔는지 집계.
    복합 명령은 포함된 tool 각각을 세고, 거절 응답은 'no_tool_call'로 센다."""
    counter = Counter()
    for item in dataset:
        content = item["messages"][-1]["content"]
        if "<tool_call>" not in content:
            counter["no_tool_call"] += 1
            continue
        for line in content.split("<tool_call>")[1:]:
            call_json = line.split("</tool_call>")[0]
            try:
                name = json.loads(call_json)["name"]
                counter[name] += 1
            except (json.JSONDecodeError, KeyError):
                counter["parse_error"] += 1
    print(f"  [{split_name}] 라벨 분포: {dict(counter)}")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    # test는 자체 노이즈 템플릿을 이미 갖고 있으므로 0으로 둔다.
    # train/val은 같은 "종류"의 노이즈를 겪어보되, 구체적 오타/문장은 test와
    # 겹치지 않게 해서 "암기"가 아니라 "패턴 일반화"를 학습/평가하게 한다.
    NOISE_PROB = {
        "train": {"typo": 0.15, "spacing": 0.20},
        "val": {"typo": 0.15, "spacing": 0.20},
        "test": {"typo": 0.0, "spacing": 0.0},
    }

    splits = {
        "train": (NUM_TRAIN, TRAIN_PHRASES),
        "val": (NUM_VAL, VAL_PHRASES),
        "test": (NUM_TEST, TEST_PHRASES),
    }

    results = {}
    for split_name, (num_loops, phrases) in splits.items():
        noise = NOISE_PROB[split_name]
        dataset = generate_basic_dataset(num_loops, phrases, typo_prob=noise["typo"], spacing_prob=noise["spacing"])
        dataset += generate_compound_dataset(NUM_COMPOUND[split_name])
        dataset += generate_negative_dataset(NUM_NEGATIVE[split_name])
        random.shuffle(dataset)
        results[split_name] = dataset
        save_jsonl(dataset, f"{split_name}.jsonl")

    total = sum(len(d) for d in results.values())
    print(f"\n✅ Train: {len(results['train'])}개")
    print(f"🧪 Validation: {len(results['val'])}개")
    print(f"📝 Test: {len(results['test'])}개")
    print(f"📦 전체: {total}개\n")

    print("📊 클래스 분포 점검 (불균형 확인용):")
    for split_name, dataset in results.items():
        print_label_distribution(dataset, split_name)