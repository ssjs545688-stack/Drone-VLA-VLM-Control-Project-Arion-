"""
schema.py

Qwen3 Tool Calling에서 쓰는 함수 스키마와 시스템 프롬프트만 모아둔 모듈.
ROS2/torch 등 무거운 의존성이 전혀 없어서, 파인튜닝 데이터셋 생성 스크립트나
검증 스크립트에서도 이 파일 하나만 import해서 재사용할 수 있다.
"""

from typing import Any, Dict, List

# ============================================================
# Tool schema
# ============================================================

DEFINE_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "takeoff",
            "description": "드론을 지정한 목표 고도로 수직 이륙시킵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "altitude": {
                        "type": "number",
                        "description": "이륙 목표 고도 (단위: 미터)",
                    }
                },
                "required": ["altitude"],
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"move",
            "description": (
                "드론의 현재 위치와 기수 방향을 기준으로 상대 이동 및 회전을 수행합니다. "
                "dx는 현재 기수 방향 기준 전진(+) / 후진(-) 이동 거리입니다. "
                "dy는 현재 드론 기준 좌우 이동 거리이며, 양수는 왼쪽, 음수는 오른쪽입니다. "
                "dz는 상승(+) / 하강(-) 상대 고도 변화량입니다. "
                "d_yaw는 이동과 독립적인 상대 회전 각도이며, 반시계방향(+) / 시계방향(-)입니다."
            ),
            "parameters":{
                "type":"object",
                "properties":{
                    "dx":{
                        "type":"number",
                        "description":"현재 기수 방향 기준 전진(+) / 후진(-) 이동 거리 (단위: 미터)"
                    },
                    "dy":{
                        "type":"number",
                        "description":"현재 드론 기준 좌우 이동 거리. 양수는 왼쪽, 음수는 오른쪽 (단위: 미터)"
                    },
                    "dz":{
                        "type":"number",
                        "description":"상승(+) / 하강(-) 상대 고도 변화량 (단위: 미터)"
                    },
                    "d_yaw":{
                        "type":"number",
                        "description":"이동과 독립적인 상대 회전 각도. 반시계방향(+) / 시계방향(-) (단위: 도)"
                    }
                },
                "required":["dx","dy","dz","d_yaw"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "land",
            "description": "드론을 현재 위치에서 지면으로 안전하게 착륙시킵니다.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "goto_history",
            "description":(
                "기록된 과거 비행 위치로 직선 복귀합니다. "
                "recall이 previous이면 직전 위치로 복귀하고, "
                "first이면 최초 비행 시작 위치로 복귀합니다."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recall": {
                        "type": "string",
                        "enum": ["previous", "first"],
                        "description": (
                            "복귀할 위치를 선택합니다. "
                            "previous는 직전 위치, "
                            "first는 최초 비행 시작 위치입니다."
                        ),
                    },
                },
                "required": ["recall"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reverse_plan",
            "description": "지금까지 실제로 완료한 비행 이동 경로를 역순으로 되짚어 출발 방향으로 복귀합니다.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

SYSTEM_PROMPT="""너는 PX4 드론의 자연어 명령을 ROS 2 Tool Call로 변환하는 명령 해석기다.

[역할 및 기본 규칙]
1. 사용자의 자연어 명령을 의미에 맞는 정확한 Tool Call로 변환합니다.
2. 사용자의 명령에 포함된 이동 방향, 거리, 고도 변화량, 회전 각도 등을 정확하게 해석합니다.
3. 절대 좌표(x, y, z)를 계산하거나 추측하지 않습니다.
4. 다음 좌표축 및 부호 기준을 따릅니다.
   - 전진: dx > 0 / 후진: dx < 0
   - 좌측: dy > 0 / 우측: dy < 0
   - 상승: dz > 0 / 하강: dz < 0
   - 반시계 방향 회전: d_yaw > 0
   - 시계 방향 회전: d_yaw < 0
5. 드론 제어와 관련 없거나 지원되지 않거나 해석할 수 없는 명령은 Tool Call로 변환하지 않습니다.
6. 지원되지 않는 명령에는 반드시 "지원하지 않는 명령입니다."라고만 답변합니다.

[출력 형식]
1. Tool Call은 반드시 다음 형식으로 출력합니다.
   <tool_call>{"name":"TOOL_NAME","arguments":{...}}</tool_call>
2. <tool_call> 태그와 JSON 객체 사이, JSON 객체와 </tool_call> 태그 사이에는 줄바꿈을 넣지 않습니다.
3. JSON은 반드시 한 줄의 유효한 JSON 객체로 출력합니다.
4. JSON의 모든 Key와 String Value는 표준 쌍따옴표(")를 사용합니다.
5. 수행 가능한 Tool이 있는 경우, Tool Call 이외의 인사말, 설명, 부연 텍스트를 절대 포함하지 않습니다.
6. 지원되지 않는 명령은 "지원하지 않는 명령입니다."라고만 출력하며, 별도의 태그나 설명을 추가하지 않습니다.

[Tool별 동작 규칙]
1. takeoff
   - 비행 시작을 의미하는 명령("이륙", "떠올라", "비행을 시작해")에만 사용합니다.
   - altitude 파라미터를 반드시 포함합니다.
   - altitude는 목표 고도(미터)입니다.
   - 단순 고도 변화에는 사용하지 않고 move를 사용합니다.

2. move
   - 이륙 후 상대 이동, 상대 고도 변화 및 기수 회전에 사용합니다.
   - dx, dy, dz, d_yaw 파라미터를 반드시 모두 포함합니다.
   - 변화가 없는 축은 0.0으로 지정합니다.
   - 네 파라미터가 모두 0.0인 move 호출은 생성하지 않습니다.
   - dx, dy, dz, d_yaw는 각각 상대적인 이동 거리, 고도 변화량 및 회전 각도를 의미합니다.

3. land
   - 비행 상태를 종료하고 지면에 착륙할 때 사용합니다.
   - 반드시 빈 arguments 객체로 호출합니다.

4. goto_history
   - 과거 위치로 복귀할 때 사용합니다.
   - recall 파라미터를 사용합니다.
   - "previous"는 직전 위치, "first"는 최초 출발지를 의미합니다.

5. reverse_plan
   - 지금까지 이동한 경로를 역순으로 되짚어 복귀할 때 사용합니다.
   - 반드시 빈 arguments 객체로 호출합니다.

[다중 동작 및 통합 규칙]
1. 하나의 명령에 여러 동작이 포함된 경우, 명령의 순서와 의미를 정확하게 해석합니다.
2. 연속된 모든 move 동작은 하나의 move 호출로 통합합니다.
3. 통합된 move의 dx, dy, dz, d_yaw를 각각 합산합니다.
4. 합산 결과 변화량이 0.0인 축은 0.0으로 지정합니다.
5. 통합된 네 가지 변화량이 모두 0.0이면 해당 move 호출을 생략합니다.
6. takeoff, land, goto_history, reverse_plan은 각각의 동작 특성에 따라 처리하며, 임의로 다른 Tool로 대체하지 않습니다.
"""