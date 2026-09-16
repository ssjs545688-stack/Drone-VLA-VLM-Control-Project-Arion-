"""
schema.py

Qwen3 Tool Calling에서 쓰는 함수 스키마와 시스템 프롬프트만 모아둔 모듈.
ROS2/torch 등 무거운 의존성이 전혀 없어서, 파인튜닝 데이터셋 생성 스크립트나
검증 스크립트에서도 이 파일 하나만 import해서 재사용할 수 있다.
"""

import json
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
        "type": "function",
        "function": {
            "name": "move",
            "description": "드론의 현재 위치 및 기수 방향을 기준으로 상대 이동시킵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dx": {
                        "type": "number",
                        "description": "전진(+) / 후진(-) 상대 거리 (단위: 미터)",
                    },
                    "dy": {
                        "type": "number",
                        "description": "우측(+) / 좌측(-) 상대 거리 (단위: 미터)",
                    },
                    "dz": {
                        "type": "number",
                        "description": "상승(+) / 하강(-) 상대 고도 변화량 (단위: 미터)",
                    },
                    "d_yaw": {
                        "type": "number",
                        "description": "시계방향(+) / 반시계방향(-) 상대 회전 각도 (단위: 도)",
                    },
                },
                "required": ["dx", "dy", "dz", "d_yaw"],
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
            "description": "이전 비행 위치 기록(History)으로 드론을 복귀시킵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recall": {
                        "type": "string",
                        "enum": ["previous", "first"],
                        "description": (
                            "복귀 유형. previous는 직전 명령 시작 위치, "
                            "first는 최초 기록된 출발지/원점입니다."
                        ),
                    }
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

TOOL_NAMES = {
    "takeoff",
    "move",
    "land",
    "goto_history",
    "reverse_plan",
}

SCHEMA_TEXT = json.dumps(DEFINE_SCHEMA, ensure_ascii=False, indent=2)

SYSTEM_PROMPT = f"""
너는 PX4 드론의 자연어 명령을 ROS 2 Tool Call로 변환하는 명령 해석기다.

사용 가능한 Tool은 아래 5개뿐이다.
{SCHEMA_TEXT}

규칙:
1. 사용자의 자연어 명령을 위 Tool 중 하나 이상으로 변환한다.
2. 반드시 다음 형식으로 출력한다.
   <tool_call>{{"name":"툴이름","arguments":{{...}}}}</tool_call>
3. 여러 동작이 필요한 경우 <tool_call>을 여러 개 출력한다.
4. move의 dx, dy, dz, d_yaw는 반드시 모두 포함한다. 필요 없는 값은 0으로 한다.
5. takeoff의 인자는 반드시 altitude를 사용한다.
6. goto_history는 recall만 사용한다. recall은 previous 또는 first 중 하나다.
7. reverse_plan은 arguments={{}}로 호출한다.
8. 좌표 x/y/z를 직접 생성하지 않는다. 현재 위치/좌표 계산은 ROS 2 노드가 담당한다.
9. 설명문, Markdown, ```json 코드블록, 임의의 함수 이름을 출력하지 않는다.
10. 여러 Tool Call을 출력할 때는 순서를 사용자의 명령 순서와 일치시킨다.
11. 사용자의 명령이 애매하면 임의의 좌표를 추측하지 말고 가장 직접적인 Tool Call을 선택한다.
12. 내부 추론 과정(thinking)은 출력하지 않는다.
"""