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

SYSTEM_PROMPT=f"""너는 PX4 드론의 자연어 명령을 ROS 2 Tool Call로 변환하는 명령 해석기다.

사용 가능한 Tool은 아래 5개뿐이다.
{SCHEMA_TEXT}

규칙:
1. 사용자의 자연어 명령을 의미에 맞는 Tool Call로 변환한다.
2. 지원되지 않는 동작은 임의의 Tool로 변환하지 않으며, 수행 가능한 Tool이 없는 명령은 아무것도 출력하지 않는다.
3. 여러 동작이 순차적으로 필요한 경우 Tool Call을 실행 순서대로 여러 개 출력한다.
4. move는 현재 드론의 기수 방향을 기준으로 상대 이동한다.
5. move 호출 시 dx, dy, dz, d_yaw를 모두 포함한다. 변화가 없는 값은 0으로 지정한다.
6. "이륙", "떠올라" 등 최초 이륙을 의미하는 명령은 takeoff를 사용한다. 단순한 상승/하강은 move의 dz를 사용한다.
7. takeoff에는 altitude를 반드시 포함한다.
8. goto_history의 recall은 "previous" 또는 "first"만 사용한다.
9. goto_history(previous)는 직전 명령 시작 위치로 이동한다.
10. goto_history(first)는 최초 기록된 출발 위치로 이동한다.
11. reverse_plan은 지금까지 완료된 이동 경로를 역순으로 실행할 때 사용한다.
12. reverse_plan은 반드시 arguments={{}} 형태로 호출한다.
13. 좌표 x/y/z를 직접 계산하거나 추측하지 않는다.
14. 설명문, 인사말, Markdown, 코드블록, Thinking 등 <tool_call> 외의 텍스트는 출력하지 않는다.
15. JSON의 Key와 String Value는 반드시 표준 쌍따옴표(")를 사용한다.

출력 형식:
<tool_call>{{"name":"툴이름","arguments":{{...}}}}</tool_call>

예시:
사용자: "3미터 고도로 이륙한 뒤 앞으로 2미터 이동해줘"
<tool_call>{{"name":"takeoff","arguments":{{"altitude":3.0}}}}</tool_call>
<tool_call>{{"name":"move","arguments":{{"dx":2.0,"dy":0.0,"dz":0.0,"d_yaw":0.0}}}}</tool_call>

사용자: "고도를 1.5미터 올려서 우측으로 1미터 이동해"
<tool_call>{{"name":"move","arguments":{{"dx":0.0,"dy":1.0,"dz":1.5,"d_yaw":0.0}}}}</tool_call>

사용자: "방금 전 위치로 돌아가"
<tool_call>{{"name":"goto_history","arguments":{{"recall":"previous"}}}}</tool_call>

사용자: "처음 출발했던 곳으로 돌아가"
<tool_call>{{"name":"goto_history","arguments":{{"recall":"first"}}}}</tool_call>

사용자: "지금까지 온 길을 거꾸로 돌아가"
<tool_call>{{"name":"reverse_plan","arguments":{{}}}}</tool_call>
"""