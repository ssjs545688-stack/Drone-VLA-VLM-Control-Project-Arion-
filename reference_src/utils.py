"""
utils.py

각도 정규화, 안전한 숫자 변환, Tool Call JSON 파싱/정규화용 순수 함수 모음.
어떤 외부 라이브러리(rclpy, torch 등)에도 의존하지 않아 단위 테스트가 쉽다.
"""

import math
from typing import Any, Dict, Optional


def clamp_angle(angle_rad: float) -> float:
    """각도를 [-pi, pi] 범위로 정규화한다."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def normalize_tool_call(raw_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """다양한 Tool Call JSON 형태를 내부 표준 {"name":..., "arguments":...} 로 통일한다."""
    import json

    if not isinstance(raw_call, dict):
        return None

    # Qwen / OpenAI 계열의 function wrapper 지원
    if isinstance(raw_call.get("function"), dict):
        fn = raw_call["function"]
        name = fn.get("name")
        arguments = fn.get("arguments", {})
    else:
        name = raw_call.get("name") or raw_call.get("action")
        arguments = raw_call.get("arguments", {})

    if not name or not isinstance(name, str):
        return None

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None

    if not isinstance(arguments, dict):
        arguments = {}

    return {"name": name.strip(), "arguments": arguments}


def extract_balanced_json(text: str) -> Optional[str]:
    """문자열 내부에서 첫 번째 완전한 JSON object를 찾는다."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None
