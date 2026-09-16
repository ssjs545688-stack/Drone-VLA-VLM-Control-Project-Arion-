#!/usr/bin/env python3
"""
Qwen3 Tool Calling 파인튜닝 데이터셋(train.jsonl) 품질 및 규격 검증 스크립트
"""

import json
import math
import re
import argparse
from pathlib import Path
from collections import Counter


TOOL_CALL_PATTERN = re.compile(r"^<tool_call>\n(.*?)\n</tool_call>$", re.DOTALL)
VALID_FUNCTIONS = {"takeoff", "land", "move", "return_along_path"}
VALID_MOVE_KEYS = {"forward", "right", "up"}


def is_number(value):
    """bool는 정수의 하위 타입이므로 제외하고 유한한 숫자만 허용한다."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_jsonl(file_path: str = "train.jsonl"):
    """JSONL의 각 줄을 검증하고 정상 샘플 수를 반환한다."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ 오류: '{file_path}' 파일을 찾을 수 없습니다.")
        return

    print(f"🔍 '{file_path}' 데이터셋 검증을 시작합니다...\n")

    stats = Counter()
    errors = []

    # 전체 파일을 메모리에 올리지 않고 한 줄씩 처리해 큰 데이터셋에도 대응한다.
    total_count = 0
    with open(path, "r", encoding="utf-8") as f:
        lines = enumerate(f, 1)
        for idx, line in lines:
            total_count += 1
            line_str = line.strip()
            if not line_str:
                errors.append((idx, "빈 줄(Empty Line) 포함"))
                continue

            # 1. 각 줄이 독립적인 JSON 객체인지 확인한다.
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                errors.append((idx, f"JSON 파싱 오류: {e}"))
                continue

            if not isinstance(data, dict):
                errors.append((idx, "각 줄의 최상위 값은 객체(dict)여야 합니다."))
                continue

            # 2. Qwen ChatML에 필요한 system/user/assistant 3개 메시지를 확인한다.
            messages = data.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                errors.append((idx, "'messages' 배열은 정확히 3개 항목이어야 합니다."))
                continue
            if not all(isinstance(message, dict) for message in messages):
                errors.append((idx, "'messages'의 각 항목은 객체(dict)여야 합니다."))
                continue

            roles = [message.get("role") for message in messages]
            if roles != ["system", "user", "assistant"]:
                errors.append((idx, f"잘못된 역할(role) 순서: {roles}"))
                continue
            if any(not isinstance(message.get("content"), str) for message in messages):
                errors.append((idx, "각 메시지의 'content'는 문자열이어야 합니다."))
                continue

            # 3. assistant가 Tool Call 하나만 정확한 태그 형식으로 반환하는지 확인한다.
            match = TOOL_CALL_PATTERN.fullmatch(messages[2]["content"])
            if not match:
                errors.append((idx, "<tool_call> ... </tool_call> 태그 형식이 올바르지 않습니다."))
                continue

            try:
                tool_data = json.loads(match.group(1))
            except json.JSONDecodeError:
                errors.append((idx, "<tool_call> 내부 JSON 문법 오류"))
                continue
            if not isinstance(tool_data, dict):
                errors.append((idx, "<tool_call> 내부 값은 객체(dict)여야 합니다."))
                continue

            # 4. 등록된 함수와 함수별 인자 스키마를 검증한다.
            func_name = tool_data.get("name")
            args = tool_data.get("arguments")
            if func_name not in VALID_FUNCTIONS:
                errors.append((idx, f"정의되지 않은 함수 이름: '{func_name}'"))
                continue
            if not isinstance(args, dict):
                errors.append((idx, "'arguments'는 객체(dict) 형태여야 합니다."))
                continue

            if func_name == "takeoff":
                if set(args) != {"altitude"} or not is_number(args["altitude"]) or args["altitude"] <= 0:
                    errors.append((idx, "takeoff은 양의 유한 숫자 'altitude'만 가져야 합니다."))
                    continue
            elif func_name in {"land", "return_along_path"}:
                if args:
                    errors.append((idx, f"{func_name} 함수는 인자를 받지 않아야 합니다."))
                    continue
            else:
                if not args or not set(args).issubset(VALID_MOVE_KEYS) or not all(is_number(value) for value in args.values()):
                    errors.append((idx, "move는 forward/right/up 중 하나 이상의 유한 숫자 인자가 필요합니다."))
                    continue

            stats[func_name] += 1

    if total_count == 0:
        print("❌ 오류: 파일이 비어 있습니다.")
        return 0

    # 📊 검증 결과 출력
    print("=" * 60)
    print("📊 [데이터셋 검증 요약 보고서]")
    print(f"- 전체 데이터 줄 수 : {total_count}개")
    print(f"- 정상 데이터 수     : {total_count - len(errors)}개")
    print(f"- 오류 데이터 수     : {len(errors)}개")
    print("=" * 60)

    print("\n📈 [함수별 호출 분포 statistics]")
    valid_count = total_count - len(errors)
    for func in sorted(VALID_FUNCTIONS):
        count = stats[func]
        print(f"  • {func:<20}: {count}개 ({count/total_count*100:.1f}%)")

    if errors:
        print("\n❌ [발견된 오류 상세 (최대 10개)]")
        for line_num, err_msg in errors[:10]:
            print(f"  • Line {line_num}: {err_msg}")
        print("\n⚠️ 오류가 발생한 데이터를 수정한 후 학습에 사용하세요.")
    else:
        print("\n🎉 모든 데이터가 표준 규격에 완벽히 부합합니다! 파인튜닝을 진행하셔도 좋습니다.")

    return valid_count


if __name__ == "__main__":
    # 명령행 인자로 다른 JSONL 파일도 검증할 수 있다.
    parser = argparse.ArgumentParser(description="Qwen3 Tool Calling JSONL 데이터셋 검증")
    parser.add_argument("file", nargs="?", default="train.jsonl", help="검증할 JSONL 파일 경로")
    args = parser.parse_args()
    validate_jsonl(args.file)