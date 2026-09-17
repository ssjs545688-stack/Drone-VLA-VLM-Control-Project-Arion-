# evaluate_lora.py
"""
LoRA 드론 제어 모델 평가 스크립트

구성
1. train(2).jsonl을 train/test로 자동 분할
2. Base Model과 Base + LoRA 모델을 동일한 테스트셋으로 평가
3. Tool name 정확도
4. 전체 tool call 정확도 (JSON arguments 포함)
5. Parameter별 정확도
6. 결과를 evaluation_results.json에 저장

실행 예:
    python evaluate_lora.py

필요 패키지:
    pip install torch transformers peft datasets tqdm
"""

import json
import re
from pathlib import Path
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================
# 1. 경로 / 설정
# ============================================================

MODEL_ID = "../models/Qwen3-0.6B"

# 학습에 사용한 JSONL
DATASET_PATH = "./train.jsonl"

# LoRA 학습 결과 폴더
LORA_DIR = "./finetuned_qwen3_drone_lora"

# 평가 결과
RESULT_PATH = "./evaluation_results.json"

# 테스트셋 비율
TEST_RATIO = 0.1

# 같은 결과를 다시 얻을 수 있도록 고정
SEED = 42

# 생성 설정
MAX_NEW_TOKENS = 256
DO_SAMPLE = False


# ============================================================
# 2. 데이터 로드
# ============================================================

def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[경고] {line_no}번째 줄 JSON 파싱 실패: {e}")

    return data


def split_dataset(data, test_ratio=0.1, seed=42):
    """
    원본 학습 데이터에서 테스트셋을 분리한다.

    주의:
    가장 좋은 평가 방법은 학습 전에 train/validation/test를
    분리해 두는 것이다.
    이미 학습을 끝낸 상태라면 여기서는 임시 평가용으로 사용한다.
    """
    import random

    data = data.copy()

    random.Random(seed).shuffle(data)

    test_size = max(1, int(len(data) * test_ratio))

    test_data = data[:test_size]
    train_data = data[test_size:]

    return train_data, test_data


# ============================================================
# 3. 데이터에서 user / 정답 tool call 추출
# ============================================================

def extract_user_text(item):
    messages = item.get("messages", [])

    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "").strip()

    return ""


def extract_assistant_text(item):
    messages = item.get("messages", [])

    for message in messages:
        if message.get("role") == "assistant":
            return message.get("content", "").strip()

    return ""


def parse_tool_call(text):
    """
    다음 형식을 파싱한다.

    <tool_call>
    {"name": "takeoff", "arguments": {"altitude": 2.1}}
    </tool_call>
    """

    if not text:
        return None

    # <tool_call> 안쪽 우선 추출
    match = re.search(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        text,
        flags=re.DOTALL,
    )

    if match:
        candidate = match.group(1).strip()
    else:
        # 태그가 없더라도 JSON이 있으면 시도
        candidate = text.strip()

    # JSON object 추출
    json_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)

    if not json_match:
        return None

    json_text = json_match.group(0)

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    name = obj.get("name")
    arguments = obj.get("arguments", {})

    if not isinstance(arguments, dict):
        arguments = {}

    return {
        "name": name,
        "arguments": arguments,
    }


# ============================================================
# 4. 모델 로드
# ============================================================

def get_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    # CPU에서는 float32가 안전하다.
    return torch.float32


def load_base_model():
    print("\n[1/4] Base Model 로딩 중...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=get_dtype(),
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    print("Base Model 로딩 완료!")

    return tokenizer, model


def load_lora_model():
    print("\n[2/4] LoRA Model 로딩 중...")

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_DIR,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=get_dtype(),
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_DIR,
    )

    model.eval()

    print("LoRA Model 로딩 완료!")

    return tokenizer, model


# ============================================================
# 5. Prompt 생성
# ============================================================

SYSTEM_PROMPT = "너는 드론 제어 AI야. 사용자의 자연어 명령을 분석하여 적절한 함수(Tool Call)를 호출해."


def build_prompt(tokenizer, user_text):
    """
    학습 데이터의 messages 구조와 동일한 형태로 prompt를 만든다.

    Qwen 계열 tokenizer가 chat template을 제공하면 사용한다.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # chat template을 지원하지 않는 경우 fallback
        prompt = (
            f"<|im_start|>system\n"
            f"{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    return prompt


# ============================================================
# 6. 모델 추론
# ============================================================

@torch.inference_mode()
def generate(model, tokenizer, user_text):
    prompt = build_prompt(tokenizer, user_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # device_map="auto"인 경우 모델의 첫 device로 이동
    if hasattr(model, "device"):
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }
    else:
        inputs = {
            key: value.to(next(model.parameters()).device)
            for key, value in inputs.items()
        }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 입력 prompt 부분 제거
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    output_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return output_text


# ============================================================
# 7. 정확도 계산
# ============================================================

def normalize_value(value):
    """
    숫자는 2와 2.0을 같은 값으로 취급한다.
    문자열은 공백을 제거한다.
    """
    if isinstance(value, float):
        if value.is_integer():
            return int(value)

    if isinstance(value, str):
        return value.strip()

    return value


def values_equal(a, b):
    return normalize_value(a) == normalize_value(b)


def arguments_exact_match(true_args, pred_args):
    if set(true_args.keys()) != set(pred_args.keys()):
        return False

    for key in true_args:
        if not values_equal(true_args[key], pred_args[key]):
            return False

    return True


def tool_call_exact_match(true_call, pred_call):
    if true_call is None or pred_call is None:
        return False

    if true_call["name"] != pred_call["name"]:
        return False

    return arguments_exact_match(
        true_call["arguments"],
        pred_call["arguments"],
    )


def calculate_metrics(results):
    total = len(results)

    if total == 0:
        return {
            "total": 0,
            "parse_rate": 0,
            "tool_name_accuracy": 0,
            "argument_accuracy": 0,
            "exact_match_accuracy": 0,
        }

    parsed = [
        r for r in results
        if r["predicted_call"] is not None
    ]

    tool_correct = 0
    argument_correct = 0
    exact_correct = 0

    for r in results:
        true_call = r["true_call"]
        pred_call = r["predicted_call"]

        if pred_call is None or true_call is None:
            continue

        if true_call["name"] == pred_call["name"]:
            tool_correct += 1

        if arguments_exact_match(
            true_call["arguments"],
            pred_call["arguments"],
        ):
            argument_correct += 1

        if tool_call_exact_match(true_call, pred_call):
            exact_correct += 1

    return {
        "total": total,
        "parse_rate": len(parsed) / total,
        "tool_name_accuracy": tool_correct / total,
        "argument_accuracy": argument_correct / total,
        "exact_match_accuracy": exact_correct / total,
    }


# ============================================================
# 8. 모델 평가
# ============================================================

def evaluate_model(model_name, model, tokenizer, test_data):
    print(f"\n{'=' * 70}")
    print(f"{model_name} 평가 시작")
    print(f"{'=' * 70}")

    results = []

    for idx, item in enumerate(tqdm(test_data)):
        user_text = extract_user_text(item)
        assistant_text = extract_assistant_text(item)

        true_call = parse_tool_call(assistant_text)

        try:
            predicted_text = generate(
                model,
                tokenizer,
                user_text,
            )

            predicted_call = parse_tool_call(predicted_text)

        except Exception as e:
            predicted_text = f"[ERROR] {e}"
            predicted_call = None

        results.append(
            {
                "index": idx,
                "input": user_text,
                "expected": assistant_text,
                "prediction": predicted_text,
                "true_call": true_call,
                "predicted_call": predicted_call,
            }
        )

    metrics = calculate_metrics(results)

    return metrics, results


# ============================================================
# 9. 잘못된 결과 출력
# ============================================================

def print_failures(results, max_items=20):
    failures = [
        r for r in results
        if not tool_call_exact_match(
            r["true_call"],
            r["predicted_call"],
        )
    ]

    print(f"\n오답 샘플 ({min(len(failures), max_items)}개):")

    for r in failures[:max_items]:
        print("\n" + "-" * 70)
        print(f"[입력] {r['input']}")
        print(f"[정답] {r['expected']}")
        print(f"[예측] {r['prediction']}")


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * 40)
    print(f"총 테스트 수       : {metrics['total']}")
    print(f"JSON 파싱 성공률   : {metrics['parse_rate'] * 100:.2f}%")
    print(f"Tool Name 정확도   : {metrics['tool_name_accuracy'] * 100:.2f}%")
    print(f"Argument 정확도    : {metrics['argument_accuracy'] * 100:.2f}%")
    print(f"전체 Exact Match   : {metrics['exact_match_accuracy'] * 100:.2f}%")


# ============================================================
# 10. Main
# ============================================================

def main():
    print("=" * 70)
    print("        LoRA 드론 제어 모델 평가")
    print("=" * 70)

    # 데이터
    print("\n[3/4] 데이터셋 로딩 중...")

    data = load_jsonl(DATASET_PATH)

    print(f"전체 데이터 : {len(data)}개")

    _, test_data = split_dataset(
        data,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    print(f"테스트 데이터 : {len(test_data)}개")

    # Base
    base_tokenizer, base_model = load_base_model()

    base_metrics, base_results = evaluate_model(
        "BASE MODEL",
        base_model,
        base_tokenizer,
        test_data,
    )

    print_metrics("BASE MODEL 결과", base_metrics)

    print_failures(base_results)

    # 메모리 정리
    del base_model
    del base_tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # LoRA
    lora_tokenizer, lora_model = load_lora_model()

    lora_metrics, lora_results = evaluate_model(
        "BASE + LORA MODEL",
        lora_model,
        lora_tokenizer,
        test_data,
    )

    print_metrics("BASE + LORA 결과", lora_metrics)

    print_failures(lora_results)

    # 비교
    print("\n" + "=" * 70)
    print("최종 비교")
    print("=" * 70)

    print(
        f"Tool Name Accuracy : "
        f"{base_metrics['tool_name_accuracy'] * 100:.2f}%"
        f" -> "
        f"{lora_metrics['tool_name_accuracy'] * 100:.2f}%"
    )

    print(
        f"Argument Accuracy  : "
        f"{base_metrics['argument_accuracy'] * 100:.2f}%"
        f" -> "
        f"{lora_metrics['argument_accuracy'] * 100:.2f}%"
    )

    print(
        f"Exact Match        : "
        f"{base_metrics['exact_match_accuracy'] * 100:.2f}%"
        f" -> "
        f"{lora_metrics['exact_match_accuracy'] * 100:.2f}%"
    )

    # 결과 저장
    output = {
        "config": {
            "model_id": MODEL_ID,
            "lora_dir": LORA_DIR,
            "dataset": DATASET_PATH,
            "test_ratio": TEST_RATIO,
            "test_size": len(test_data),
            "seed": SEED,
        },
        "base_model": {
            "metrics": base_metrics,
            "results": base_results,
        },
        "lora_model": {
            "metrics": lora_metrics,
            "results": lora_results,
        },
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n평가 결과 저장 완료: {RESULT_PATH}")


if __name__ == "__main__":
    main()
