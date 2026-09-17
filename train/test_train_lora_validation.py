import os
import json
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# ============================================================
# 1. 기본 설정
# ============================================================

MODEL_ID = "../models/Qwen3-0.6B"
DATASET_PATH = "./train.jsonl"

OUTPUT_DIR = "./finetuned_qwen3_drone_lora"
SPLIT_DIR = "./dataset_split"

SEED = 42

# 데이터 분할 비율
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================
# 2. 재현성 설정
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 3. GPU / dtype 확인
# ============================================================

print("=" * 60)
print("환경 확인")
print("=" * 60)

print(f"PyTorch       : {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전     : {torch.version.cuda}")

    # BF16 지원 여부 확인
    bf16_supported = torch.cuda.is_bf16_supported()
else:
    bf16_supported = False

print(f"BF16 지원     : {bf16_supported}")
print("=" * 60)


# ============================================================
# 4. 데이터셋 로드
# ============================================================

print("\n[1/7] 데이터셋 로딩 중...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

print(f"전체 데이터 수: {len(dataset)}")


# ============================================================
# 5. Train / Validation / Test 분할
#
# 최종 구조:
# Train      80%
# Validation 10%
# Test       10%
#
# Test 데이터는 학습 과정에서 절대 사용하지 않음.
# ============================================================

print("\n[2/7] 데이터셋 분할 중...")

# 먼저 80% / 20%
split_1 = dataset.train_test_split(
    test_size=(VALID_RATIO + TEST_RATIO),
    seed=SEED,
)

train_dataset = split_1["train"]
temp_dataset = split_1["test"]

# 남은 20%를 다시 10% / 10%
split_2 = temp_dataset.train_test_split(
    test_size=0.5,
    seed=SEED,
)

valid_dataset = split_2["train"]
test_dataset = split_2["test"]

print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(valid_dataset)}")
print(f"Test       : {len(test_dataset)}")


# ============================================================
# 6. 분할 데이터 저장
#
# 나중에 test 데이터만 사용해서 별도의 최종 평가를
# 수행할 수 있도록 저장함.
# ============================================================

os.makedirs(SPLIT_DIR, exist_ok=True)

train_dataset.to_json(
    os.path.join(SPLIT_DIR, "train.jsonl"),
    force_ascii=False,
)

valid_dataset.to_json(
    os.path.join(SPLIT_DIR, "validation.jsonl"),
    force_ascii=False,
)

test_dataset.to_json(
    os.path.join(SPLIT_DIR, "test.jsonl"),
    force_ascii=False,
)

print(f"\n분할 데이터 저장 완료: {SPLIT_DIR}/")


# ============================================================
# 7. Tokenizer
# ============================================================

print("\n[3/7] Tokenizer 로딩 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer 로딩 완료")


# ============================================================
# 8. Model
# ============================================================

print("\n[4/7] 모델 로딩 중...")

# GPU가 있으면 BF16 -> FP16 순서로 사용
# CPU라면 float32 사용
if torch.cuda.is_available():
    if bf16_supported:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16
else:
    model_dtype = torch.float32

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
    device_map="auto",
    trust_remote_code=True,
)

# 학습 시 cache 비활성화
model.config.use_cache = False

print("모델 로딩 완료")


# ============================================================
# 9. LoRA 설정
# ============================================================

print("\n[5/7] LoRA 설정 중...")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("LoRA 설정 완료")


# ============================================================
# 10. 학습 설정
#
# 핵심 변경사항
#
# eval_strategy="epoch"
#   -> 매 epoch마다 Validation 수행
#
# save_strategy="epoch"
#   -> 매 epoch마다 checkpoint 저장
#
# load_best_model_at_end=True
#   -> Validation Loss가 가장 좋은 모델을 마지막에 복원
#
# metric_for_best_model="eval_loss"
#   -> 가장 낮은 Validation Loss를 기준으로 best model 선택
#
# EarlyStoppingCallback
#   -> Validation Loss가 일정 epoch 동안 개선되지 않으면
#      학습을 조기 종료
# ============================================================

print("\n[6/7] Trainer 설정 중...")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # -------------------------
    # 학습
    # -------------------------
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,

    # -------------------------
    # Validation
    # -------------------------
    eval_strategy="epoch",

    # -------------------------
    # Checkpoint
    # -------------------------
    save_strategy="epoch",
    save_total_limit=2,

    # Validation Loss 기준 best model
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # -------------------------
    # Logging
    # -------------------------
    logging_strategy="steps",
    logging_steps=10,

    # -------------------------
    # Precision
    # -------------------------
    fp16=torch.cuda.is_available() and not bf16_supported,
    bf16=torch.cuda.is_available() and bf16_supported,

    # -------------------------
    # Sequence
    # -------------------------
    max_length=512,
    packing=False,

    # dataset의 text 컬럼 사용
    dataset_text_field="text",

    # -------------------------
    # 기타
    # -------------------------
    report_to="none",
    seed=SEED,
)


# ============================================================
# 11. SFT Trainer
# ============================================================

trainer = SFTTrainer(
    model=model,

    train_dataset=train_dataset,
    eval_dataset=valid_dataset,

    processing_class=tokenizer,

    args=training_args,

    peft_config=lora_config,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.0,
        )
    ],
)


# ============================================================
# 12. 학습 시작
# ============================================================

print("\n" + "=" * 60)
print("학습 시작")
print("=" * 60)

print(f"Train samples      : {len(train_dataset)}")
print(f"Validation samples : {len(valid_dataset)}")
print(f"Test samples       : {len(test_dataset)}")
print("=" * 60)

train_result = trainer.train()


# ============================================================
# 13. Best Model Validation 결과 확인
# ============================================================

print("\n" + "=" * 60)
print("최종 Validation 평가")
print("=" * 60)

eval_result = trainer.evaluate()

for key, value in eval_result.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value}")


# ============================================================
# 14. 학습 결과 저장
# ============================================================

print("\n[7/7] 모델 저장 중...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Best model이 복원된 상태에서 저장
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 학습 결과 저장
metrics = {
    "train": train_result.metrics,
    "validation": eval_result,
    "dataset_size": {
        "total": len(dataset),
        "train": len(train_dataset),
        "validation": len(valid_dataset),
        "test": len(test_dataset),
    },
}

with open(
    os.path.join(OUTPUT_DIR, "training_metrics.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 15. 학습 로그 저장
#
# 나중에 train_loss / eval_loss 그래프를 그릴 때 사용 가능
# ============================================================

with open(
    os.path.join(OUTPUT_DIR, "log_history.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        trainer.state.log_history,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 16. 완료
# ============================================================

print("\n" + "=" * 60)
print("학습 완료!")
print("=" * 60)

print(f"LoRA 모델       : {OUTPUT_DIR}")
print(f"Train 데이터   : {SPLIT_DIR}/train.jsonl")
print(f"Validation 데이터: {SPLIT_DIR}/validation.jsonl")
print(f"Test 데이터     : {SPLIT_DIR}/test.jsonl")

print("\n중요:")
print("- Validation은 매 epoch마다 수행됨")
print("- Validation Loss가 가장 좋은 모델이 최종 모델로 복원됨")
print("- Validation 개선이 없으면 Early Stopping 발생")
print("- Test 데이터는 학습에 사용되지 않음")
print("- 최종 성능 평가는 dataset_split/test.jsonl로 별도 수행")
print("=" * 60)
