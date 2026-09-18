import os,json,random
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,EarlyStoppingCallback
from peft import LoraConfig
from trl import SFTConfig,SFTTrainer

# 경로
MODEL_ID="../models/Qwen3-1.7B"
TRAIN_DATASET_PATH="./dataset/train.jsonl"
VAL_DATASET_PATH="./dataset/val.jsonl"
OUTPUT_DIR="../models/finetuned_qwen3-1.7B_drone_lora"

SEED=42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

print("="*60)
print("환경 확인")
print("="*60)
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA 사용 가능  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"CUDA 버전      : {torch.version.cuda}")
    bf16_supported=torch.cuda.is_bf16_supported()
else:
    bf16_supported=False

print(f"BF16 지원     : {bf16_supported}")
print("="*60)

# 데이터셋
print("\n[1/5] 데이터셋 로딩 중...")

dataset=load_dataset("json",data_files={
    "train":TRAIN_DATASET_PATH,
    "validation":VAL_DATASET_PATH
})

train_dataset=dataset["train"]
valid_dataset=dataset["validation"]

print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(valid_dataset)}")

# Tokenizer
print("\n[2/5] Tokenizer 로딩 중...")

tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token

print("[2/5] Tokenizer 로딩 완료")

# 모델
print("\n[3/5] 모델 로딩 중...")

if torch.cuda.is_available():
    model_dtype=torch.bfloat16 if bf16_supported else torch.float16
else:
    model_dtype=torch.float32

model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=model_dtype,
    device_map="auto"
)

model.config.use_cache=False

print("[3/5] 모델 로딩 완료")

# LoRA
print("\n[4/5] LoRA 설정 중...")

lora_config=LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("[4/5] LoRA 설정 완료")

# Trainer
print("\n[5/5] Trainer 설정 중...")

training_args=SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_strategy="steps",
    logging_steps=10,
    fp16=torch.cuda.is_available() and not bf16_supported,
    bf16=torch.cuda.is_available() and bf16_supported,
    max_length=512,
    packing=False,
    report_to="none",
    seed=SEED
)

trainer=SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    args=training_args,
    peft_config=lora_config,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2,
            early_stopping_threshold=0.0
        )
    ]
)

print("\n"+"="*60)
print("학습 시작")
print(f"Train samples      : {len(train_dataset)}")
print(f"Validation samples : {len(valid_dataset)}")
print("="*60)

train_result=trainer.train()

# Validation
print("\n"+"="*60)
print("최종 Validation 평가")
print("="*60)

eval_result=trainer.evaluate()

for key,value in eval_result.items():
    if isinstance(value,(int,float)):
        print(f"{key}: {value}")

# 모델 저장
print("\n모델 저장 중...")

os.makedirs(OUTPUT_DIR,exist_ok=True)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics={
    "train":train_result.metrics,
    "validation":eval_result,
    "dataset_size":{
        "train":len(train_dataset),
        "validation":len(valid_dataset)
    }
}

with open(os.path.join(OUTPUT_DIR,"training_metrics.json"),"w",encoding="utf-8") as f:
    json.dump(metrics,f,ensure_ascii=False,indent=2)

with open(os.path.join(OUTPUT_DIR,"log_history.json"),"w",encoding="utf-8") as f:
    json.dump(trainer.state.log_history,f,ensure_ascii=False,indent=2)

print("\n"+"="*60)
print("학습 완료!")
print(f"LoRA 모델        : {OUTPUT_DIR}")
print(f"Train 데이터     : {TRAIN_DATASET_PATH}")
print(f"Validation 데이터: {VAL_DATASET_PATH}")
print("- Validation은 매 epoch마다 수행됨")
print("- Validation Loss가 가장 좋은 모델이 최종 모델로 복원됨")
print("- Validation 개선이 없으면 Early Stopping 발생")
print("="*60)