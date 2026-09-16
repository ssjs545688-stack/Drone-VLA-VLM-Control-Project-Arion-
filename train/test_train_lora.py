# train_lora.py
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# 1. 경로 설정
MODEL_ID = "../models/Qwen3-0.6B"
DATASET_PATH = "./train.jsonl"
OUTPUT_DIR = "./finetuned_qwen3_drone_lora"

# 2. 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 3. LoRA 설정 (어댑터 파라미터만 학습하여 메모리 절약)
lora_config = LoraConfig(
    r=16,                         # LoRA Rank
    lora_alpha=32,                # LoRA Alpha Scaling
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 4. 데이터셋 로드
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 5. 학습 인자 (Training Arguments) 설정
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_length=512,
    dataset_text_field="text",
    packing=False
)

# 6. SFTTrainer를 통한 학습 진행
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args
)

print("🚀 파인튜닝 시작...")
trainer.train()

# 7. LoRA 가중치 저장
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ 학습 완료 및 어댑터 저장 완료: {OUTPUT_DIR}")