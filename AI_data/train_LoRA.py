# train_lora.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,TaskType
from trl import SFTTrainer,SFTConfig

# 경로
MODEL_ID="../models/Qwen3-0.6B"
TRAIN_DATASET_PATH="./dataset/train.jsonl"
VAL_DATASET_PATH="./dataset/val.jsonl"
TEST_DATASET_PATH="./dataset/test.jsonl"
OUTPUT_DIR="../models/finetuned_qwen3_drone_lora"

# 토크나이저 & 모델 로드
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

# LoRA 설정
lora_config=LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 데이터셋 로드
dataset=load_dataset("json",data_files={
    "train":TRAIN_DATASET_PATH,
    "validation":VAL_DATASET_PATH,
    "test":TEST_DATASET_PATH
})

print(f"📚 Train: {len(dataset['train'])}개")
print(f"🧪 Validation: {len(dataset['validation'])}개")
print(f"📝 Test: {len(dataset['test'])}개")

# 학습 설정
training_args=SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    max_length=512,
    dataset_text_field="text",
    packing=False
)

# SFT 학습
trainer=SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args
)

print("🚀 파인튜닝 시작...")
trainer.train()

# Test 데이터로 최종 평가
print("\n📝 Test 데이터 최종 평가...")
test_result=trainer.evaluate(
    eval_dataset=dataset["test"],
    metric_key_prefix="test"
)

print(f"📊 Test Loss: {test_result['test_loss']:.4f}")

# 모델 저장
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ 학습 완료 및 어댑터 저장 완료: {OUTPUT_DIR}")