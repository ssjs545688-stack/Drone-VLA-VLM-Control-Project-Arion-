import os,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL="../models/Qwen3-1.7B"
LORA_MODEL="../models/finetuned_qwen3-1.7B_drone_lora"
OUTPUT_MODEL="../models/qwen3-1.7B-drone-merged"

print("="*60)
print("LoRA Merge")
print("="*60)

dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print("\n[1/3] Base Model 로딩 중...")
model=AutoModelForCausalLM.from_pretrained(BASE_MODEL,torch_dtype=dtype,device_map="auto")
tokenizer=AutoTokenizer.from_pretrained(LORA_MODEL)
print("[1/3] Base Model 로딩 완료")

print("\n[2/3] LoRA Adapter 로딩 및 Merge 중...")
model=PeftModel.from_pretrained(model,LORA_MODEL)
model=model.merge_and_unload()
print("[2/3] LoRA Merge 완료")

print("\n[3/3] 모델 저장 중...")
os.makedirs(OUTPUT_MODEL,exist_ok=True)
model.save_pretrained(OUTPUT_MODEL,safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_MODEL)

print("[3/3] 모델 저장 완료")
print("\n"+"="*60)
print("LoRA Merge 완료!")
print(f"Base Model  : {BASE_MODEL}")
print(f"LoRA Model  : {LORA_MODEL}")
print(f"Merged Model: {OUTPUT_MODEL}")
print("="*60)