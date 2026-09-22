from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer
import json

model_id="../models/qwen3-1.7B-drone-merged"
quant_path="../models/Qwen3-1.7B-AWQ-INT4"
VAL_DATASET_PATH="./dataset/val.jsonl"

quant_config=QuantizeConfig(bits=4,group_size=128,quant_method="awq")

with open(VAL_DATASET_PATH,"r",encoding="utf-8") as f:
    calibration_data=[
        next(m["content"] for m in json.loads(line)["messages"] if m["role"]=="user")
        for line in f if line.strip()
    ]

print(f"Calibration data: {len(calibration_data)}개")

model=GPTQModel.load(model_id,quant_config=quant_config)
tokenizer=AutoTokenizer.from_pretrained(model_id)

model.quantize(calibration_data)

model.save(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"양자화 완료: {quant_path}")