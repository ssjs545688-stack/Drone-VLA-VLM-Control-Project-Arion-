#!/usr/bin/env python3
"""
Qwen3-1.7B 모델 평가
base / lora / merged / quantized 중 1~2개 선택
"""

import sys,os,json,re
sys.path.append(os.path.abspath("../ros2_ws/src/llm_drone_control"))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
from llm_drone_control.schema import DEFINE_SCHEMA,SYSTEM_PROMPT

TRAIN_DATASET_PATH="./dataset/train.jsonl"
VAL_DATASET_PATH="./dataset/val.jsonl"
TEST_DATASET_PATH="./dataset/test.jsonl"

MODEL_ID="../models/Qwen3-1.7B"
LORA_DIR="../models/finetuned_qwen3-1.7B_drone_lora"
MERGED_DIR="../models/qwen3-1.7B-drone-merged"
QUANTIZED_DIR="../models/qwen3-1.7B-drone-int4-awq"

RESULT_PATH="../models/evaluation_results.json"

MAX_NEW_TOKENS=256
DO_SAMPLE=False

# 1개 평가
# MODEL_A="quantized"; MODEL_B=None

# 2개 비교
# MODEL_A="merged"; MODEL_B="quantized"

MODEL_A="quantized"
MODEL_B=None

EVAL_SPLITS=["train","validation","test"]

MODEL_PATHS={
    "base":MODEL_ID,
    "lora":LORA_DIR,
    "merged":MERGED_DIR,
    "quantized":QUANTIZED_DIR
}

MODEL_NAMES={
    "base":"원본 모델",
    "lora":"원본 + LoRA Adapter",
    "merged":"원본 + LoRA Merged",
    "quantized":"INT4 AWQ 양자화 모델"
}

DATASETS={
    "train":TRAIN_DATASET_PATH,
    "validation":VAL_DATASET_PATH,
    "test":TEST_DATASET_PATH
}


def load_jsonl(path):
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_text(item,role):
    for m in item.get("messages",[]):
        if m.get("role")==role:
            return m.get("content","")
    return ""


def parse_tool_calls(text):
    matches=re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>",text,re.S)
    calls=[]
    for x in matches:
        try:
            obj=json.loads(x)
            if isinstance(obj,dict) and "name" in obj:
                calls.append(obj)
        except:
            pass
    return calls


def get_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(model_type):
    if model_type not in MODEL_PATHS:
        raise ValueError(f"잘못된 모델: {model_type}")

    print(f"\n[{model_type}] {MODEL_NAMES[model_type]}")
    print(f"경로: {MODEL_PATHS[model_type]}")

    if model_type=="quantized":
        from awq import AutoAWQForCausalLM

        tokenizer=AutoTokenizer.from_pretrained(
            QUANTIZED_DIR,
            trust_remote_code=True
        )

        model=AutoAWQForCausalLM.from_quantized(
            QUANTIZED_DIR,
            device_map="auto",
            trust_remote_code=True
        )
        return model,tokenizer

    tokenizer=AutoTokenizer.from_pretrained(
        MODEL_PATHS[model_type],
        trust_remote_code=True
    )

    if model_type=="lora":
        from peft import PeftModel

        base=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=get_dtype(),
            device_map="auto",
            trust_remote_code=True
        )

        model=PeftModel.from_pretrained(base,LORA_DIR)
        return model,tokenizer

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS[model_type],
        torch_dtype=get_dtype(),
        device_map="auto",
        trust_remote_code=True
    )

    return model,tokenizer


def build_prompt(tokenizer,text):
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":text}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tools=DEFINE_SCHEMA,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def generate(model,tokenizer,text):
    prompt=build_prompt(tokenizer,text)

    inputs=tokenizer(
        prompt,
        return_tensors="pt"
    )

    device=next(model.parameters()).device
    inputs={k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        output=model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated=output[0][inputs["input_ids"].shape[1]:]
    result=tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    return re.sub(
        r"<tool_call>\s*",
        "<tool_call>",
        result
    ).strip()


def norm(v):
    if isinstance(v,float) and v==0:
        return 0.0
    return v


def args_match(pred,gt):
    pa=pred.get("arguments",{})
    ga=gt.get("arguments",{})

    if set(pa)!=set(ga):
        return False

    for k in ga:
        if norm(pa[k])!=norm(ga[k]):
            return False

    return True


def calls_match(pred,gt):
    if len(pred)!=len(gt):
        return False

    return all(
        p.get("name")==g.get("name") and args_match(p,g)
        for p,g in zip(pred,gt)
    )


def calculate_metrics(results):
    total=len(results)
    tool_results=[r for r in results if r["expected_calls"]]
    negative_results=[r for r in results if not r["expected_calls"]]

    parse_ok=sum(
        bool(r["predicted_calls"])
        for r in tool_results
    )

    tool_name_ok=sum(
        bool(r["predicted_calls"]) and
        r["predicted_calls"][0].get("name")==r["expected_calls"][0].get("name")
        for r in tool_results
    )

    argument_ok=sum(
        bool(r["predicted_calls"]) and
        r["predicted_calls"][0].get("name")==r["expected_calls"][0].get("name") and
        args_match(r["predicted_calls"][0],r["expected_calls"][0])
        for r in tool_results
    )

    exact_ok=sum(
        calls_match(r["predicted_calls"],r["expected_calls"])
        for r in tool_results
    )

    refusal_ok=sum(
        not r["predicted_calls"]
        for r in negative_results
    )

    params={}
    param_keys=["dx","dy","dz","d_yaw","altitude"]

    for key in param_keys:
        expected=0
        correct=0

        for r in tool_results:
            for call in r["expected_calls"]:
                args=call.get("arguments",{})
                if key in args:
                    expected+=1
                    pred_calls=r["predicted_calls"]

                    if pred_calls:
                        pred_args=pred_calls[0].get("arguments",{})
                        if key in pred_args and norm(pred_args[key])==norm(args[key]):
                            correct+=1

        params[key]=100*correct/expected if expected else 0.0

    return {
        "total":total,
        "tool_call_samples":len(tool_results),
        "negative_samples":len(negative_results),
        "tool_call_parse_rate":100*parse_ok/len(tool_results) if tool_results else 0.0,
        "tool_name_accuracy":100*tool_name_ok/len(tool_results) if tool_results else 0.0,
        "argument_accuracy":100*argument_ok/len(tool_results) if tool_results else 0.0,
        "exact_match_accuracy":100*exact_ok/len(tool_results) if tool_results else 0.0,
        "tool_call_exact_match_accuracy":100*exact_ok/len(tool_results) if tool_results else 0.0,
        "refusal_accuracy":100*refusal_ok/len(negative_results) if negative_results else 0.0,
        "overall_exact_match_accuracy":100*(exact_ok+refusal_ok)/total if total else 0.0,
        "parameter_accuracy":params
    }


def evaluate(model,tokenizer,data,split,model_type):
    results=[]

    for i,item in enumerate(tqdm(
        data,
        desc=f"{model_type} / {split}"
    )):
        user=extract_text(item,"user")
        expected=parse_tool_calls(
            extract_text(item,"assistant")
        )

        try:
            output=generate(
                model,
                tokenizer,
                user
            )
            predicted=parse_tool_calls(output)
            error=None
        except Exception as e:
            output=""
            predicted=[]
            error=str(e)

        results.append({
            "index":i,
            "input":user,
            "expected_calls":expected,
            "predicted_calls":predicted,
            "output":output,
            "error":error
        })

    return results


def get_errors(results):
    return [
        r for r in results
        if not calls_match(
            r["predicted_calls"],
            r["expected_calls"]
        )
    ]


def print_failures(results,limit=10):
    errors=get_errors(results)

    if not errors:
        print("실패 없음")
        return

    print(f"\n실패 샘플 {len(errors)}개")

    for r in errors[:limit]:
        print("\n---")
        print("입력:",r["input"])
        print("정답:",r["expected_calls"])
        print("예측:",r["predicted_calls"])
        if r["error"]:
            print("ERROR:",r["error"])


def build_comparison(output):
    a=output["models"][MODEL_A]
    b=output["models"][MODEL_B]

    comparison={}

    for split in EVAL_SPLITS:
        ma=a["metrics"][split]
        mb=b["metrics"][split]

        comparison[split]={
            "model_a":MODEL_A,
            "model_b":MODEL_B,
            "overall_exact_match_diff":
                ma["overall_exact_match_accuracy"]-
                mb["overall_exact_match_accuracy"],
            "tool_call_exact_match_diff":
                ma["tool_call_exact_match_accuracy"]-
                mb["tool_call_exact_match_accuracy"],
            "argument_accuracy_diff":
                ma["argument_accuracy"]-
                mb["argument_accuracy"],
            "refusal_accuracy_diff":
                ma["refusal_accuracy"]-
                mb["refusal_accuracy"],
            "parameter_accuracy_diff":{
                k:
                ma["parameter_accuracy"][k]-
                mb["parameter_accuracy"][k]
                for k in ma["parameter_accuracy"]
            }
        }

    return comparison


def print_comparison(output):
    a=output["models"][MODEL_A]
    b=output["models"][MODEL_B]

    print("\n"+"="*70)
    print("모델 비교")
    print("="*70)

    print(f"A: {MODEL_NAMES[MODEL_A]}")
    print(f"B: {MODEL_NAMES[MODEL_B]}")

    for split in EVAL_SPLITS:
        ma=a["metrics"][split]
        mb=b["metrics"][split]

        print(f"\n[{split}]")
        print(
            f"Overall Exact : "
            f"{ma['overall_exact_match_accuracy']:.2f}% / "
            f"{mb['overall_exact_match_accuracy']:.2f}%"
        )
        print(
            f"Tool Exact    : "
            f"{ma['tool_call_exact_match_accuracy']:.2f}% / "
            f"{mb['tool_call_exact_match_accuracy']:.2f}%"
        )
        print(
            f"Argument      : "
            f"{ma['argument_accuracy']:.2f}% / "
            f"{mb['argument_accuracy']:.2f}%"
        )
        print(
            f"Refusal       : "
            f"{ma['refusal_accuracy']:.2f}% / "
            f"{mb['refusal_accuracy']:.2f}%"
        )

        print("Parameter:")
        for k in ma["parameter_accuracy"]:
            print(
                f"  {k:8s}: "
                f"{ma['parameter_accuracy'][k]:.2f}% / "
                f"{mb['parameter_accuracy'][k]:.2f}%"
            )


def main():
    valid={"base","lora","merged","quantized"}

    if MODEL_A not in valid:
        raise ValueError(
            "MODEL_A는 base/lora/merged/quantized 중 하나여야 합니다."
        )

    if MODEL_B is not None:
        if MODEL_B not in valid:
            raise ValueError(
                "MODEL_B는 base/lora/merged/quantized 또는 None이어야 합니다."
            )

        if MODEL_A==MODEL_B:
            raise ValueError(
                "MODEL_A와 MODEL_B는 서로 다른 모델이어야 합니다."
            )

    model_types=[MODEL_A]
    if MODEL_B:
        model_types.append(MODEL_B)

    output={
        "config":{
            "model_a":MODEL_A,
            "model_b":MODEL_B,
            "model_a_name":MODEL_NAMES[MODEL_A],
            "model_b_name":MODEL_NAMES[MODEL_B] if MODEL_B else None,
            "eval_splits":EVAL_SPLITS
        },
        "models":{}
    }

    for model_type in model_types:
        model,tokenizer=load_model(model_type)

        model_output={
            "name":MODEL_NAMES[model_type],
            "path":MODEL_PATHS[model_type],
            "metrics":{},
            "results":{}
        }

        for split in EVAL_SPLITS:
            print(f"\n{'='*70}")
            print(f"{MODEL_NAMES[model_type]} / {split}")
            print("="*70)

            data=load_jsonl(DATASETS[split])

            print(f"Dataset: {len(data)}")

            results=evaluate(
                model,
                tokenizer,
                data,
                split,
                model_type
            )

            metrics=calculate_metrics(results)

            model_output["metrics"][split]=metrics
            model_output["results"][split]=results

            print("\n결과:")
            for k,v in metrics.items():
                if k!="parameter_accuracy":
                    print(f"{k}: {v}")

            print("\nParameter:")
            for k,v in metrics["parameter_accuracy"].items():
                print(f"  {k}: {v:.2f}%")

            print_failures(results)

        output["models"][model_type]=model_output

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if MODEL_B:
        output["comparison"]=build_comparison(output)
        print_comparison(output)

    with open(RESULT_PATH,"w",encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\n결과 저장: {RESULT_PATH}")


if __name__=="__main__":
    main()