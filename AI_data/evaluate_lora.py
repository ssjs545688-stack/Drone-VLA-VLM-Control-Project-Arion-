"""
학습 데이터/검증 데이터/테스트 데이터 모두 평가를 수행하는 코드
"""
import sys,os,json,re
sys.path.append(os.path.abspath("../ros2_ws/src/llm_drone_control"))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from llm_drone_control.schema import DEFINE_SCHEMA,SYSTEM_PROMPT

TRAIN_DATASET_PATH="./dataset/train.jsonl"
VAL_DATASET_PATH="./dataset/val.jsonl"
TEST_DATASET_PATH="./dataset/test.jsonl"

MODEL_ID="../models/Qwen3-1.7B"
LORA_DIR="../models/finetuned_qwen3-1.7B_drone_lora"
RESULT_PATH="../models/finetuned_qwen3-1.7B_drone_lora/evaluation_results.json"

MAX_NEW_TOKENS=256
DO_SAMPLE=False

# 평가할 데이터셋 설정 ["train","validation","test"]
EVAL_SPLITS=["train","validation","test"]

def load_jsonl(path):
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: data.append(json.loads(line))
            except json.JSONDecodeError as e: print(f"[경고] {n}번째 줄 JSON 파싱 실패: {e}")
    return data


def extract_text(item,role):
    return next((m.get("content","").strip() for m in item.get("messages",[]) if m.get("role")==role),"")


def parse_tool_calls(text):
    if not text:return []
    matches=re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>",text,re.DOTALL) or [text.strip()]
    calls=[]
    for x in matches:
        try: obj=json.loads(x.strip())
        except json.JSONDecodeError: continue
        if isinstance(obj,dict):
            args=obj.get("arguments",{})
            calls.append({"name":obj.get("name"),"arguments":args if isinstance(args,dict) else {}})
    return calls


def get_dtype():
    if not torch.cuda.is_available(): return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(lora=False):
    path=LORA_DIR if lora else MODEL_ID
    tokenizer=AutoTokenizer.from_pretrained(path,trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token

    base=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,torch_dtype=get_dtype(),device_map="auto",trust_remote_code=True
    )

    model=PeftModel.from_pretrained(base,LORA_DIR) if lora else base
    model.eval()
    return tokenizer,model


def build_prompt(tokenizer,text):
    return tokenizer.apply_chat_template(
        [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":text}],
        tools=DEFINE_SCHEMA,tokenize=False,add_generation_prompt=True,enable_thinking=False
    )


@torch.inference_mode()
def generate(model,tokenizer,text):
    inputs=tokenizer(build_prompt(tokenizer,text),return_tensors="pt")
    device=next(model.parameters()).device
    inputs={k:v.to(device) for k,v in inputs.items()}
    output=model.generate(
        **inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id
    )
    raw = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True
    ).strip()
    
    # <tool_call> 태그 내부의 불필요한 줄바꿈 제거
    cleaned = re.sub(r'<tool_call>\s*', '<tool_call>', raw)
    cleaned = re.sub(r'\s*</tool_call>', '</tool_call>', cleaned)
    return cleaned


def norm(v):
    if isinstance(v,float) and v.is_integer(): return int(v)
    return v.strip() if isinstance(v,str) else v


def args_match(a,b):
    return set(a)==set(b) and all(norm(a[k])==norm(b[k]) for k in a)


def calls_match(a,b):
    return len(a)==len(b) and all(
        x["name"]==y["name"] and args_match(x["arguments"],y["arguments"])
        for x,y in zip(a,b)
    )


def calculate_metrics(results):
    total=len(results)
    tools=[r for r in results if r["true_calls"]]
    refusals=[r for r in results if not r["true_calls"]]

    if not total:
        return {k:0 for k in [
            "total","tool_call_samples","negative_samples",
            "tool_call_parse_rate","tool_name_accuracy","argument_accuracy",
            "exact_match_accuracy","tool_call_exact_match_accuracy",
            "refusal_accuracy"
        ]}|{"parameter_accuracy":{}}

    parse=sum(bool(r["predicted_calls"]) for r in tools)/len(tools)
    name_ok=args_ok=exact_ok=0
    ptotal,pcorrect={},{}

    for r in tools:
        t,p=r["true_calls"],r["predicted_calls"]

        if len(t)==len(p):
            n_ok=a_ok=True
            for tc,pc in zip(t,p):
                if tc["name"]!=pc["name"]: n_ok=False
                if not args_match(tc["arguments"],pc["arguments"]): a_ok=False

                for k,v in tc["arguments"].items():
                    ptotal[k]=ptotal.get(k,0)+1
                    if k in pc["arguments"] and norm(v)==norm(pc["arguments"][k]):
                        pcorrect[k]=pcorrect.get(k,0)+1

            name_ok+=n_ok
            args_ok+=a_ok

        exact_ok+=calls_match(t,p)

    refusal=sum(not r["predicted_calls"] for r in refusals)/len(refusals) if refusals else 0
    overall=sum(calls_match(r["true_calls"],r["predicted_calls"]) for r in results)/total

    return {
        "total":total,
        "tool_call_samples":len(tools),
        "negative_samples":len(refusals),
        "tool_call_parse_rate":parse,
        "tool_name_accuracy":name_ok/len(tools) if tools else 0,
        "argument_accuracy":args_ok/len(tools) if tools else 0,
        "exact_match_accuracy":overall,
        "tool_call_exact_match_accuracy":exact_ok/len(tools) if tools else 0,
        "refusal_accuracy":refusal,
        "parameter_accuracy":{k:pcorrect.get(k,0)/v for k,v in ptotal.items()}
    }


def evaluate(model,tokenizer,data):
    results=[]
    for i,item in enumerate(tqdm(data)):
        user=extract_text(item,"user")
        expected=extract_text(item,"assistant")
        true=parse_tool_calls(expected)

        try:
            prediction=generate(model,tokenizer,user)
            predicted=parse_tool_calls(prediction)
        except Exception as e:
            prediction=f"[ERROR] {e}"
            predicted=[]

        results.append({
            "index":i,"input":user,"expected":expected,
            "prediction":prediction,"true_calls":true,"predicted_calls":predicted
        })

    return calculate_metrics(results),results


def get_errors(results):
    return [
        {k:r[k] for k in [
            "index","input","expected","prediction","true_calls","predicted_calls"
        ]}
        for r in results
        if not calls_match(r["true_calls"],r["predicted_calls"])
    ]


def print_failures(results,max_items=20):
    errors=[r for r in results if not calls_match(r["true_calls"],r["predicted_calls"])]
    print(f"\n오답 샘플 ({min(len(errors),max_items)}개):")
    for r in errors[:max_items]:
        print("\n"+"-"*70)
        print(f"[입력] {r['input']}")
        print(f"[정답] {r['expected']}")
        print(f"[예측] {r['prediction']}")

def print_comparison(output):
    metrics=[
        ("tool_call_parse_rate","Tool Call 파싱 성공률"),
        ("tool_name_accuracy","Tool Name 정확도"),
        ("argument_accuracy","Argument 정확도"),
        ("tool_call_exact_match_accuracy","Tool Call Exact Match"),
        ("refusal_accuracy","Refusal 정확도"),
        ("exact_match_accuracy","전체 Exact Match")
    ]

    for split in EVAL_SPLITS:
        base=output["base_model"][split]["metrics"]
        lora=output["lora_model"][split]["metrics"]

        print(f"\n{'='*70}")
        print(f"{split.upper()} - BASE → LORA 비교")
        print("="*70)

        for key,label in metrics:
            b=base[key]*100
            l=lora[key]*100
            print(f"{label:26s}: {b:6.2f}% → {l:6.2f}%  ({l-b:+.2f}%p)")

        print("\nParameter별 정확도")

        keys=set(base["parameter_accuracy"])|set(lora["parameter_accuracy"])

        for key in sorted(keys):
            b=base["parameter_accuracy"].get(key,0)*100
            l=lora["parameter_accuracy"].get(key,0)*100
            print(f"  {key:8s}: {b:6.2f}% → {l:6.2f}%  ({l-b:+.2f}%p)")

def build_comparison(output):
    metrics=[
        "tool_call_parse_rate",
        "tool_name_accuracy",
        "argument_accuracy",
        "tool_call_exact_match_accuracy",
        "refusal_accuracy",
        "exact_match_accuracy"
    ]

    comparison={}

    for split in EVAL_SPLITS:
        base=output["base_model"][split]["metrics"]
        lora=output["lora_model"][split]["metrics"]

        comparison[split]={}

        for key in metrics:
            b=base[key]
            l=lora[key]

            comparison[split][key]={
                "base":b,
                "lora":l,
                "improvement_pp":(l-b)*100
            }

        comparison[split]["parameter_accuracy"]={}

        keys=set(base["parameter_accuracy"])|set(lora["parameter_accuracy"])

        for key in sorted(keys):
            b=base["parameter_accuracy"].get(key,0)
            l=lora["parameter_accuracy"].get(key,0)

            comparison[split]["parameter_accuracy"][key]={
                "base":b,
                "lora":l,
                "improvement_pp":(l-b)*100
            }

    return comparison

def main():
    print("="*70+"\n        LoRA 드론 제어 모델 평가\n"+"="*70)

    all_datasets={
        "train":load_jsonl(TRAIN_DATASET_PATH),
        "validation":load_jsonl(VAL_DATASET_PATH),
        "test":load_jsonl(TEST_DATASET_PATH)
    }

    datasets={k:all_datasets[k] for k in EVAL_SPLITS}

    for name,data in datasets.items():
        print(f"{name.capitalize()} 데이터 : {len(data)}개")

    output={
        "config":{
            "model_id":MODEL_ID,
            "lora_dir":LORA_DIR,
            "train_dataset":TRAIN_DATASET_PATH,
            "validation_dataset":VAL_DATASET_PATH,
            "test_dataset":TEST_DATASET_PATH,
            "eval_splits":EVAL_SPLITS,
            "train_size":len(all_datasets["train"]),
            "validation_size":len(all_datasets["validation"]),
            "test_size":len(all_datasets["test"]),
            "max_new_tokens":MAX_NEW_TOKENS,
            "do_sample":DO_SAMPLE
        },
        "comparison":{}
    }

    for model_name,is_lora in [("base_model",False),("lora_model",True)]:
        print(f"\n{'='*70}\n{model_name.upper()} 평가\n{'='*70}")

        tokenizer,model=load_model(is_lora)
        output[model_name]={}

        for split,data in datasets.items():
            print(f"\n{'='*70}\n{split.upper()} 평가 시작\n{'='*70}")

            metrics,results=evaluate(model,tokenizer,data)
            print_failures(results)

            output[model_name][split]={
                "metrics":metrics,
                "errors":get_errors(results)
            }

        del model,tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output["comparison"]=build_comparison(output)
    print_comparison(output)

    with open(RESULT_PATH,"w",encoding="utf-8") as f:
        json.dump(output,f,ensure_ascii=False,indent=2)

    print(f"\n평가 결과 저장 완료: {RESULT_PATH}")


if __name__=="__main__":
    main()