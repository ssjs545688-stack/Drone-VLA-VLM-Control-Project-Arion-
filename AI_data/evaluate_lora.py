import json,re
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
from llm_drone_control.schema import SYSTEM_PROMPT


# 경로
MODEL_ID="../models/Qwen3-0.6B"
TEST_DATASET_PATH="./dataset/test.jsonl"
LORA_DIR="../models/finetuned_qwen3_drone_lora"
RESULT_PATH="./evaluation_results.json"

MAX_NEW_TOKENS=256
DO_SAMPLE=False

def load_jsonl(path):
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for line_no,line in enumerate(f,1):
            line=line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[경고] {line_no}번째 줄 JSON 파싱 실패: {e}")
    return data


def extract_user_text(item):
    for message in item.get("messages",[]):
        if message.get("role")=="user":
            return message.get("content","").strip()
    return ""


def extract_assistant_text(item):
    for message in item.get("messages",[]):
        if message.get("role")=="assistant":
            return message.get("content","").strip()
    return ""


def parse_tool_call(text):
    if not text:
        return None

    match=re.search(r"<tool_call>\s*(.*?)\s*</tool_call>",text,re.DOTALL)
    candidate=match.group(1).strip() if match else text.strip()

    json_match=re.search(r"\{.*\}",candidate,re.DOTALL)
    if not json_match:
        return None

    try:
        obj=json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(obj,dict):
        return None

    arguments=obj.get("arguments",{})
    if not isinstance(arguments,dict):
        arguments={}

    return {
        "name":obj.get("name"),
        "arguments":arguments
    }


def get_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_base_model():
    print("\n[1/2] Base Model 로딩 중...")

    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=get_dtype(),
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()
    print("Base Model 로딩 완료!")
    return tokenizer,model


def load_lora_model():
    print("\n[2/2] Base + LoRA Model 로딩 중...")

    tokenizer=AutoTokenizer.from_pretrained(LORA_DIR,trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    base_model=AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=get_dtype(),
        device_map="auto",
        trust_remote_code=True
    )

    model=PeftModel.from_pretrained(base_model,LORA_DIR)
    model.eval()

    print("Base + LoRA Model 로딩 완료!")
    return tokenizer,model


def build_prompt(tokenizer,user_text):
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":user_text}
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


@torch.inference_mode()
def generate(model,tokenizer,user_text):
    prompt=build_prompt(tokenizer,user_text)

    inputs=tokenizer(
        prompt,
        return_tensors="pt"
    )

    device=next(model.parameters()).device
    inputs={k:v.to(device) for k,v in inputs.items()}

    output_ids=model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_ids=output_ids[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()


def normalize_value(value):
    if isinstance(value,float) and value.is_integer():
        return int(value)

    if isinstance(value,str):
        return value.strip()

    return value


def values_equal(a,b):
    return normalize_value(a)==normalize_value(b)


def arguments_exact_match(true_args,pred_args):
    if set(true_args.keys())!=set(pred_args.keys()):
        return False

    return all(
        values_equal(true_args[key],pred_args[key])
        for key in true_args
    )


def tool_call_exact_match(true_call,pred_call):
    if true_call is None or pred_call is None:
        return False

    return (
        true_call["name"]==pred_call["name"]
        and arguments_exact_match(
            true_call["arguments"],
            pred_call["arguments"]
        )
    )


def calculate_metrics(results):
    total=len(results)

    if total==0:
        return {
            "total":0,
            "parse_rate":0,
            "tool_name_accuracy":0,
            "argument_accuracy":0,
            "exact_match_accuracy":0,
            "parameter_accuracy":{}
        }

    parse_count=0
    tool_correct=0
    argument_correct=0
    exact_correct=0

    parameter_total={}
    parameter_correct={}

    for r in results:
        true_call=r["true_call"]
        pred_call=r["predicted_call"]

        if pred_call is not None:
            parse_count+=1

        if true_call is None or pred_call is None:
            continue

        if true_call["name"]==pred_call["name"]:
            tool_correct+=1

        if arguments_exact_match(
            true_call["arguments"],
            pred_call["arguments"]
        ):
            argument_correct+=1

        if tool_call_exact_match(true_call,pred_call):
            exact_correct+=1

        true_args=true_call["arguments"]
        pred_args=pred_call["arguments"]

        for key,true_value in true_args.items():
            parameter_total[key]=parameter_total.get(key,0)+1

            if key in pred_args and values_equal(
                true_value,pred_args[key]
            ):
                parameter_correct[key]=parameter_correct.get(key,0)+1

    parameter_accuracy={
        key:parameter_correct.get(key,0)/count
        for key,count in parameter_total.items()
    }

    return {
        "total":total,
        "parse_rate":parse_count/total,
        "tool_name_accuracy":tool_correct/total,
        "argument_accuracy":argument_correct/total,
        "exact_match_accuracy":exact_correct/total,
        "parameter_accuracy":parameter_accuracy
    }


def evaluate_model(model_name,model,tokenizer,test_data):
    print(f"\n{'='*70}")
    print(f"{model_name} 평가 시작")
    print(f"{'='*70}")

    results=[]

    for idx,item in enumerate(tqdm(test_data)):
        user_text=extract_user_text(item)
        assistant_text=extract_assistant_text(item)

        true_call=parse_tool_call(assistant_text)

        try:
            predicted_text=generate(
                model,
                tokenizer,
                user_text
            )
            predicted_call=parse_tool_call(predicted_text)

        except Exception as e:
            predicted_text=f"[ERROR] {e}"
            predicted_call=None

        results.append({
            "index":idx,
            "input":user_text,
            "expected":assistant_text,
            "prediction":predicted_text,
            "true_call":true_call,
            "predicted_call":predicted_call
        })

    return calculate_metrics(results),results


def print_metrics(name,metrics):
    print(f"\n{name}")
    print("-"*40)
    print(f"총 테스트 수       : {metrics['total']}")
    print(f"JSON 파싱 성공률   : {metrics['parse_rate']*100:.2f}%")
    print(f"Tool Name 정확도   : {metrics['tool_name_accuracy']*100:.2f}%")
    print(f"Argument 정확도    : {metrics['argument_accuracy']*100:.2f}%")
    print(f"전체 Exact Match   : {metrics['exact_match_accuracy']*100:.2f}%")

    print("\nParameter별 정확도")
    for key,value in metrics["parameter_accuracy"].items():
        print(f"  {key:8s}: {value*100:.2f}%")


def print_failures(results,max_items=20):
    failures=[
        r for r in results
        if not tool_call_exact_match(
            r["true_call"],
            r["predicted_call"]
        )
    ]

    print(f"\n오답 샘플 ({min(len(failures),max_items)}개):")

    for r in failures[:max_items]:
        print("\n"+"-"*70)
        print(f"[입력] {r['input']}")
        print(f"[정답] {r['expected']}")
        print(f"[예측] {r['prediction']}")


def main():
    print("="*70)
    print("        LoRA 드론 제어 모델 평가")
    print("="*70)

    print("\n테스트 데이터 로딩 중...")
    test_data=load_jsonl(TEST_DATASET_PATH)

    print(f"Test 데이터 : {len(test_data)}개")
    print(f"Test 경로   : {TEST_DATASET_PATH}")

    # Base Model
    base_tokenizer,base_model=load_base_model()

    base_metrics,base_results=evaluate_model(
        "BASE MODEL",
        base_model,
        base_tokenizer,
        test_data
    )

    print_metrics("BASE MODEL 결과",base_metrics)
    print_failures(base_results)

    del base_model
    del base_tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Base + LoRA
    lora_tokenizer,lora_model=load_lora_model()

    lora_metrics,lora_results=evaluate_model(
        "BASE + LORA MODEL",
        lora_model,
        lora_tokenizer,
        test_data
    )

    print_metrics("BASE + LORA 결과",lora_metrics)
    print_failures(lora_results)

    # 비교
    print("\n"+"="*70)
    print("최종 비교")
    print("="*70)

    print(
        f"Tool Name Accuracy : "
        f"{base_metrics['tool_name_accuracy']*100:.2f}%"
        f" -> "
        f"{lora_metrics['tool_name_accuracy']*100:.2f}%"
    )

    print(
        f"Argument Accuracy  : "
        f"{base_metrics['argument_accuracy']*100:.2f}%"
        f" -> "
        f"{lora_metrics['argument_accuracy']*100:.2f}%"
    )

    print(
        f"Exact Match        : "
        f"{base_metrics['exact_match_accuracy']*100:.2f}%"
        f" -> "
        f"{lora_metrics['exact_match_accuracy']*100:.2f}%"
    )

    # 결과 저장
    output={
        "config":{
            "model_id":MODEL_ID,
            "lora_dir":LORA_DIR,
            "test_dataset":TEST_DATASET_PATH,
            "test_size":len(test_data)
        },
        "base_model":{
            "metrics":base_metrics,
            "results":base_results
        },
        "lora_model":{
            "metrics":lora_metrics,
            "results":lora_results
        }
    }

    with open(RESULT_PATH,"w",encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"\n평가 결과 저장 완료: {RESULT_PATH}")


if __name__=="__main__":
    main()