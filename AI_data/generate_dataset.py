#!/usr/bin/env python3

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.append(os.path.abspath("../ros2_ws/src/llm_drone_control"))

from llm_drone_control.schema import SYSTEM_PROMPT
from templates import (
    TRAIN_COMMON, VAL_EXTRA, TEST_OOD, TEST_PREFIX, TEST_SUFFIX,
    TRAIN_CONNECTORS, VAL_CONNECTORS, TEST_CONNECTORS,
    NEG_TRAIN, NEG_VAL, NEG_TEST,
)

SEED = 42
random.seed(SEED)

NUM_TRAIN, NUM_VAL, NUM_TEST = 1250, 250, 350
NUM_COMPOUND = {"train": 600, "val": 120, "test": 200}
NUM_NEGATIVE = {"train": 200, "val": 40, "test": 60}
OUTPUT_PATH = "dataset"

DIST_RANGE_M = (0.3, 8.0)
DIST_RANGE_CM = (20, 500)
ANGLE_RANGE = (5, 180)
ALT_RANGE = (1.0, 5.0)

DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
UNITS = ["", "십", "백", "천"]
NATIVE = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열"}

AXES = {
    "forward": (1, 0, 0), "back": (-1, 0, 0),
    "left": (0, 1, 0), "right": (0, -1, 0),
    "up": (0, 0, 1), "down": (0, 0, -1),
}

DIRS = list(AXES)

TYPO = {
    # 1. 주요 단어 (명사 / 방향 / 동작 키워드)
    "전진": ["점진", "전징"], "후진": ["후잔", "후징"], "회전": ["회잔", "회정"],
    "이동": ["이덩", "이돈"], "이륙": ["이룩", "이륟"], "착륙": ["창륙", "착룩"],
    "상승": ["싱승", "상숭"], "하강": ["히강", "하간"], "방향": ["방햔", "방항"],
    "방향을": ["방햘을", "빙향을"], "선회": ["선화", "성회"], "복귀": ["복긔", "봇귀"],
    "비행": ["비행", "비행"], "설정": ["설정", "설정"], "시작": ["시작", "시작"],
    "개시": ["게시", "개사"], "종료": ["종로", "졸료"], "높이": ["노피", "높아"],
    "고도": ["고더", "거도"], "좌측": ["좌즉", "좌픅"], "우측": ["우즉", "우픅"],
    "시계": ["시게", "식계"], "반시계": ["반시게", "받시계"],

    # 2. 주요 어미 및 동사 변형
    "이동해": ["이동헤", "이동하"], "이동시켜": ["이동시캬", "이동시케"], "이동해줘": ["이동헤줘", "이동해쥐"],
    "이동시켜줘": ["이동시캬줘", "이동시켜쥐"], "움직": ["운직", "음직"], "움직여": ["움직요", "움직여어"],
    "움직여줘": ["움직여쥐", "움직여죠"], "움직여라": ["움직여러", "움지겨라"], "전진해": ["전진헤", "전진하"],
    "전진해줘": ["전진헤줘", "전진해쥐"], "후진해": ["후진햐", "후진헤"], "후진해줘": ["후진햐줘", "후진해쥐"],
    "회전해": ["회잔해", "회전헤"], "회전시켜": ["회전시캬", "회전시케"], "회전해줘": ["회전헤줘", "회전해쥐"],
    "돌려": ["돌료", "돌려어"], "돌려줘": ["돌려쥐", "돌려죠"], "돌아가": ["도라가", "돌아아가"],
    "돌아와": ["도라와", "돌아와아"], "전환": ["전횐", "저놘"], "전환해": ["전환헤", "전환하"],
    "전환해줘": ["전환헤줘", "전환해쥐"], "틀어": ["트러", "틀어어"], "틀어줘": ["틀어쥐", "틀어죠"],
    "선회해": ["선회헤", "선회하"], "복귀해": ["복귀헤", "복귀하"], "복귀해줘": ["복귀헤줘", "복귀해쥐"],
    "되돌아가": ["되돌아기", "되도라가"], "되돌아와": ["되돌아와아", "되도라와"], "되짚어가": ["되짚어기", "되지퍼가"],
    "따라가": ["따라아가", "따라기"], "올라가": ["오라가", "올러가"], "상승해": ["상승헤", "상승하"],
    "상승시켜": ["상승시캬", "상승시케"], "높여": ["높혀", "노펴"], "높여줘": ["높혀줘", "노펴줘"],
    "떠올라": ["떠올러", "떠오라"], "띄워": ["띄어", "띠워"], "띄워줘": ["띄어줘", "띠워줘"],
    "내려가": ["내려기", "내려가아"], "내려와": ["내려와아", "내러와"], "내려줘": ["내려쥐", "내려죠"],
    "하강해": ["하강헤", "하강하"], "낮춰": ["낮쳐", "나춰"], "낮춰줘": ["낮쳐줘", "나춰줘"],
    "이륙해": ["이룩해", "이륙헤"], "이륙시켜": ["이룩시캬", "이륙시케"], "이륙해줘": ["이룩해줘", "이륙헤줘"],
    "착륙해": ["착륙헤", "착륙하"], "착륙시켜": ["착륙시캬", "착륙시케"], "착륙해줘": ["착륙헤줘", "착륙해쥐"],
    "비행해": ["비행헤", "비행하"], "비행시켜": ["비행시캬", "비행시케"], "날아가": ["날러가", "나라가"],
    "설정해": ["설정헤", "설정하"], "설정해줘": ["설정헤줘", "설정해쥐"], "맞춰": ["마춰", "맞쳐"],
    "맞춰줘": ["마춰줘", "맞쳐줘"], "시작해": ["시작헤", "시작하"], "시작해줘": ["시작헤줘", "시작해쥐"],
    "개시해": ["개시헤", "게시해"], "종료해": ["종료헤", "종료하"], "끝내": ["끈내", "끝내어"],
    "끝내줘": ["끈내줘", "끝내쥐"], "가줘": ["가쥐", "가죠"], "가세요": ["가새요", "가세용"],
    "가자": ["가쟈", "가즈아"], "가라": ["가러", "가라아"],
    "해줘": ["헤줘", "해쥐"], "해주세요": ["해주새요", "해주세용"], "해라": ["헤라", "해라아"],
    "시켜": ["시캬", "시케"], "시켜줘": ["시캬줘", "시켜쥐"], "부탁해": ["부탁헤", "부탁하"],
    "부탁할게": ["부탁할께", "부탁할개"], "부탁드릴게": ["부탁드릴께", "부탁드릴개"]
}


def uniq(xs):
    return list(dict.fromkeys(xs))


def normalize(s):
    return " ".join(s.split()).strip()


def fmt_num(n):
    n = float(n)
    return str(int(n)) if n.is_integer() else f"{n:g}"


def int_kor(n):
    if n == 0:
        return "영"

    s, ns = "", str(abs(int(n)))
    length = len(ns)

    for i, c in enumerate(ns):
        digit = int(c)
        unit_index = length - i - 1

        if digit:
            if not (digit == 1 and unit_index > 0):
                s += DIGITS[digit]
            if unit_index < 4:
                s += UNITS[unit_index]

    return s


def number_to_korean(n):
    n = float(n)

    if n.is_integer():
        return int_kor(int(n))

    a, b = f"{n:.1f}".split(".")
    return f"{int_kor(int(a))}점{DIGITS[int(b)]}"


def angle_variants(a, k, split):
    if split == "train":
        return [str(a), k]
    if split == "val":
        return [str(a)]
    return [str(a), k]


def number_variants(n, unit, split="train"):
    n = float(n)
    s, k = fmt_num(n), number_to_korean(n)

    if unit == "m":
        if split == "train":
            base = [
                f"{s}m", f"{s} m", f"{s}미터", f"{s} 미터", f"{k}미터", f"{k} 미터",
                f"{s}m 정도", f"{s}m쯤", f"{s}m만큼", f"{s}미터 정도", f"{s}미터쯤",
                f"{s}미터만큼", f"약 {s}m", f"대략 {s}m", f"한 {s}m쯤",
                f"약 {s}미터", f"대략 {s}미터",
            ]
            if n.is_integer() and 1 <= int(n) <= 10:
                native = NATIVE[int(n)]
                base += [f"{native}미터", f"{native} 미터", f"{native}미터 정도", f"{native}미터쯤", f"{native}미터만큼"]

        elif split == "val":
            base = [f"{s}m 정도", f"약 {s}미터", f"{k}미터 정도", f"{s}미터만큼"]

        else:
            base = [
                f"{s}미터 남짓", f"{s}미터가량", f"{s}미터쯤 되는 거리",
                f"{s}미터 정도의 거리", f"{s}미터만큼의 거리", f"{k}미터가량", f"{k}미터 남짓",
            ]

        return uniq(base)

    if split == "train":
        base = [
            f"{int(n)}cm", f"{int(n)} cm", f"{int(n)}센티", f"{int(n)} 센티",
            f"{int(n)}센티미터", f"{int(n)} 센티미터", f"{k}센티", f"{k} 센티",
            f"{k}센티미터", f"{k} 센티미터", f"약 {int(n)}cm", f"대략 {int(n)}센티미터",
        ]

    elif split == "val":
        base = [f"{int(n)}cm 정도", f"약 {int(n)}센티", f"{int(n)}센티미터쯤", f"{k}센티미터 정도"]

    else:
        base = [
            f"{int(n)}센티 남짓", f"{int(n)}센티가량",
            f"{int(n)}센티미터쯤 되는 거리", f"{int(n)}센티미터 정도의 거리",
            f"{k}센티가량", f"{k}센티 남짓",
        ]

    return uniq(base)


def dist(unit=None, split="train"):
    unit=unit or random.choice(["m","cm"])
    if unit=="cm":
        n=random.randint(*DIST_RANGE_CM)
        return n,n/100,"cm"
    n=round(random.uniform(*DIST_RANGE_M),1)
    return n,n,"m"


def altitude(split="train"):
    return round(random.uniform(*ALT_RANGE),1)


def angle(split="train"):
    return random.randint(*ANGLE_RANGE)


def add_realistic_noise(text):
    # 1. TYPO 딕셔너리를 활용한 오타 주입 (5% 확률)
    if random.random() < 0.05:
        # 텍스트에 포함되어 있고 길이 2 이상인 키 검색
        keys = [k for k in TYPO if len(k) >= 2 and k in text and TYPO[k]]
        if keys:
            key = random.choice(keys)
            # 리스트에 정의된 오타 중 하나를 무작위 선택
            typo_val = random.choice(TYPO[key])
            text = text.replace(key, typo_val, 1)

    # 2. 띄어쓰기 오류 주입 (8% 확률)
    if random.random() < 0.08:
        parts = text.split()
        if len(parts) > 2:
            i = random.randrange(len(parts) - 1)
            parts[i] += parts.pop(i + 1)
            text = " ".join(parts)

    return text


ACTION_ORDER = list(TRAIN_COMMON)


def make_move_call(action, value):
    x, y, z = AXES[action]
    return "move", {"dx": round(x * value, 4), "dy": round(y * value, 4), "dz": round(z * value, 4), "d_yaw": 0.0}


def make_sample(text, tool, args):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": f"<tool_call>{json.dumps({'name': tool, 'arguments': args}, ensure_ascii=False)}</tool_call>"},
        ]
    }


def make_multi(text, calls):
    blocks = [f"<tool_call>{json.dumps({'name': tool, 'arguments': args}, ensure_ascii=False)}</tool_call>" for tool, args in calls]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "\n".join(blocks)},
        ]
    }


def is_nonzero_move(args, eps=1e-6):
    return any(abs(float(args.get(k, 0.0))) > eps for k in ("dx", "dy", "dz", "d_yaw"))


def clean_move_args(args):
    return {k: round(float(v), 4) for k, v in args.items()}


def merge_move_calls(calls):
    merged, has_move = [], False
    move_sum = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "d_yaw": 0.0}

    for tool, args in calls:
        if tool == "move":
            has_move = True
            for key in move_sum:
                move_sum[key] += args.get(key, 0.0)
        else:
            if has_move:
                if is_nonzero_move(move_sum):
                    merged.append(("move", clean_move_args(move_sum)))
                move_sum = {key: 0.0 for key in move_sum}
                has_move = False
            merged.append((tool, args))

    if has_move and is_nonzero_move(move_sum):
        merged.append(("move", clean_move_args(move_sum)))

    return merged


def make_refusal(text):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "지원하지 않는 명령입니다."},
        ]
    }


def add_test_style(text):
    if random.random() < 0.65:
        text = random.choice(TEST_PREFIX) + text
    if random.random() < 0.55:
        text += random.choice(TEST_SUFFIX)
    return text


def apply_split_style(text, split):
    if split == "train":
        return add_realistic_noise(text)
    if split == "test":
        return add_realistic_noise(add_test_style(text))
    return text


def get_templates(split, action):
    return {"train": TRAIN_COMMON, "val": VAL_EXTRA, "test": TEST_OOD}[split][action]


def generate_action(split, action):
    if action in ("land", "previous", "first", "reverse"):
        text = apply_split_style(random.choice(get_templates(split, action)), split)
        tool = {"land": "land", "previous": "goto_history", "first": "goto_history", "reverse": "reverse_plan"}[action]
        args = {"recall": action} if action in ("previous", "first") else {}
        return text, tool, args

    if action == "takeoff":
        a = altitude(split)
        v = random.choice(number_variants(a, "m", split))
        text = random.choice(get_templates(split, action)).format(v=v)
        return apply_split_style(text, split), "takeoff", {"altitude": a}

    if action in DIRS:
        raw, value, unit = dist(split=split)
        v = random.choice(number_variants(raw, unit, split))
        text = random.choice(get_templates(split, action)).format(v=v)
        return apply_split_style(text, split), *make_move_call(action, value)

    a = angle(split)
    angle_text = random.choice(angle_variants(a, number_to_korean(a), split))
    text = random.choice(get_templates(split, action)).format(a=angle_text)
    yaw = -float(a) if action == "cw" else float(a)
    return apply_split_style(text, split), "move", {"dx": 0.0, "dy": 0.0, "dz": 0.0, "d_yaw": yaw}


def basic(n, split, start_index=0):
    return [
        make_sample(text, tool, args)
        for i in range(n)
        for action in [ACTION_ORDER[(start_index + i) % len(ACTION_ORDER)]]
        for text, tool, args in [generate_action(split, action)]
    ]


def compound(n, split, start_index=0):
    out = []
    patterns = [
        ["takeoff", "move"], ["takeoff", "move", "land"], ["takeoff", "move", "move"],
        ["move", "move"], ["move", "move", "land"], ["takeoff", "move", "move", "land"],
    ]
    moves = ["forward", "back", "left", "right", "up", "down", "cw", "ccw"]
    connectors = {"train": TRAIN_CONNECTORS, "val": VAL_CONNECTORS, "test": TEST_CONNECTORS}[split]

    for _ in range(n):
        chosen = [random.choice(moves) if x == "move" else x for x in random.choice(patterns)]

        for i in range(1, len(chosen)):
            while chosen[i] == chosen[i - 1] and chosen[i] in moves:
                chosen[i] = random.choice(moves)

        texts, calls = [], []
        for action in chosen:
            text, tool, args = generate_action(split, action)
            texts.append(text)
            calls.append((tool, args))

        out.append(make_multi(random.choice(connectors).join(texts), merge_move_calls(calls)))

    return out


def negative(n, split, start_index=0):
    pool = {"train": NEG_TRAIN, "val": NEG_VAL, "test": NEG_TEST}[split]
    out, seen, tries = [], set(), 0

    while len(out) < n and tries < n * 300:
        text = random.choice(pool)

        if split == "train":
            text = add_realistic_noise(text)
        elif split == "test":
            text = add_realistic_noise(add_test_style(text))

        text = normalize(text)
        tries += 1

        if text in seen:
            continue

        seen.add(text)
        out.append(make_refusal(text))

    if len(out) < n:
        raise RuntimeError(f"{split} negative 데이터 생성 실패: {len(out)}/{n}")

    return out


def generate_deduplicated_set(generator, n, split):
    out, seen, tries = [], set(), 0

    while len(out) < n and tries < n * 300:
        sample = generator(1, split, start_index=tries)[0]
        tries += 1
        text = normalize(sample["messages"][1]["content"])

        if text in seen:
            continue

        seen.add(text)
        out.append(sample)

    if len(out) < n:
        raise RuntimeError(f"{split} 데이터 생성 중 중복 과다 발생: {len(out)}/{n}")

    return out


def sample_texts(data):
    return {normalize(sample["messages"][1]["content"]) for sample in data}


def remove_cross_split_duplicates(datasets):
    train_texts = sample_texts(datasets["train"])
    datasets["val"] = [s for s in datasets["val"] if normalize(s["messages"][1]["content"]) not in train_texts]
    tv_texts = train_texts | sample_texts(datasets["val"])
    datasets["test"] = [s for s in datasets["test"] if normalize(s["messages"][1]["content"]) not in tv_texts]


def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for sample in data:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def print_label_distribution(data, name):
    counter = Counter()

    for sample in data:
        content = sample["messages"][2]["content"]

        if "지원하지 않는 명령" in content:
            counter["negative"] += 1
            continue

        for action in ["takeoff", "land", "goto_history", "reverse_plan", "move"]:
            if f'"name": "{action}"' in content or f'"name":"{action}"' in content:
                counter[action] += 1

    print(f"\n[{name}] Total {len(data)} samples | Label counts: {dict(counter)}")


if __name__ == "__main__":
    datasets = {}

    for split, count in [("train", NUM_TRAIN), ("val", NUM_VAL), ("test", NUM_TEST)]:
        basic_data = generate_deduplicated_set(basic, count, split)
        compound_data = generate_deduplicated_set(compound, NUM_COMPOUND[split], split)
        negative_data = negative(NUM_NEGATIVE[split], split)

        data = basic_data + compound_data + negative_data
        random.shuffle(data)
        datasets[split] = data

    remove_cross_split_duplicates(datasets)

    for split, data in datasets.items():
        save_jsonl(data, f"{OUTPUT_PATH}/{split}.jsonl")
        print_label_distribution(data, split)

    print("\n========================================")
    print("Dataset Generation Completed Successfully")
    print(f"Train: {len(datasets['train'])} | Val: {len(datasets['val'])} | Test: {len(datasets['test'])}")
    print(f"Directory: ./{OUTPUT_PATH}/")
    print("========================================")