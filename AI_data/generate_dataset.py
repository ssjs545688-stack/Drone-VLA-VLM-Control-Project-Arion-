#!/usr/bin/env python3
import json,random
from collections import Counter
from pathlib import Path

try:
    from llm_drone_control.schema import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT="You are an AI assistant that controls a drone using tool calls."

SEED=42
random.seed(SEED)

# 데이터 수 = NUM + COMPOUND + NEGATIVE
NUM_TRAIN,NUM_VAL,NUM_TEST=600,120,180
NUM_COMPOUND={"train":300,"val":50,"test":70}
NUM_NEGATIVE={"train":80,"val":20,"test":30}
OUTPUT_PATH="dataset"

DIST_RANGE_M=(0.3,8.0)
DIST_RANGE_CM=(20,500)
ANGLE_RANGE=(5,180)
ALT_RANGE=(1.0,5.0)

DIGITS=["","일","이","삼","사","오","육","칠","팔","구"]
UNITS=["","십","백","천"]
NATIVE={1:"한",2:"두",3:"세",4:"네",5:"다섯",6:"여섯",7:"일곱",8:"여덟",9:"아홉",10:"열"}

AXES={
    "forward":(1,0,0),"back":(-1,0,0),"left":(0,1,0),
    "right":(0,-1,0),"up":(0,0,1),"down":(0,0,-1)
}
DIRS=list(AXES)

def uniq(xs): return list(dict.fromkeys(xs))

def fmt_num(n):
    n=float(n)
    return str(int(n)) if n.is_integer() else f"{n:g}"

def int_kor(n):
    if n==0:return "영"
    s="";ns=str(abs(int(n)));l=len(ns)
    for i,c in enumerate(ns):
        d=int(c);u=l-i-1
        if d:
            s+=("" if d==1 and u else DIGITS[d])
            if u<4:s+=UNITS[u]
    return s

def number_to_korean(n):
    n=float(n)
    if n.is_integer():return int_kor(int(n))
    a,b=f"{n:.1f}".split(".")
    return f"{int_kor(int(a))}점{DIGITS[int(b)]}"

# 숫자표현 다양하게
def number_variants(n,unit,level="train"):
    n=float(n);s=fmt_num(n);k=number_to_korean(n)

    if unit=="m":
        if level=="train":
            base=[
                f"{s}m",f"{s} m",f"{s}미터",f"{s} 미터",
                f"{k}미터",f"{k} 미터",
                f"{s}m 정도",f"{s}m쯤",f"{s}m만큼",
                f"{s}미터 정도",f"{s}미터쯤",f"{s}미터만큼",
                f"약 {s}m",f"대략 {s}m",f"한 {s}m쯤",
                f"약 {s}미터",f"대략 {s}미터"
            ]
            if n.is_integer() and 1<=int(n)<=10:
                x=NATIVE[int(n)]
                base += [
                    f"{x}미터",f"{x} 미터",
                    f"{x}미터 정도",f"{x}미터쯤",
                    f"{x}미터만큼"
                ]
        elif level=="val":
            base=[
                f"{s}m 정도",f"약 {s}미터",f"{s}미터쯤",
                f"{k}미터 정도",f"{s}미터만큼"
            ]
        else:
            base=[
                f"{s}미터 남짓",f"{s}미터가량",
                f"{s}미터쯤 되는 거리",f"{s}미터 정도의 거리",
                f"{s}미터만큼의 거리",f"{k}미터가량"
            ]
        return uniq(base)

    if level=="train":
        base=[
            f"{int(n)}cm",f"{int(n)} cm",
            f"{int(n)}센티",f"{int(n)} 센티",
            f"{int(n)}센티미터",f"{int(n)} 센티미터",
            f"{k}센티",f"{k} 센티",
            f"{k}센티미터",f"{k} 센티미터",
            f"약 {int(n)}cm",f"대략 {int(n)}센티미터"
        ]
    elif level=="val":
        base=[
            f"{int(n)}cm 정도",f"약 {int(n)}센티",
            f"{int(n)}센티미터쯤",f"{k}센티미터 정도"
        ]
    else:
        base=[
            f"{int(n)}센티 남짓",f"{int(n)}센티가량",
            f"{int(n)}센티미터쯤 되는 거리",
            f"{int(n)}센티미터 정도의 거리",
            f"{k}센티가량"
        ]
    return uniq(base)

def dist(unit=None):
    unit=unit or random.choice(["m","cm"])
    if unit=="cm":
        n=random.randint(*DIST_RANGE_CM)
        return n,n/100,"cm"
    n=round(random.uniform(*DIST_RANGE_M),1)
    return n,n,"m"

def altitude():return round(random.uniform(*ALT_RANGE),1)
def angle():return random.randint(*ANGLE_RANGE)

TYPO={
    "이동해":"이동해줘","복귀해":"복귀해줘","높여라":"높혀라",
    "회전해":"회잔해","전진해":"전진해라","후진해":"후진해줘",
    "움직여":"움직여줘","내려가":"내려와","올라가":"올라와",
    "돌려줘":"돌려쥐"
}

# tarin data 노이즈
def add_train_noise(s):
    if random.random()<.03:
        keys=[k for k in TYPO if k in s]
        if keys:
            k=random.choice(keys);s=s.replace(k,TYPO[k],1)
    if random.random()<.04:
        p=s.split()
        if len(p)>2:
            i=random.randrange(len(p)-1)
            p[i]+=p.pop(i+1)
            s=" ".join(p)
    return s

TRAIN_COMMON={
"takeoff":[
    "고도 {v}로 이륙해","{v} 높이로 이륙해",
    "{v} 고도로 이륙시켜","드론을 {v} 고도로 이륙시켜",
    "지상에서 {v} 고도로 이륙해","지면에서 이륙해서 {v} 고도로 올라가",
    "드론을 지상에서 띄워 {v} 고도로 이륙시켜",
    "이륙해서 고도 {v}까지 올라가","드론을 처음 이륙시켜 {v} 고도로 올려",
    "처음 이륙할 때 고도를 {v}로 설정해","드론을 {v} 고도로 띄워서 이륙시켜"
],
"land":[
    "착륙해","착륙해줘","지금 위치에 착륙해","현재 위치에 내려줘",
    "지면으로 착륙해라","바닥으로 내려가","지금 자리에서 내려",
    "안전하게 착륙해줘","현재 위치에서 착륙해","지상으로 내려와",
    "비행을 끝내고 착륙해","그대로 내려앉아"
],
"previous":[
    "직전 위치로 돌아가","방금 전 위치로 돌아가","이전 위치로 복귀해",
    "바로 전 위치로 가줘","아까 있던 위치로 돌아가",
    "직전에 있던 곳으로 돌아와","한 단계 전 위치로 돌아가",
    "방금 이동하기 전 위치로 가","이전 지점으로 되돌아가"
],
"first":[
    "처음 위치로 돌아가","최초 위치로 복귀해","시작 위치로 돌아가",
    "처음 출발했던 곳으로 가","원래 위치로 돌아가","출발 위치로 복귀해",
    "비행 시작점으로 돌아가","처음 있던 곳으로 돌아와",
    "초기 위치로 복귀해","처음 위치로 되돌아가"
],
"reverse":[
    "왔던 경로로 되돌아가","지나온 경로를 역으로 돌아가",
    "이동했던 경로를 거꾸로 따라가","왔던 길을 반대로 돌아가",
    "지나온 동선을 따라 되돌아가","이전에 이동한 경로를 역순으로 가",
    "방금까지 왔던 경로를 되짚어가"
],
"forward":[
    "앞으로 {v} 전진해","{v} 앞으로 가줘","전방으로 {v} 이동해",
    "앞쪽으로 {v} 가","앞으로 {v}미터 가줘","기체 앞쪽으로 {v} 이동해",
    "전진해서 {v}만큼 가","앞으로 {v}만큼 움직여","정면으로 {v} 이동해",
    "기수가 향한 방향으로 {v} 이동해","기체 머리 방향으로 {v} 가",
    "코가 가리키는 방향으로 {v} 이동해","기체가 바라보는 쪽으로 {v} 가",
    "기체가 향하는 방향으로 {v} 이동해", "기체가 보는 방향으로 {v} 전진해"
],
"back":[
    "뒤로 {v} 물러나","{v} 뒤로 가줘","후방으로 {v} 이동해",
    "뒤쪽으로 {v} 가","뒤로 {v}미터 가줘","후진해서 {v}만큼 가",
    "뒤로 {v}만큼 움직여","기체 뒤쪽으로 {v} 이동해",
    "기체 뒤로 {v} 가","기수 반대 방향으로 {v} 이동해",
    "꼬리 방향으로 {v} 가","기체가 바라보는 방향의 반대로 {v} 이동해",
    "기체가 향하는 방향의 반대로 {v} 이동해", "기체가 보는 방향의 반대쪽으로 {v} 가"
],
"left":[
    "왼쪽으로 {v} 이동해","{v} 왼쪽으로 가줘","좌측으로 {v} 가",
    "왼쪽으로 {v} 움직여","왼쪽으로 {v}미터 이동해",
    "좌측으로 {v}만큼 이동해","왼쪽으로 {v} 이동해줘",
    "기체 왼쪽으로 {v} 가","좌현으로 {v} 이동해",
    "기체 기준 왼쪽으로 {v} 이동해","왼쪽 측면으로 {v} 이동해"
],
"right":[
    "오른쪽으로 {v} 이동해","{v} 오른쪽으로 가줘","우측으로 {v} 가",
    "오른쪽으로 {v} 움직여","오른쪽으로 {v}미터 이동해",
    "우측으로 {v}만큼 이동해","오른쪽으로 {v} 이동해줘",
    "기체 오른쪽으로 {v} 가","우현으로 {v} 이동해",
    "기체 기준 오른쪽으로 {v} 이동해","오른쪽 측면으로 {v} 이동해"
],
"up":[
    "위로 {v} 상승해","{v}만큼 위로 올라가","위쪽으로 {v} 이동해",
    "현재 고도에서 {v}만큼 상승해","지금 위치에서 {v}만큼 올라가",
    "현재 위치에서 위로 {v} 이동해","현재 고도에서 {v} 높여",
    "기체를 현재 위치에서 {v}만큼 들어올려",
    "비행 중인 상태에서 {v}만큼 상승해","현재 위치 기준으로 {v} 올라가",
    "공중에서 {v}만큼 더 올라가","이미 떠 있는 상태에서 {v}만큼 상승해"
],
"down":[
    "아래로 {v} 내려가","{v}만큼 아래로 내려가","하강해서 {v} 가",
    "아래쪽으로 {v} 이동해","수직으로 {v} 내려가","{v}만큼 하강해",
    "고도를 {v} 낮춰","아래로 {v}미터 내려가"
],
"cw":[
    "시계 방향으로 {a}도 회전해",
    "오른쪽으로 {a}도 회전해",
    "시계 방향으로 {a}도 돌려",
    "우회전해서 {a}도 돌아",
    "{a}도만큼 시계 방향으로 돌아",
    "기체를 오른쪽으로 {a}도 돌려",
    "오른쪽으로 {a}도 틀어",
    "우측으로 {a}도 회전해",
    "기체 방향을 오른쪽으로 {a}도 바꿔"
],
"ccw":[
    "반시계 방향으로 {a}도 회전해",
    "왼쪽으로 {a}도 회전해",
    "반시계 방향으로 {a}도 돌려",
    "좌회전해서 {a}도 돌아",
    "{a}도만큼 반시계 방향으로 돌아",
    "기체를 왼쪽으로 {a}도 돌려",
    "왼쪽으로 {a}도 틀어",
    "좌측으로 {a}도 회전해",
    "기체 방향을 왼쪽으로 {a}도 바꿔"
]}

VAL_EXTRA={
"takeoff":[
    "{v} 높이로 이륙시켜","고도 {v}로 이륙해",
    "{v} 고도로 이륙하게 해","지상에서 {v} 고도로 이륙해",
    "처음 이륙해서 고도 {v}까지 올라가"
],
"land":[
    "현재 자리에서 지상으로 내려","바로 지상에 내려줘",
    "현재 위치에서 지면으로 내려가","비행을 멈추고 내려와"
],
"previous":[
    "방금 전 자리로 다시 가","바로 전에 있던 곳으로 돌아가",
    "이전 지점으로 다시 이동해"
],
"first":[
    "비행을 시작한 위치로 돌아가","출발했던 지점으로 돌아와",
    "처음 출발점으로 다시 가"
],
"reverse":[
    "지나온 길을 따라 다시 돌아가","이동했던 길을 반대로 따라가",
    "왔던 경로를 거꾸로 되짚어가"
],
"forward":[
    "앞쪽으로 {v} 가줘","정면 쪽으로 {v} 이동해",
    "전방을 향해 {v}만큼 가"
],
"back":[
    "뒤쪽으로 {v} 물러가","후방으로 {v}만큼 이동해",
    "뒤편을 향해 {v} 가줘"
],
"left":[
    "왼편으로 {v} 가줘","좌측 방향으로 {v} 이동해",
    "왼쪽 편으로 {v}만큼 가"
],
"right":[
    "오른편으로 {v} 가줘","우측 방향으로 {v} 이동해",
    "오른쪽 편으로 {v}만큼 가"
],
"up":[
    "현재 위치에서 위쪽으로 {v}만큼 올라가",
    "비행 중인 상태에서 {v}만큼 상승해",
    "현재 고도에서 {v}만큼 더 올라가",
    "지금 위치에서 수직으로 {v} 이동해",
    "이미 떠 있는 상태에서 {v}만큼 상승해"
],
"down":[
    "아래쪽으로 {v}만큼 내려가","{v} 높이만큼 하강해",
    "수직으로 아래로 {v} 이동해"
],
"cw":[
    "오른쪽으로 {a}도 돌아","시계 방향으로 {a}도 돌려줘",
    "{a}도 우회전해"
],
"ccw":[
    "왼쪽으로 {a}도 돌아","반시계 방향으로 {a}도 돌려줘",
    "{a}도 좌회전해"
]}

# Train/Val에서 직접 사용하지 않는 OOD 표현만 사용
TEST_OOD={
"takeoff":[
    "{v} 고도로 이륙시켜봐",
    "고도 {v}에서 비행을 시작해",
    "지상에서 {v} 고도로 이륙해",
    "처음 비행을 {v} 고도에서 시작해",
    "이륙 고도를 {v}로 설정해",
    "지면을 떠나 {v} 고도로 이륙해"
],
"land":[
    "비행을 종료하고 착륙해",
    "현재 비행을 끝내고 지상에 내려",
    "기체를 지상에 착지시켜",
    "비행을 마치고 땅으로 내려가",
    "현재 위치에서 비행을 종료해"
],
"previous":[
    "바로 앞서 있던 위치로 이동해",
    "직전에 머물렀던 위치를 다시 찾아가",
    "한 단계 이전 위치로 되돌아가",
    "가장 최근에 이동하기 전 위치로 돌아가",
    "직전 위치를 다시 방문해"
],
"first":[
    "비행을 시작했던 위치로 되돌아가",
    "처음 출발했던 좌표로 돌아가",
    "비행 시작 지점으로 복귀해",
    "최초 출발 위치를 다시 찾아가",
    "처음 비행을 시작한 곳으로 돌아가"
],
"reverse":[
    "지금까지 이동한 경로를 반대 순서로 따라가",
    "앞서 지나온 경로를 역방향으로 되짚어가",
    "이전에 이동한 순서를 거꾸로 따라가",
    "지금까지 지나온 경로를 역순으로 이동해",
    "방금까지의 이동 경로를 반대로 되돌아가"
],
"forward":[
    "기수가 향하는 방향으로 {v} 이동해",
    "기체의 전방을 따라 {v} 전진해",
    "기체 앞쪽을 기준으로 {v}만큼 이동해",
    "기체 정면 방향으로 {v}만큼 전진해",
    "기체의 머리 방향을 따라 {v} 이동해"
],
"back":[
    "기수가 향하는 방향의 반대로 {v} 이동해",
    "기체의 후방을 따라 {v}만큼 이동해",
    "기체 뒤쪽을 기준으로 {v} 후진해",
    "기체 정면의 반대 방향으로 {v} 이동해",
    "기체 꼬리가 향한 쪽으로 {v} 이동해"
],
"left":[
    "기체의 좌현 방향으로 {v} 이동해",
    "기체 왼편을 따라 {v}만큼 이동해",
    "기체 기준 좌측으로 {v} 이동해",
    "기체의 왼쪽 측면으로 {v}만큼 이동해",
    "기체의 좌측 방향을 따라 {v} 이동해"
],
"right":[
    "기체의 우현 방향으로 {v} 이동해",
    "기체 오른편을 따라 {v}만큼 이동해",
    "기체 기준 우측으로 {v} 이동해",
    "기체의 오른쪽 측면으로 {v}만큼 이동해",
    "기체의 우측 방향을 따라 {v} 이동해"
],
"up":[
    "현재 비행 위치에서 {v}만큼 고도를 높여",
    "현재 고도에서 {v}만큼 더 상승해",
    "기체의 현재 높이에서 {v}만큼 올라가",
    "비행 중인 상태에서 {v}만큼 위로 이동해",
    "현재 위치를 유지한 채 고도만 {v}만큼 높여"
],
"down":[
    "현재 비행 위치에서 {v}만큼 고도를 낮춰",
    "현재 고도에서 {v}만큼 더 내려가",
    "기체의 현재 높이에서 {v}만큼 하강해",
    "비행 중인 상태에서 {v}만큼 아래로 이동해",
    "현재 위치를 유지한 채 고도만 {v}만큼 낮춰"
],
"cw":[
    "기체의 방향을 오른쪽으로 {a}도 전환해",
    "현재 진행 방향에서 시계 방향으로 {a}도 회전해",
    "기체의 진행 방향을 우측으로 {a}도 바꿔",
    "현재 방향에서 오른쪽으로 {a}도 선회해",
    "기체의 방향을 시계 방향으로 {a}도 틀어"
],
"ccw":[
    "기체의 방향을 왼쪽으로 {a}도 전환해",
    "현재 진행 방향에서 반시계 방향으로 {a}도 회전해",
    "기체의 진행 방향을 좌측으로 {a}도 바꿔",
    "현재 방향에서 왼쪽으로 {a}도 선회해",
    "기체의 방향을 반시계 방향으로 {a}도 틀어"
]}

TEST_PREFIX=[
    "야 , ","드론아 ","자, ","음... ","오케이, ",
    "지금 바로 ","일단 ","헤이 , ","좋아, ","잠깐, "
]
TEST_SUFFIX=[
    " 부탁해"," 빨리 해"," ㄱㄱ"," 바로 해",
    " 처리해"," 알겠지?"," 해버려"," 좀 해줘"
]

TRAIN_CONNECTORS=[" 그리고 "," 한 다음 "," 하고 나서 "," 이동한 뒤 "," 그 다음에 "," 이어서 "]
VAL_CONNECTORS=[" 그리고 "," 한 뒤 "," 이어서 "," 하고 "]
TEST_CONNECTORS=[" 그리고 "," 한 다음 "," 이어서 "," 그 뒤에 "," 이후 바로 "," 하고 나서 "]

NEG_TRAIN=[
    "오늘 날씨 어때?","내일 날씨 알려줘","노래 좀 틀어줘","음악 재생해줘",
    "커피 한 잔 타줘","물 좀 가져다줘","지금 몇 시야?","현재 시간 알려줘",
    "라면 끓이는 법 알려줘","김치찌개 레시피 알려줘","사진 한 장 찍어봐",
    "동영상 촬영해줘","주변 사람에게 메시지 보내줘","친구에게 문자 보내줘",
    "인터넷 검색해줘","웹에서 정보 찾아줘","뉴스 알려줘","오늘 뉴스 검색해줘",
    "주변 사람에게 전화해줘","알람 설정해줘","타이머 설정해줘","계산해줘",
    "오늘 일정 알려줘","내일 일정 확인해줘","메일 보내줘","파일 열어줘",
    "컴퓨터 종료해줘","불 좀 꺼줘","에어컨 켜줘","근처 식당 찾아줘",
    "사진을 분석해줘","사람 얼굴을 인식해줘","주변을 촬영해줘",
    "드론 배터리 상태 알려줘","드론 카메라 영상을 보여줘","주변 장애물을 알려줘",
    "GPS 위치 알려줘","현재 위치를 지도에 표시해줘","비행 기록을 보여줘"
]

NEG_VAL=[
    "날씨 알려줘","내일 비 와?","노래 재생해줘","노래 불러줘",
    "현재 시간 말해줘","지금은 몇 시야?","커피 만들어줘","물 가져다줘",
    "요리법 알려줘","사진 찍어줘","동영상 찍어줘","뉴스 검색해줘",
    "인터넷에서 찾아줘","문자 보내줘","전화 걸어줘","알람 맞춰줘",
    "일정 확인해줘","메일 작성해줘","근처 맛집 찾아줘","주변 사람 얼굴을 인식해줘",
    "드론의 카메라 화면을 보여줘","배터리 잔량 알려줘","GPS 좌표 알려줘",
    "비행 기록 확인해줘","주변 상황을 설명해줘","장애물 위치 알려줘"
]

NEG_TEST=[
    "오늘 비 오냐?","내일 날씨 보고 알려줘","노래 하나 틀어봐","음악 좀 재생해",
    "몇 시인지 확인해줘","현재 시간을 알려줘","라면 레시피 좀","요리 방법 알려줘",
    "불 좀 꺼줄래?","에어컨 켜줘","자율주행 자동차는 언제 나오냐?",
    "내일 날씨 보고 일정도 짜줘","근처 맛집 추천해줘","사진 찍어서 보내줘",
    "드론 카메라로 사람 얼굴 인식해줘","주변 상황을 보고 알아서 판단해줘",
    "인터넷에서 가장 가까운 카페 찾아줘","친구한테 문자 보내줘",
    "전화 좀 걸어줘","알람 하나 맞춰줘","최신 뉴스 찾아줘","메일 하나 보내줘",
    "드론 배터리 상태 확인해줘","GPS 좌표 확인해줘","비행 기록 보여줘",
    "주변 장애물 위치 알려줘","카메라 화면 확인해줘"
]

ACTION_ORDER=list(TRAIN_COMMON)

def make_move_call(action,value):
    x,y,z=AXES[action]
    return "move",{"dx":x*value,"dy":y*value,"dz":z*value,"d_yaw":0.0}

def make_sample(text,tool,args):
    return {"messages":[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":text},
        {"role":"assistant","content":f'<tool_call>{json.dumps({"name":tool,"arguments":args},ensure_ascii=False)}</tool_call>'}
    ]}

def make_multi(text,calls):
    blocks=[f'<tool_call>{json.dumps({"name":tool,"arguments":args},ensure_ascii=False)}</tool_call>' for tool,args in calls]
    return {"messages":[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":text},
        {"role":"assistant","content":"\n".join(blocks)}
    ]}

def make_refusal(text):
    return {"messages":[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":text},
        {"role":"assistant","content":"지원하지 않는 명령입니다."}
    ]}

def add_test_style(text,guarantee=False):
    if guarantee or random.random()<.65:text=random.choice(TEST_PREFIX)+text
    if guarantee or random.random()<.55:text+=random.choice(TEST_SUFFIX)
    return text

def generate_action(split,action):
    if action in ("land","previous","first","reverse"):
        t=random.choice(TRAIN_COMMON[action] if split=="train" else VAL_EXTRA[action] if split=="val" else TEST_OOD[action])
        if split=="test":t=add_test_style(t)
        elif split=="train":t=add_train_noise(t)
        tool={"land":"land","previous":"goto_history","first":"goto_history","reverse":"reverse_plan"}[action]
        if action=="land" or action=="reverse":
            args={}
        else:
            args={"recall":action}
        return t,tool,args

    if action=="takeoff":
        a=altitude()
        v=random.choice(number_variants(a,"m",split))
        t=random.choice(TRAIN_COMMON[action] if split=="train" else VAL_EXTRA[action] if split=="val" else TEST_OOD[action]).format(v=v)
        if split=="test":t=add_test_style(t)
        elif split=="train":t=add_train_noise(t)
        return t,"takeoff",{"altitude":a}

    if action in DIRS:
        raw,value,unit=dist()
        v=random.choice(number_variants(raw,unit,split))
        t=random.choice(TRAIN_COMMON[action] if split=="train" else VAL_EXTRA[action] if split=="val" else TEST_OOD[action]).format(v=v)
        if split=="test":t=add_test_style(t)
        elif split=="train":t=add_train_noise(t)
        return t,*make_move_call(action,value)

    a=angle();k=number_to_korean(a)
    if split=="train":t=random.choice(TRAIN_COMMON[action]).format(a=random.choice([a,k]))
    elif split=="val":t=random.choice(VAL_EXTRA[action]).format(a=random.choice([a,k]))
    else:t=random.choice(TEST_OOD[action]).format(a=random.choice([a,k]));t=add_test_style(t)
    if split=="train":t=add_train_noise(t)
    yaw=-a if action=="cw" else a
    return t,"move",{"dx":0.0,"dy":0.0,"dz":0.0,"d_yaw":yaw}

# 단일 명령
def basic(n,split,start_index=0):
    out=[]
    for i in range(n):
        action=ACTION_ORDER[(start_index+i)%len(ACTION_ORDER)]
        text,tool,args=generate_action(split,action)
        out.append(make_sample(text,tool,args))
    return out

# 복합 명령
def compound(n,split,start_index=0):
    out=[]
    patterns=[
        ["takeoff","move"],["takeoff","move","land"],
        ["takeoff","move","move"],["move","move"],
        ["move","move","land"],["takeoff","move","move","land"]
    ]
    moves=["forward","back","left","right","up","down","cw","ccw"]
    connectors={"train":TRAIN_CONNECTORS,"val":VAL_CONNECTORS,"test":TEST_CONNECTORS}[split]

    for _ in range(n):
        chosen=random.choice(patterns)
        chosen=[random.choice(moves) if x=="move" else x for x in chosen]
        for i in range(1,len(chosen)):
            while chosen[i]==chosen[i-1] and chosen[i] in moves:
                chosen[i]=random.choice(moves)

        texts=[];calls=[]
        for a in chosen:
            text,tool,args=generate_action(split,a)
            texts.append(text);calls.append((tool,args))

        text=random.choice(connectors).join(texts)
        if split=="test" and random.random()<.8:
            text=add_test_style(text)
        elif split=="train":
            text=add_train_noise(text)

        out.append(make_multi(text,calls))
    return out

def negative(n,split,start_index=0):
    pool={"train":NEG_TRAIN,"val":NEG_VAL,"test":NEG_TEST}[split]
    out=[];seen=set();tries=0
    while len(out)<n and tries<n*100:
        t=random.choice(pool)
        if split=="train":t=add_train_noise(t)
        elif split=="test":t=add_test_style(t)
        t=normalize(t);tries+=1
        if t in seen:continue
        seen.add(t);out.append(make_refusal(t))
    if len(out)<n:raise RuntimeError(f"{split} negative 생성 실패: {len(out)}/{n}")
    return out

def normalize(s):
    return " ".join(s.split()).strip()

# 중복 제거
def generate_deduplicated_set(generator,n,split):
    out=[];seen=set();tries=0
    while len(out)<n and tries<n*100:
        sample=generator(1,split,start_index=tries)[0];tries+=1
        text=normalize(sample["messages"][1]["content"])
        if text in seen:continue
        seen.add(text);out.append(sample)
    if len(out)<n:raise RuntimeError(f"{split} 중복 제거 후 {n}개 생성 실패: {len(out)}개")
    return out

def sample_texts(data):
    return {normalize(x["messages"][1]["content"]) for x in data}

def remove_cross_split_duplicates(datasets):
    splits=["train","val","test"]
    sets={s:sample_texts(datasets[s]) for s in splits}
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            dup=sets[a]&sets[b]
            if dup:raise RuntimeError(f"{a}/{b} 데이터 중복 발견: {len(dup)}개\n{next(iter(dup))}")

def save_jsonl(data,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for x in data:f.write(json.dumps(x,ensure_ascii=False)+"\n")

def print_label_distribution(data,name):
    c=Counter()
    for x in data:
        s=x["messages"][2]["content"]
        if "지원하지 않는 명령" in s:c["negative"]+=1
        else:
            for action in ["takeoff","land","goto_history","reverse_plan","move"]:
                if f'"name": "{action}"' in s:c[action]+=1
    print(f"\n[{name}] {len(data)} samples")
    print(dict(c))

def print_examples(data,name,k=5):
    print(f"\n===== {name} examples =====")
    for x in random.sample(data,min(k,len(data))):
        print("USER:",x["messages"][1]["content"])
        print("OUT :",x["messages"][2]["content"])
        print()

if __name__=="__main__":
    datasets={}
    for split,n in [("train",NUM_TRAIN),("val",NUM_VAL),("test",NUM_TEST)]:
        basic_data=generate_deduplicated_set(basic,n,split)
        compound_data=generate_deduplicated_set(compound,NUM_COMPOUND[split],split)
        negative_data=negative(NUM_NEGATIVE[split],split)
        data=basic_data+compound_data+negative_data
        random.shuffle(data)
        datasets[split]=data

    remove_cross_split_duplicates(datasets)

    for split,data in datasets.items():
        save_jsonl(data,f"{OUTPUT_PATH}/{split}.jsonl")
        print_label_distribution(data,split)
        print_examples(data,split)

    print("\n========================================")
    print("Dataset generation complete")
    print(f"Train: {len(datasets['train'])}")
    print(f"Val  : {len(datasets['val'])}")
    print(f"Test : {len(datasets['test'])}")
    print(f"Output: {OUTPUT_PATH}/")
    print("========================================")