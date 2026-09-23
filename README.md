# Drone-VLA-VLM-Control-Project-Arion

## 1. 프로젝트 개요

이 프로젝트는 자연어 명령을 이해하는 로컬 LLM과 PX4 기반 비행 제어를 결합하여, 드론을 자율적으로 제어하는 시스템을 구현하는 연구 프로젝트입니다.

사용자의 자연어 명령은 로컬 Qwen3 모델이 구조화된 Tool Call 형태로 변환하고, 이를 ROS 2 노드가 PX4 Offboard 제어 명령으로 실행합니다. 현재 단계는 외부 API 없이 로컬 모델만 사용하여 추론을 수행하는 시뮬레이션 검증 중심의 구현입니다.

### 핵심 목표

- 로컬에서 동작하는 자연어 드론 제어
- PX4 SITL + Gazebo 기반 시뮬레이션 검증
- Qwen3 기반 Tool Calling 구조
- INT4 AWQ 양자화 모델을 활용한 온디바이스 추론
- ROS 2 서비스/토픽 기반 명령 처리

### 프로젝트 정보

- **조장:** 신현수
- **VLA, 파인튜닝:** 정진한, 최지호
- **함수 스키마 담당:** 김태형, 정원혁
- **지도강사:** 박승휘

---

## 2. 문서 구조

- [프로젝트 개요](#1-프로젝트-개요)
- [저장소 구성](#3-저장소-구성)
- [시스템 아키텍처](#4-시스템-아키텍처)
- [실행 흐름](#5-실행-흐름)
- [개발 환경](#6-개발-환경)
- [필수 요구 사항](#7-필수-요구-사항)
- [설치 및 빌드](#8-설치-및-빌드)
- [실행 가이드](#9-실행-가이드)
- [명령 처리 방식](#10-명령-처리-방식)
- [좌표계와 안전 주의사항](#11-좌표계와-안전-주의사항)
- [모델 학습 및 평가](#12-모델-학습-및-평가)
- [참고 문서](#13-참고-문서)
- [향후 계획](#14-향후-계획)
- [Git Commit Convention](#15-git-commit-convention)

---

## 3. 저장소 구성

```text
Drone-VLA-VLM-Control-Project-Arion-
├── AI_data/                    # 데이터 생성, 학습, 평가 스크립트
│   ├── evaluate_model.py
│   ├── generate_dataset.py
│   ├── merge_lora.py
│   ├── quantize_awq.py
│   ├── templates.py
│   ├── train_lora_with_validation.py
│   └── dataset/
├── docs/                       # 내부 문서 및 분석 보고서
│   ├── drone_control_test_commands.md
│   ├── llm_drone_control_code_interaction_report.md
│   └── px4_msgs_details.md
├── images/
├── log/                        # 빌드 및 실행 로그
├── models/                     # 학습/병합/양자화 모델 저장소
│   ├── Qwen3-1.7B/
│   ├── finetuned_qwen3-1.7B_drone_lora/
│   ├── qwen3-1.7B-drone-merged/
│   └── qwen3-1.7B-drone-int4-awq/
├── ros2_ws/
│   └── src/
│       ├── guide_interfaces/
│       ├── llm_drone_control/
│       └── px4_msgs/
├── test/                       # 비행 제어/스키마 테스트
├── train/                      # 학습 관련 스크립트, 데이터 및 MLflow 결과
├── CHANGELOG.md
├── LICENSE
├── README.md
├── requirements.txt
└── reference_src/              # 참고 구현 소스
└── world/                      # 가제보 월드 파일
│   └── red_target.sdf
```

### 현재 구현된 핵심 패키지

```text
ros2_ws/src/llm_drone_control/
├── config/
│   └── model.yaml
│   └── camera.yaml
├── launch/
│   └── drone_control.launch.py
│   └── vision.launch.py
├── llm_drone_control/
│   ├── flight_controller.py
│   ├── llm_service.py
│   ├── schema.py
│   └── smartphone_bridge.py
│   └── object_detector.py
├── package.xml
├── setup.py
├── setup.cfg
└── resource/
```

> 현재 코드베이스 기준으로 실제 동작 노드는 `llm_service`, `flight_controller`, `smartphone_bridge` 세 가지이며, `drone_dashboard.py`는 현재 저장소에 포함되어 있지 않습니다.

> `object_detector.py`는 기능은 구현하였으나 llm과 연결은 하지 못하였습니다.

---

## 4. 시스템 아키텍처

### 구성 요소

| 구분 | 위치 | 설명 |
|---|---|---|
| 온디바이스 제어부 | 개발 PC 또는 온보드 컴퓨터 | `llm_drone_control` ROS 2 패키지, 로컬 Qwen3 INT4 AWQ 모델 |
| 시뮬레이션 PC | 별도 PC | PX4 SITL, Gazebo, MicroXRCE-DDS Agent |
| 지상 관제 | 선택 사항 | QGroundControl v5.0.4 |
| 모델 학습 | GPU 서버 | `AI_data/`, `train/`, `models/` |

### 역할 분리

- **LLM 계층:** 자연어 명령을 Tool Call 형태로 변환
- **ROS 2 계층:** `/llm` 서비스, `/llm_response` 토픽, `/voice_command` 입력 처리
- **비행 제어 계층:** `flight_controller.py`가 Tool Call을 검증하고 PX4 Offboard 모드로 전환
- **PX4 계층:** `VehicleCommand`, `OffboardControlMode`, `TrajectorySetpoint` 발행

---

### 함수 스키마(Function Schema)와 Tool Calling 계약

이 프로젝트에서 LLM은 사용자 명령을 자유 텍스트로 직접 출력하지 않고, 미리 정의된 함수 스키마에 맞는 Tool Call JSON으로 변환합니다. 함수 스키마는 [ros2_ws/src/llm_drone_control/llm_drone_control/schema.py](ros2_ws/src/llm_drone_control/llm_drone_control/schema.py)에 정의되어 있으며, 모델이 생성할 수 있는 동작 범위를 제한하고 안전성을 확보하는 역할을 합니다.

핵심 개념은 다음과 같습니다.

- **Tool schema**: 사용할 수 있는 함수 이름, 설명, 인자 타입, 필수 인자 목록을 정의
- **LLM system prompt**: 모델이 명령을 해석할 때 따라야 할 좌표계, 부호 기준, 출력 형식 규칙을 부여
- **Tool Call output format**: `<tool_call>{"name": "move", "arguments": {...}}</tool_call>` 형식으로만 응답하도록 강제
- **Safety constraint**: 지원하지 않는 명령은 Tool Call을 생성하지 않고, 고정된 거절 문구를 반환

현재 정의된 함수는 다음과 같습니다.

| Tool 이름 | 입력 인자 | 의미 |
|---|---|---|
| `takeoff` | `altitude` | 지정 고도까지 이륙 |
| `move` | `dx`, `dy`, `dz`, `d_yaw` | 상대 이동과 회전 명령 |
| `land` | 없음 | 현재 위치에서 착륙 |
| `goto_history` | `recall` (`previous` 또는 `first`) | 과거 위치로 복귀 |
| `reverse_plan` | 없음 | 이동 경로를 역순으로 복귀 |

예시 Tool Call은 아래와 같습니다.

```text
<tool_call>{"name":"move","arguments":{"dx":2.0,"dy":0.0,"dz":0.0,"d_yaw":0.0}}</tool_call>
```

이 구조를 통해 LLM이 임의로 비행 좌표를 추측하지 않고, 사전에 정해진 동작 규칙을 따라야만 합니다. 특히 `dx`, `dy`, `dz`, `d_yaw`는 드론의 좌표계와 부호 규칙을 반영하도록 설계되어 있으며, `SYSTEM_PROMPT`에 정의된 표준은 모델이 동일한 의미로 해석하도록 보장합니다.

---

## 5. 실행 흐름

```text
사용자 명령
   │
   ├─ 텍스트/음성 입력
   │
   └─ /llm 서비스 또는 /voice_command 토픽
         │
         v
   llm_service.py
   └─ 로컬 Qwen3 추론
         │
         v
   <tool_call>{...}</tool_call>
         │
         v
   flight_controller.py
   └─ Tool Call 검증, 상태머신 처리, PX4 Offboard 제어
         │
         v
   PX4 Autopilot / SITL / Gazebo
```

### 주요 모듈

| 파일 | 역할 |
|---|---|
| `schema.py` | Tool 스키마와 시스템 프롬프트 정의 |
| `llm_service.py` | 로컬 모델 추론, Tool Call 추출, 응답 발행 |
| `flight_controller.py` | Tool Call 검증, 상태 전이, PX4 메시지 전송 |
| `smartphone_bridge.py` | FastAPI 웹 서버와 ROS 2 토픽 연결 |
| `launch/drone_control.launch.py` | 통합 런치 구성 파일 |
| `config/model.yaml` | 로컬 모델 경로 및 추론 옵션 설정 |

---

## 6. 개발 환경

| 구분 | 사용 환경 |
|---|---|
| 호스트 OS | Ubuntu 22.04 64-bit |
| 개발 언어 | Python 3.10 |
| ROS 2 | ROS 2 Humble (`rclpy`, `colcon`) |
| AI/ML | PyTorch, Transformers, Accelerate, PEFT |
| 스마트폰 입력 | FastAPI, Uvicorn |
| 비행 제어 | PX4 Autopilot, `px4_msgs`, PX4 Offboard API |
| 시뮬레이터 | PX4 SITL + Gazebo |
| 통신/관제 | MicroXRCE-DDS Agent, QGroundControl v5.0.4 |
| 추론 모델 | 로컬 Qwen3-1.7B INT4 AWQ |

---

## 7. 필수 요구 사항

### 제어 컴퓨터/온디바이스 컴퓨터

- Ubuntu 22.04 64-bit
- ROS 2 Humble
- Python 3.10 이상
- PyTorch, Transformers, Accelerate, PyYAML
- PX4 호환 `px4_msgs`
- 모바일 입력 사용 시 FastAPI, Uvicorn

### 시뮬레이션 컴퓨터

- PX4 SITL
- Gazebo
- MicroXRCE-DDS Agent
- ROS 2 Humble
- 제어 컴퓨터와 동일한 `ROS_DOMAIN_ID` 유지 필요

---

## 8. 설치 및 빌드

### 1) Python 의존성 설치

```bash
python3 -m pip install --user \
torch==2.14.0+cpu \
--index-url https://download.pytorch.org/whl/cpu

python3 -m pip install --user \
transformers==5.17.0 \
accelerate==1.15.0 \
peft==0.20.0 \
gptqmodel==7.5.0 \
torchao==0.18.0 \
triton==3.8.0 \
numpy==2.2.6 \
scipy==1.15.3
pyyaml \
fastapi \
uvicorn
```

> 기준 ㅡ cpu pc 환경


### 2) ROS 2 workspace 빌드

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

> `px4_msgs`와 `guide_interfaces`가 workspace에 포함되어 있는지 확인해야 합니다. PX4와 메시지 버전이 일치하지 않으면 빌드 오류가 발생할 수 있습니다.

### 3) 모델 경로 확인

실제 모델 경로는 `ros2_ws/src/llm_drone_control/config/model.yaml`에서 설정합니다.

```yaml
model:
  path: "~/Drone-VLA-VLM-Control-Project-Arion-/models/qwen3-1.7B-drone-int4-awq"
  max_new_tokens: 64
  do_sample: false
  torch_dtype: "auto"
  device_map: "auto"
```

필수 배포 경로 예시:

```text
ros2_ws/src/llm_drone_control/       # ROS 2 패키지
models/qwen3-1.7B-drone-int4-awq/    # 로컬 추론 모델
```

---

## 9. 실행 가이드

### 9.1 시뮬레이션 PC 실행

```bash
# MicroXRCE-DDS Agent 실행
MicroXRCEAgent udp4 -p 8888

# PX4 SITL + Gazebo 실행
cd ~/PX4-Autopilot
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=gz_x500_mono_cam \
PX4_GZ_MODEL_POSE="0,0,0.1,0,0,1.57" \
PX4_GZ_WORLD=default \
build/px4_sitl_default/bin/px4
```

### 9.2 ROS 2 런치 실행

현재 저장소에서 실제로 사용되는 런치 파일은 `drone_control.launch.py` 입니다.

```bash
cd ros2_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch llm_drone_control drone_control.launch.py
```

이 launch 파일은 다음 노드를 함께 실행합니다.

- `llm_service`
- `flight_controller`
- `smartphone_bridge`

### 9.3 브리지만 단독 실행

```bash
cd ros2_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run llm_drone_control smartphone_bridge
```

> 현재 구조상 핵심 실행 경로는 `drone_control.launch.py`이며, 별도 `sim.launch.py`는 이 프로젝트 코드베이스에 실제 구현되어 있지 않습니다.

### 9.4 명령 요청 및 상태 확인

```bash
# 서비스 직접 호출
ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '2m 이륙해줘'}"

# 단일 토픽 확인
ros2 topic echo /llm_response
ros2 topic echo /fmu/out/vehicle_local_position
```

---

## 10. 명령 처리 방식

1. 사용자가 `/llm` 서비스 또는 `/voice_command` 토픽으로 명령을 전달합니다.
2. `llm_service.py`가 로컬 Qwen3 INT4 AWQ 모델을 사용해 응답을 생성합니다.
3. 응답 문자열에서 `<tool_call>...</tool_call>` 형태의 JSON을 추출합니다.
4. `flight_controller.py`가 Tool 이름과 인자를 검증합니다.
5. PX4의 `OffboardControlMode`, `TrajectorySetpoint`, `VehicleCommand`를 발행합니다.

### 지원 Tool

| Tool | 동작 |
|---|---|
| `takeoff` | 현재 위치 기준으로 지정 고도까지 이륙 |
| `move` | 상대 이동, 상대 고도 변화, 회전 수행 |
| `land` | PX4 착륙 명령 실행 |
| `goto_history` | 직전 위치 또는 최초 위치로 복귀 |
| `reverse_plan` | 이동 경로를 역순으로 되짚어 복귀 |

---

## 11. 좌표계와 안전 주의사항

PX4는 NED 좌표계를 사용합니다.

- X: 북쪽 양수
- Y: 동쪽 양수
- Z: 아래쪽 양수
- 고도 2m는 일반적으로 `z = -2` 방향으로 표현됩니다.

### 안전 고려 사항

- 본 프로젝트는 시뮬레이션 검증용입니다.
- 실기체 적용 전에는 geofence, failsafe, 비상 정지, 배터리 감시 등을 별도로 추가해야 합니다.
- LLM의 응답을 그대로 비행 명령으로 사용해서는 안 됩니다.
- 실제 운용 환경에서는 입력 검증과 최종 승인 절차가 필요합니다.

---

## 12. 모델 학습 및 평가

Qwen3-1.7B 기반 드론 제어 언어 모델을 학습하고 평가하는 절차를 정리합니다.

### 목표

한국어 자연어 드론 명령을 PX4 제어용 Tool Call로 변환하는 모델을 훈련합니다.

예시:

```text
앞으로 3미터 이동해
→ move(dx=3, dy=0, dz=0, d_yaw=0)
```

### 12.1 데이터셋 구성

| 구분 | 수량 |
|---|---:|
| Train | 2,050개 |
| Validation | 410개 |
| Test | 610개 |
| 합계 | 3,070개 |

데이터 다양화 전략:

- 기본 명령 + 복합 명령 + 미지원 명령(Negative) 구성
- 숫자 표현 변화: `3미터`, `세 미터`, `3m`
- 띄어쓰기 변형, 오타, 구어체 반영
- OOD 문장을 포함해 일반화 능력 강화

### 12.2 Fine-tuning 설정

- **Base Model:** Qwen3-1.7B
- **방식:** LoRA Fine-tuning

| 항목 | 값 |
|---|---|
| Rank (`r`) | 16 |
| Alpha (`alpha`) | 32 |
| Dropout | 0.05 |
| Target Modules | `q / k / v / o / gate / up / down` |

주요 학습 전략:

- **Response-only Loss:** 시스템 프롬프트와 사용자 입력은 loss 계산에서 제외
- **Early Stopping:** validation loss가 개선되지 않으면 조기 종료

### 12.3 모델 변환 파이프라인

```text
Qwen3-1.7B (Base Model)
        │
        ▼  LoRA Fine-tuning
Base + LoRA Adapter
        │
        ▼  Merge
Merged Model
        │
        ▼  INT4 AWQ Quantization
INT4 AWQ Quantized Model  ← 현재 런타임 사용 모델
```

| 단계 | 저장 경로 | 설명 |
|---|---|---|
| Base | `models/Qwen3-1.7B/` | 원본 기반 모델 |
| LoRA Adapter | `models/finetuned_qwen3-1.7B_drone_lora/` | 파인튜닝 산출물 |
| Merged | `models/qwen3-1.7B-drone-merged/` | LoRA 병합 완료 모델 |
| INT4 AWQ | `models/qwen3-1.7B-drone-int4-awq/` | 현재 실행 모델 |

### 12.4 평가 항목

| 항목 | 설명 |
|---|---|
| Parse | `<tool_call>...</tool_call>` 블록을 정상적으로 파싱하는지 |
| Tool Name | 함수 이름이 정답과 일치하는지 |
| Argument | 인자 값과 이름이 모두 일치하는지 |
| Overall | Tool Name과 Argument가 모두 맞는 완전 정답 비율 |
| Refusal | 지원하지 않는 명령을 안전하게 거절하는 비율 |

### 12.5 성능 비교

| 모델 | Parse | Tool Name | Argument | Overall | Refusal |
|---|---:|---:|---:|---:|---:|
| BASE | 92.73% | 70.18% | 32.00% | 36.56% | 80.00% |
| Merged | 99.64% | 94.73% | 88.55% | 89.67% | 100.00% |
| INT4 AWQ | 99.27% | 89.64% | 78.36% | 80.49% | 100.00% |

분석:

- BASE → Merged: LoRA 파인튜닝으로 인자 정확도가 크게 향상됨
- Merged → INT4 AWQ: 양자화 후 약간의 성능 저하가 있지만 안전성 유지
- 온디바이스 채택 이유: 메모리와 연산 부담을 크게 줄이면서도 충분한 성능을 확보

---

## 13. 참고 문서

### 공식 문서

- [PX4 ROS 2 Offboard Control Guide](https://docs.px4.io/main/en/ros2/offboard_control.html)
- [PX4-Autopilot](https://github.com/PX4/PX4-Autopilot)
- [px4_msgs](https://github.com/PX4/px4_msgs)
- [ROS 2 Humble Documentation](https://docs.ros.org/en/humble/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [QGroundControl v5.0.4](https://github.com/mavlink/qgroundcontrol/releases/tag/v5.0.4)

### 프로젝트 내부 문서

- [비행 제어 테스트 명령어](docs/drone_control_test_commands.md)
- [LLM 드론 제어 코드 상호작용 보고서](docs/llm_drone_control_code_interaction_report.md)
- [PX4 ROS 2 메시지 상세](docs/px4_msgs_details.md)
- [모델 성능 비교표](models/performance_comparison.md)
- [개발 일지](CHANGELOG.md)

---

## 14. 향후 계획

- [x] 스마트폰 웹 브리지 음성·텍스트 입력 연동
- [x] 온디바이스 추론 최적화 및 INT4 AWQ 양자화
- [ ] 카메라 영상과 VLM 기반 환경 인식
- [ ] geofence, failsafe, 비상 정지 등 안전 기능 강화
- [ ] 실기체용 Pixhawk 및 companion computer 연동
- [ ] 비행 로그 저장과 시각화 대시보드

---

## 15. Git Commit Convention

| Prefix | Description |
|---|---|
| **[feature]** | 새로운 기능 추가 또는 알고리즘 수정 |
| **[bugfix]** | 버그 수정 |
| **[ignore]** | 주석 추가, 코드 정리 등 알고리즘에 영향을 주지 않는 변경 |
| **[test]** | 주 코드에 영향을 주지 않는 테스트용 참고 파일 |

개발 과정의 상세 일지는 [CHANGELOG.md](CHANGELOG.md)를 참고합니다.

---

본 프로젝트는 학습 및 연구 목적으로 작성되었으며, 실기체 사용 전에는 충분한 시뮬레이션과 안전 검증이 필요합니다.
