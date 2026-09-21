# Drone-VLA-VLM-Control-Project-Arion

## 🚁 프로젝트 소개

자연어 명령을 이해하는 소형 언어 모델과 PX4 기반 비행 제어를 결합한 드론 자율 제어 프로젝트입니다. 사용자의 명령을 Qwen3 모델이 구조화된 Tool Call로 변환하고, ROS 2 노드가 이를 PX4 Offboard 제어 명령으로 실행합니다.

현재 프로젝트는 **온디바이스 추론을 목표로 하는 시뮬레이션 검증 단계**입니다. 학습과 추론에 필요한 Qwen3 모델 및 LoRA 어댑터는 외부 API가 아니라 로컬 파일에서 불러오며, PX4 SITL과 Gazebo는 별도의 시뮬레이션 PC에서 실행합니다.

## 📝 프로젝트 정보

**조장:** 신현수
**VLA 파인튜닝:** 정진한, 최지호
**함수 스키마 담당:** 김태형, 정원혁
**지도강사:** 박승휘

## 📚 문서 목차

- [프로젝트 소개](#-프로젝트-소개)
- [프로젝트 정보](#-프로젝트-정보)
- [주요 기능](#-주요-기능)
- [시스템 구성](#-시스템-구성)
- [온디바이스 대상](#-온디바이스-대상)
- [환경 요구 사항](#-환경-요구-사항)
- [설치 및 빌드](#-설치-및-빌드)
- [실행 가이드](#-실행-가이드)
- [명령 처리 방식](#-명령-처리-방식)
- [좌표계와 안전 주의사항](#-좌표계와-안전-주의사항)
- [학습 및 평가](#-학습-및-평가)
- [참고 저장소 및 사용 도구](#-참고-저장소-및-사용-도구)
- [향후 계획](#-향후-계획)
- [프로젝트 개발 일지](#-프로젝트-개발-일지)



## 📌 Git Commit Convention

| Prefix | Description |
|---|---|
| **[feature]** | 새로운 기능 추가 또는 알고리즘 수정 |
| **[bugfix]** | 버그 수정 |
| **[ignore]** | 주석 추가, 코드 정리 등 알고리즘에 영향을 주지 않는 변경 |
| **[test]** | 주 코드에 영향을 주지 않는 테스트용 참고 파일 |

## 📅 프로젝트 개발 일지

### [2026.09.14] 작업 환경 구성 및 기초 교육

- Ubuntu 기반 개발 환경 구축
- PX4 Autopilot, QGroundControl 등 필수 도구 설치
- 드론 제어 및 비행 기초 이론 교육

### [2026.09.15] 모델 선정 및 시스템 분석

- `Qwen3-0.6B` 텍스트 모델과 OpenCV 비전 기술 적용 방향 검토
- PX4 기반 제어 시스템 아키텍처 분석
- 자연어 명령을 Tool Call로 변환하기 위한 함수 스키마 정의

### [2026.09.16] 스키마 구조 확정 및 파인튜닝

- 확정된 스키마 기반 파인튜닝 데이터셋 제작
- 소형 모델의 처리 부담을 줄이기 위한 JSON 기반 스키마로 전환
- ROS 2 제어 코드의 모듈화 및 Qwen 모델 파인튜닝 시작

### [2026.09.17] 데이터셋 보강

- 파인튜닝 결과를 평가하고 train, validation, test 데이터셋 형식 분리
- 모델의 변경을 고려 (`Qwen3-0.6B`-> `Qwen3-1.7B` )

```text
사용자 자연어 명령
        |
        v
ROS 2 /llm 서비스
        |
        v
Qwen3-1.7B + LoRA 로컬 추론
        |
        v
<tool_call> JSON 응답
        |
        v
flight_controller
        |
        v
PX4 ROS 2 토픽 -> PX4 SITL -> Gazebo
```

### [2026.09.18] 
- STT node 추가 및 의존성 추가
- llm_drone_control패키지 통합런치 
- 함수 스키마 수정

## ✨ 주요 기능

- Qwen3-1.7B 기반 로컬 자연어 명령 해석
- LoRA 파인튜닝 모델을 이용한 드론 명령 스키마 응답
- `takeoff`, `land` Tool Call 기반 PX4 Offboard 제어
- ROS 2 서비스 `/llm` 및 응답 토픽 `/llm_response` 사용
- `stt_node`를 통한 마이크 음성 입력 및 Whisper 기반 STT
- LLM 요청과 응답을 `llm_logs/` 디렉터리에 파일로 저장
- PX4 NED 좌표계 기반 고도 제어
- PX4 상태, 위치, 착륙 상태를 이용한 비행 상태 관리
- 학습, 검증, 추론 코드를 분리한 프로젝트 구조

## 🏗️ 시스템 구성

| 구분 | 실행 위치 | 주요 구성 |
|---|---|---|
| 온디바이스 제어부 | 온보드 컴퓨터 또는 개발 PC | `ros2_ws/src/llm_drone_control`, Qwen3 모델, LoRA 어댑터 |
| 비행 시뮬레이터 | 로컬 시뮬레이션 PC | PX4 SITL, Gazebo, MicroXRCE-DDS Agent |
| 지상 관제 | 선택 사항 | QGroundControl v5.0.4 |
| 모델 학습 | 학습용 PC 또는 GPU 서버 | `AI_data/`, `train/`, `models/` |

## 📁 온디바이스 대상

온디바이스로 옮길 대상은 ROS 2 제어 패키지와 추론 모델입니다.

### 온디바이스에 포함되는 폴더와 파일

```text
ros2_ws/src/llm_drone_control/
├── llm_drone_control/
│   ├── llm_service.py       # 로컬 Qwen3 + LoRA 추론 및 메시지 로그 저장
│   ├── stt_node.py          # 마이크 음성 입력 및 LLM 요청 노드
│   ├── flight_controller.py # PX4 Offboard 제어 노드
│   └── schema.py            # Tool Call 스키마 및 시스템 프롬프트
├── config/model.yaml        # 모델 및 LoRA 경로 설정
├── launch/drone_control.launch.py
├── package.xml
└── setup.py

models/
├── Qwen3-1.7B/                    # 기본 로컬 모델
└── finetuned_qwen3-1.7B_drone_lora/    # 드론 명령 LoRA 어댑터
```

`model.yaml`의 현재 기본 설정은 다음 경로를 사용합니다.

```yaml
model:
  path: "~/Drone-VLA-VLM-Control-Project-Arion-/models/Qwen3-1.7B"
  lora_path: "~/Drone-VLA-VLM-Control-Project-Arion-/models/finetuned_qwen3-1.7B_drone_lora"
```

Qwen3-1.7B 모델과 해당 LoRA 어댑터를 사용할 경우 `model.yaml`의 `path`와 `lora_path`를 실제 모델 디렉터리에 맞게 변경합니다.

따라서 실제 온디바이스 배포 시에는 다음 항목이 필요합니다.

- Ubuntu 및 ROS 2 Humble 실행 환경
- `llm_drone_control` ROS 2 패키지
- `Qwen3-0.6B` 모델 파일
- `finetuned_qwen3_drone_lora` LoRA 파일
- PyTorch, Transformers, PEFT, Accelerate, PyYAML
- `SpeechRecognition`, `openai-whisper`, `PyAudio` 음성 입력 의존성
- PX4와 통신할 수 있는 ROS 2 DDS 네트워크

### 온디바이스에 포함되지 않는 폴더와 구성

- `AI_data/`, `train/`: 데이터셋 생성, 학습, 평가용 코드
- `test/`: 개발 및 시뮬레이션 검증용 테스트
- `docs/`: 설계 및 코드 상호작용 문서
- `build/`, `install/`, `log/`: ROS 2 빌드 산출물
- PX4 SITL, Gazebo, MicroXRCE-DDS Agent: 시뮬레이션 PC에서 실행
- QGroundControl: 선택적인 지상 관제 프로그램

현재 온디바이스 제어 노드는 로컬 모델 추론과 ROS 2/PX4 제어를 담당합니다. 카메라 영상 기반 VLM 추론과 실기체 안전 기능은 향후 추가 대상입니다.

## 📦 환경 요구 사항

### 온디바이스 또는 제어 컴퓨터

- Ubuntu 22.04 64-bit 권장
- ROS 2 Humble
- Python 3
- PX4와 호환되는 `px4_msgs`
- PyTorch 2.1.2, Transformers 4.51.3, Accelerate, PEFT, PyYAML

### 시뮬레이션 컴퓨터

- PX4 SITL
- Gazebo
- MicroXRCE-DDS Agent
- ROS 2 Humble 및 동일한 `ROS_DOMAIN_ID`

## 🛠️ 설치 및 빌드

### 1. Python 의존성 설치

```bash
python3 -m pip install --user \
  "numpy==1.24.4" \
  "scipy==1.10.1" \
  "torch==2.1.2" \
  "transformers==4.51.3" \
  "accelerate>=0.34.2" \
  "peft" \
        "pyyaml" \
        "SpeechRecognition" \
        "openai-whisper" \
        "PyAudio"
```

### 2. ROS 2 workspace 빌드

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

`px4_msgs`와 `guide_interfaces`가 workspace에 포함되어 있는지 확인합니다. PX4와 `px4_msgs`의 메시지 버전은 반드시 일치해야 합니다.

## 🚀 실행 가이드

### 시뮬레이션 PC

MicroXRCE-DDS Agent를 실행합니다.

```bash
MicroXRCEAgent udp4 -p 8888
```

그 다음 PX4 SITL과 Gazebo를 실행합니다. PX4 프로젝트의 빌드 방식과 사용하는 기체 모델에 따라 명령은 달라질 수 있습니다.

```bash
cd ~/PX4-Autopilot
PX4_SYS_AUTOSTART=4001 \
PX4_SIM_MODEL=gz_x500 \
build/px4_sitl_default/bin/px4
```

QGroundControl은 선택적으로 실행합니다.

```bash
./QGroundControl-x86_64.AppImage
```

### 온디바이스 제어 컴퓨터

ROS 2 workspace를 소싱한 뒤 launch 파일을 실행합니다.

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch llm_drone_control drone_control.launch.py
```

현재 통합 launch 파일은 다음 세 노드를 실행합니다.

- `llm_service`: 로컬 Qwen3 + LoRA 모델을 로드하고 `/llm` 서비스를 제공합니다.
- `flight_controller`: `/llm_response`를 받아 PX4 Offboard 제어 토픽을 발행합니다.
- `stt_node`: 엔터 입력 후 마이크 음성을 인식해 `/llm` 서비스로 전달합니다.

`stt_node` 실행 방법:

1. 통합 launch 실행 후 STT 노드 터미널에서 엔터를 누릅니다.
2. 안내 메시지가 표시되면 한국어 음성 명령을 말합니다.
3. 음성 인식 결과가 `/llm` 서비스로 전달되고, LLM 응답이 `/llm_response`로 발행됩니다.

LLM 서비스는 실행 위치를 기준으로 `llm_logs/` 디렉터리를 생성합니다. 실행할 때마다 `llm_log_YYYYMMDD_HHMMSS.log` 파일을 만들고, 각 요청의 `[질문]`과 `[답변]`을 추가 저장합니다.

### 명령 요청

```bash
ros2 service call /llm guide_interfaces/srv/GuideLLM \
  "{prompt: '2m 이륙해줘'}"
```

PX4 토픽과 통신 상태를 확인합니다.

```bash
ros2 topic list | grep fmu
ros2 topic echo /fmu/out/vehicle_local_position
ros2 topic echo /llm_response
```

## 🧠 명령 처리 방식

1. 사용자가 `/llm` 서비스로 자연어 명령을 전달합니다.
2. `llm_service.py`가 로컬 Qwen3 모델과 LoRA 어댑터를 사용해 응답을 생성합니다.
3. 응답에서 `<tool_call>...</tool_call>` 형식의 JSON을 추출합니다.
4. `flight_controller.py`가 Tool 이름과 인자를 확인합니다.
5. 비행 제어 노드가 PX4의 `OffboardControlMode`, `TrajectorySetpoint`, `VehicleCommand`를 발행합니다.

현재 기본 제어 노드에서 직접 처리하는 주요 명령은 다음과 같습니다.

| Tool | 동작 |
|---|---|
| `takeoff` | 현재 위치를 기준으로 지정 고도까지 이륙 |
| `land` | PX4 착륙 명령 실행 |
| `move` | 상대 이동, 상대 고도 변화 및 기수 회전 |
| `goto_history` | 과거 위치로 복귀 (직전 위치,최초 출발지) |
| `reverse_plan` | 경로를 역순으로 되짚어 복귀 |


## 🧭 좌표계와 안전 주의사항

PX4는 NED 좌표계를 사용합니다.

- X: 북쪽 양수
- Y: 동쪽 양수
- Z: 아래쪽 양수
- 고도 2m: 기준 지점에서 일반적으로 `z = -2` 방향

본 프로젝트는 시뮬레이션 검증용입니다. 실기체에 적용하기 전에는 최대 고도, 이동 거리, geofence, failsafe, 통신 끊김, 비상 정지, 배터리 상태 검사를 별도로 추가해야 합니다. LLM 응답을 그대로 비행 명령으로 사용하지 말고, 실제 운용 환경에서는 입력 검증과 승인 단계를 두어야 합니다.

## 🧪 학습 및 평가

학습과 평가에 사용하는 주요 경로는 다음과 같습니다.

```text
AI_data/
├── generate_dataset.py
├── train_LoRA.py
├── train_lora_with_validation.py
└── evaluate_lora.py

train/
├── train.jsonl
├── validate_dataset.py
└── test_train_lora_validation.py
```

학습 결과인 LoRA 어댑터는 `models/finetuned_qwen3_drone_lora/`에 저장하고, `ros2_ws/src/llm_drone_control/config/model.yaml`의 `lora_path`에서 해당 위치를 지정합니다.

## 🔗 참고 저장소 및 사용 도구

### 이전 기수 참고 저장소

- [Criss-J/ws_ros2](https://github.com/Criss-J/ws_ros2.git)

### PX4 및 Gazebo 시뮬레이터 참고 저장소

- [PX4-ROS2-Gazebo-Drone-Simulation-Template](https://github.com/SathanBERNARD/PX4-ROS2-Gazebo-Drone-Simulation-Template.git)

### QGroundControl

- [QGroundControl v5.0.4 release](https://github.com/mavlink/qgroundcontrol/releases/tag/v5.0.4)

### 공식 참고 문서

- [PX4 ROS 2 Offboard Control Guide](https://docs.px4.io/main/en/ros2/offboard_control.html)
- [PX4-Autopilot](https://github.com/PX4/PX4-Autopilot)
- [px4_msgs](https://github.com/PX4/px4_msgs)
- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)

## 🔮 향후 계획

- [ ] 카메라 영상과 VLM을 이용한 환경 인식
- [x] 음성 입력 및 Whisper STT 연동
- [ ] 온디바이스 추론 최적화 및 양자화
- [ ] geofence, failsafe, 비상 정지 등 안전 기능 강화
- [ ] 실기체용 Pixhawk 및 companion computer 연동
- [ ] 비행 로그 저장과 시각화 대시보드


본 프로젝트는 학습 및 연구 목적으로 작성되었습니다. 실기체 사용 전 충분한 시뮬레이션과 안전 검증이 필요합니다.