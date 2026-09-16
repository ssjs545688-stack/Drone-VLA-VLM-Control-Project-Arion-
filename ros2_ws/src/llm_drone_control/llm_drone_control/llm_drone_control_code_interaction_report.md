# LLM Drone Control 코드 역할 및 상호작용 보고서

## 1. 보고서 개요

### 1.1 목적

본 보고서는 `llm_drone_control` 패키지를 구성하는 드론 제어 코드의 역할, 실행 방식, 모듈 사이의 상호작용을 정리한 문서다. 분석 대상은 다음 일곱 파일이다.

| 구분 | 파일 | 핵심 역할 |
|---|---|---|
| 통합 노드 | `drone_unified_node.py` | ROS 2 노드 생성 및 전체 모듈 조립 |
| 미션 실행 | `mission_executor.py` | Tool Call 큐와 비행 상태 머신 관리 |
| PX4 통신 | `px4_interface.py` | PX4 ROS 2 입력 토픽 메시지 발행 |
| LLM 엔진 | `qwen_engine.py` | Qwen 모델 추론, Tool Call 파싱 및 검증 |
| 공통 유틸리티 | `utils.py` | 각도, 숫자, JSON 처리 순수 함수 제공 |
| Tool 정의 | `schema.py` | 사용 가능한 Tool과 LLM 시스템 프롬프트 정의 |
| 독립 예제 | `offboard_ctrl_example.py` | LLM 없이 동작하는 PX4 Offboard 기본 비행 예제 |

### 1.2 전체 구조 요약

통합 실행 경로는 다음과 같다.

```text
사용자 자연어
    |
    v
DroneUnifiedNode.process_command()
    |
    v
QwenToolCaller
    |  프롬프트 생성 -> 모델 추론 -> 출력 파싱 -> 안전성 검증
    v
검증된 Tool Call 목록
    |
    v
MissionExecutor.enqueue_tool_calls()
    |
    v
MissionExecutor.start_next_mission()
    |
    v
PX4Interface
    |  VehicleCommand / TrajectorySetpoint / OffboardControlMode
    v
PX4 ROS 2 입력 토픽
    |
    v
PX4 비행 컨트롤러
    |
    v
VehicleLocalPosition / VehicleStatus / VehicleLandDetected
    |
    v
DroneUnifiedNode 콜백 -> MissionExecutor.tick()
```

핵심 설계는 LLM이 직접 좌표나 PX4 메시지를 만들지 않는다는 점이다. LLM은 사용자의 의도를 `takeoff`, `move`, `land`, `goto_history`, `reverse_plan` 중 하나 이상의 Tool Call로 변환한다. 실제 좌표 계산, 안전 한계 검증, 미션 순서 관리, PX4 메시지 발행은 Python 제어 코드가 담당한다.

---

## 2. 파일별 역할

## 2.1 `schema.py`: Tool 계약과 LLM 지침

### 주요 구성

- `DEFINE_SCHEMA`
- `TOOL_NAMES`
- `SCHEMA_TEXT`
- `SYSTEM_PROMPT`

### 담당 기능

`schema.py`는 LLM이 생성할 수 있는 명령의 계약을 정의한다. ROS 2나 PyTorch에 의존하지 않는 가벼운 모듈이므로 학습 데이터 생성, Tool Call 테스트, 검증 스크립트에서도 재사용할 수 있다.

| Tool | 기능 | 주요 인자 |
|---|---|---|
| `takeoff` | 현재 위치에서 지정 고도로 수직 이륙 | `altitude` |
| `move` | 기체 방향 기준 상대 이동 및 회전 | `dx`, `dy`, `dz`, `d_yaw` |
| `land` | 현재 위치에서 착륙 | 없음 |
| `goto_history` | 과거 위치로 이동 | `recall`: `previous` 또는 `first` |
| `reverse_plan` | 완료된 이동 경로를 역순으로 복귀 | 없음 |

### 동작 방식

1. `DEFINE_SCHEMA`에 Tool 이름, 설명, 인자 타입, 필수 인자를 정의한다.
2. `SCHEMA_TEXT`가 스키마를 JSON 문자열로 변환한다.
3. `SYSTEM_PROMPT`가 스키마와 출력 규칙을 Qwen에게 전달한다.
4. Qwen은 다음 형태의 문자열을 출력하도록 지시받는다.

```text
<tool_call>{"name":"move","arguments":{"dx":1,"dy":0,"dz":0,"d_yaw":0}}</tool_call>
```

### 상호작용

- `qwen_engine.py`가 `SYSTEM_PROMPT`를 사용해 프롬프트를 생성한다.
- `qwen_engine.py`가 `TOOL_NAMES`를 이용해 Tool 이름을 검증한다.
- `mission_executor.py`는 이 스키마와 동일한 Tool 이름을 분기 기준으로 사용한다.

즉, `schema.py`는 LLM과 실행기의 공통 인터페이스 역할을 한다.

---

## 2.2 `utils.py`: 공통 순수 함수

### 주요 함수

| 함수 | 역할 |
|---|---|
| `clamp_angle()` | 라디안 각도를 `[-pi, pi]` 범위로 정규화 |
| `is_finite_number()` | 값이 유한한 숫자인지 확인 |
| `safe_float()` | 변환 실패 또는 비정상 숫자를 기본값으로 변환 |
| `normalize_tool_call()` | 다양한 Tool JSON 형식을 내부 표준 형식으로 변환 |
| `extract_balanced_json()` | 문자열에서 중첩 구조를 고려해 완전한 JSON object 추출 |

### 동작 방식

`utils.py`는 상태를 보관하지 않는 순수 함수 모듈이다. 외부 ROS 2 메시지나 모델을 직접 다루지 않기 때문에 단위 테스트가 쉽다.

특히 `normalize_tool_call()`은 다음과 같은 입력 차이를 통일한다.

```text
function.name / function.arguments 형식
name / arguments 형식
action / arguments 형식
arguments가 JSON 문자열인 형식
```

모든 입력은 내부적으로 다음 구조가 된다.

```python
{"name": "move", "arguments": {...}}
```

### 상호작용

- `qwen_engine.py`가 Tool Call 파싱 과정에서 `normalize_tool_call()`과 `extract_balanced_json()`을 사용한다.
- `qwen_engine.py`가 숫자 검증 과정에서 `is_finite_number()`을 사용한다.
- `drone_unified_node.py`가 PX4 pose를 읽을 때 `safe_float()`을 사용한다.
- `mission_executor.py`가 각도와 역방향 이동을 처리할 때 `clamp_angle()`과 `safe_float()`을 사용한다.
- `px4_interface.py`가 yaw setpoint를 발행할 때 `clamp_angle()`을 사용한다.

---

## 2.3 `qwen_engine.py`: 자연어를 Tool Call로 변환

### 주요 클래스

- `ValidationLimits`
- `QwenToolCaller`

### 담당 기능

`qwen_engine.py`는 로컬 Qwen3 모델과 통신하지만 PX4나 ROS 2에는 직접 의존하지 않는다. 현재 드론 상태 문자열과 사용자의 자연어를 입력받아 검증된 Tool Call 목록을 반환한다.

### 실행 단계

#### 1단계: 모델 로딩

`QwenToolCaller.__init__()`에서 `_load_qwen()`을 호출한다.

- 모델 디렉터리 존재 여부를 확인한다.
- `config.json`과 tokenizer 파일을 확인한다.
- 로컬 파일만 사용해 tokenizer와 model을 로딩한다.
- CUDA 사용 가능 여부에 따라 `float16` 또는 `float32`를 선택한다.
- 모델을 평가 모드로 설정한다.

#### 2단계: 프롬프트 생성

`build_prompt()`는 다음 내용을 하나의 사용자 메시지로 결합한다.

```text
현재 드론 상태
사용자 명령
```

그 뒤 `schema.py`의 `SYSTEM_PROMPT`와 함께 chat template을 적용한다.

#### 3단계: 모델 추론

`invoke()`는 tokenizer로 입력을 토큰화한 뒤 모델의 `generate()`를 호출한다. 입력 prompt에 해당하는 토큰은 제거하고 새로 생성된 부분만 문자열로 반환한다.

#### 4단계: Tool Call 파싱

`parse_tool_calls()`는 다음 순서로 모델 출력을 해석한다.

1. `<think>...</think>` 블록 제거
2. `<tool_call>...</tool_call>` 내부 JSON 검색
3. JSON 파싱 실패 시 균형이 맞는 JSON object 추출
4. Tool Call이 없으면 plain JSON object 또는 JSON array 시도
5. `normalize_tool_call()`로 내부 표준 구조 변환

#### 5단계: 안전성 검증

`validate_tool_call()`은 다음 항목을 검증한다.

- Tool 이름이 허용 목록에 포함되는지
- `arguments`가 object인지
- 필수 인자가 존재하는지
- 모든 숫자가 유한한 값인지
- 이륙 고도 제한
- 수평 이동 거리 제한
- 수직 이동량 제한
- yaw 변화량 제한
- `goto_history.recall` 값이 `first` 또는 `previous`인지

`validate_all()`은 여러 호출을 순서대로 검사하며 첫 번째 오류에서 전체 목록을 거부한다.

### 상호작용

- 입력: `drone_unified_node.py`의 `_build_state_text()`와 사용자 자연어
- 설정: `schema.py`의 `SYSTEM_PROMPT`, `TOOL_NAMES`
- 파싱 지원: `utils.py`
- 출력: `drone_unified_node.py`의 `mission.enqueue_tool_calls()`로 전달되는 검증된 Tool Call 목록

---

## 2.4 `mission_executor.py`: 미션 상태 머신

### 주요 클래스

- `ArrivalThresholds`
- `MissionExecutor`

### 담당 기능

`MissionExecutor`는 Tool Call을 실제 실행 순서로 바꾸는 핵심 상태 머신이다. ROS 2 Node를 상속하지 않고, PX4 인터페이스와 현재 pose 조회 함수를 주입받는 구조다.

### 내부 상태

| 상태/자료 | 의미 |
|---|---|
| `mission_queue` | 아직 실행하지 않은 Tool Call 큐 |
| `active_tool` | 현재 실행 중인 Tool |
| `mission_state` | `STANDBY`, `EXECUTING`, `LANDING`, `HOVER` 등 상태 |
| `target_x/y/z/yaw` | 현재 목표 pose |
| `has_target` | setpoint 발행 대상 존재 여부 |
| `in_landing` | PX4 착륙 신호를 기다리는 중인지 여부 |
| `position_history` | 작업 시작 직전의 실제 pose 기록 |
| `forward_move_history` | 완료된 원래 `move` 명령 기록 |
| `executing_reverse` | 역경로 실행 여부 |
| `arrival_count` | 목표 도착 조건 연속 충족 횟수 |

### Tool 실행 방식

#### `takeoff`

`_start_takeoff()`가 현재 x/y 위치를 유지하면서 목표 z를 `-altitude`로 계산한다. PX4 NED 좌표계에서 위쪽 고도는 음수이므로 고도에 음수 부호를 적용한다. 이후 OFFBOARD 전환과 ARM 명령을 요청한다.

#### `move`

`_start_move()`는 사용자가 지정한 body-frame 이동량을 현재 yaw 기준 world-frame 이동량으로 변환한다.

```text
delta_x = dx*cos(yaw) - dy*sin(yaw)
delta_y = dx*sin(yaw) + dy*cos(yaw)
```

고도 변화는 PX4 NED 규칙에 맞춰 `target_z = current_z - dz`로 계산한다. yaw 변화는 degree에서 radian으로 변환한 뒤 정규화한다.

#### `land`

`_start_land()`는 목표 좌표 도착 방식이 아니라 PX4의 착륙 감지 상태를 기다리는 `LANDING` 상태로 전환한다.

#### `goto_history`

`_start_goto_history()`는 `resolve_history()`로 과거 pose를 찾고 해당 pose를 새 목표로 설정한다. history 이동은 일반 forward move 경로가 아니므로 `forward_move_history`를 초기화한다.

#### `reverse_plan`

`_start_reverse_plan()`은 완료된 `move` 목록을 `build_reverse_plan()`으로 역순 변환한다.

원래 이동이 다음과 같다면:

```text
translation(dx, dy, dz) + rotation(d_yaw)
```

역방향은 다음 순서로 생성된다.

```text
rotation(-d_yaw) -> translation(-dx, -dy, -dz)
```

역방향 이동 중 생성된 명령은 원래 forward history에 다시 기록하지 않는다.

### 타이머 동작

`tick()`은 `drone_unified_node.py`의 ROS timer에서 호출된다.

1. PX4 Offboard heartbeat 발행
2. 착륙 중이면 `is_landed` 확인
3. 일반 미션이면 목표 position/yaw setpoint 발행
4. 현재 pose와 목표 pose 사이의 거리 및 yaw 오차 계산
5. 오차가 임계값보다 작으면 `arrival_count` 증가
6. 일정 횟수 연속 도착하면 현재 Tool 완료
7. 큐에 다음 Tool이 있으면 즉시 실행

한 번이라도 오차가 임계값을 벗어나면 `arrival_count`를 0으로 초기화해 순간적인 위치 오판정을 줄인다.

### 상호작용

- 입력: `drone_unified_node.py`에서 전달하는 검증된 Tool Call
- 현재 pose: `drone_unified_node.py._get_local_pose()` 콜백
- PX4 명령: `px4_interface.py`
- 각도/숫자 처리: `utils.py`
- 결과 상태: `drone_unified_node.py`의 이미지 표시 및 다음 명령 입력 가능 여부에 반영

---

## 2.5 `px4_interface.py`: PX4 명령 발행 계층

### 주요 구성

- `VehicleIds`
- `default_pub_qos()`
- `default_sub_qos()`
- `PX4Interface`

### 담당 기능

`PX4Interface`는 ROS 2 Node 자체가 아니라 외부 Node를 주입받아 PX4 입력 publisher를 생성한다. 미션 로직이 PX4 메시지의 세부 필드를 직접 알지 않도록 통신 책임을 한 곳에 모은다.

### publisher

| ROS 2 토픽 | 메시지 | 용도 |
|---|---|---|
| `/fmu/in/offboard_control_mode` | `OffboardControlMode` | 위치 제어 heartbeat |
| `/fmu/in/trajectory_setpoint` | `TrajectorySetpoint` | 목표 위치와 yaw |
| `/fmu/in/vehicle_command` | `VehicleCommand` | ARM, OFFBOARD, LAND 명령 |

### 주요 메서드

- `arm()`: `VEHICLE_CMD_COMPONENT_ARM_DISARM`, `param1=1.0`
- `disarm()`: `VEHICLE_CMD_COMPONENT_ARM_DISARM`, `param1=0.0`
- `engage_offboard()`: `VEHICLE_CMD_DO_SET_MODE`, `param1=1.0`, `param2=6.0`
- `land()`: `VEHICLE_CMD_NAV_LAND`
- `publish_offboard_heartbeat()`: 위치 제어 모드 heartbeat 발행
- `publish_setpoint()`: NED 위치와 yaw 발행
- `publish_vehicle_command()`: VehicleCommand 공통 필드와 기체 ID 설정

`publish_vehicle_command()`는 target/source system 및 component 값을 채우고 `from_external=True`로 설정한다. 이를 통해 명령 대상과 외부 제어 여부를 PX4에 명확히 전달한다.

### 상호작용

- 호출자: `mission_executor.py`
- 의존 객체: `drone_unified_node.py`가 제공하는 ROS 2 Node
- 좌표 보정: `utils.py.clamp_angle()`
- 외부 결과: PX4가 발행하는 상태 토픽은 `drone_unified_node.py`가 수신한다.

---

## 2.6 `drone_unified_node.py`: 전체 모듈 조립 및 ROS 경계

### 담당 기능

이 파일은 실제 비행 제어 로직을 모두 직접 구현하기보다, 각 모듈을 ROS 2 환경에 연결하는 진입점이다.

### 초기화 과정

`DroneUnifiedNode.__init__()`는 다음 순서로 실행된다.

1. ROS 2 Node 생성
2. `ReentrantCallbackGroup`과 `RLock` 생성
3. ROS parameter 선언 및 읽기
4. PX4 위치, 상태, 착륙 감지 subscriber 생성
5. 카메라 이미지 subscriber 생성
6. `PX4Interface` 생성
7. `QwenToolCaller` 생성 및 로컬 모델 로딩
8. `MissionExecutor` 생성
9. 주기 timer 생성
10. 사용자 입력 스레드 시작

### 사용자 명령 흐름

`user_input_loop()`가 터미널 입력을 받는다.

- `exit`, `quit`: ROS 종료
- `멈춰`, `정지`, `stop`: 현재 미션 중단 및 현재 위치 hover
- 일반 명령: 위치 수신 여부와 실행 중인 미션 여부를 확인한 뒤 `process_command()` 호출

`process_command()`는 다음 역할을 담당한다.

1. 미션 실행 중인지 확인
2. 현재 pose와 history 정보를 상태 문자열로 생성
3. Qwen 호출
4. Tool Call 파싱 결과 확인
5. Tool Call 전체 검증
6. 검증된 목록을 `MissionExecutor`에 전달

### ROS callback과 timer

- `position_callback()`: 최신 `VehicleLocalPosition` 저장
- `status_callback()`: 최신 `VehicleStatus` 저장
- `land_detected_callback()`: 최신 `VehicleLandDetected` 저장
- `image_callback()`: 카메라 이미지를 OpenCV로 표시하고 큐/history 상태를 화면에 추가
- `timer_callback()`: 착륙 여부를 읽어 `mission.tick()` 호출

`MultiThreadedExecutor`와 사용자 입력 스레드가 동시에 상태에 접근하므로 주요 상태 접근은 `state_lock`으로 보호한다.

### 상호작용

`drone_unified_node.py`는 다음 모듈을 직접 조립한다.

```text
schema.py             <- qwen_engine.py가 사용
utils.py              <- pose, parsing, angle 처리
qwen_engine.py        <- 자연어 -> 검증된 Tool Call
mission_executor.py   <- Tool Call -> 목표/상태 변화
px4_interface.py      <- 목표/상태 -> PX4 메시지
```

---

## 2.7 `offboard_ctrl_example.py`: 독립 PX4 기본 예제

### 담당 기능

이 파일은 통합 LLM 제어 경로와 별개인 단일 ROS 2 예제다. Qwen, Tool schema, MissionExecutor를 사용하지 않는다.

### 실행 순서

`timer_callback()`이 0.1초 주기로 다음 작업을 한다.

1. `OffboardControlMode` heartbeat 발행
2. 설정된 초기 setpoint 횟수까지 카운터 증가
3. 충분한 setpoint가 발행되면 OFFBOARD 전환 및 ARM 요청
4. PX4가 OFFBOARD 상태이고 목표 고도에 도달하지 않았으면 위치 setpoint 발행
5. 목표 고도에 도달하면 LAND 명령 발행
6. 이미지 subscriber는 카메라 영상을 OpenCV 창에 표시

### 통합 구조와의 관계

이 파일은 `px4_interface.py`가 분리되기 전의 저수준 publisher 패턴을 보여주는 참고 구현이다. 두 코드 모두 같은 PX4 입력 토픽과 유사한 메시지 구성을 사용한다.

| 독립 예제 | 통합 구조 |
|---|---|
| `OffboardControl.arm()` | `PX4Interface.arm()` |
| `publish_offboard_control_heartbeat_signal()` | `PX4Interface.publish_offboard_heartbeat()` |
| `publish_position_setpoint()` | `PX4Interface.publish_setpoint()` |
| `publish_vehicle_command()` | `PX4Interface.publish_vehicle_command()` |
| `timer_callback()` 내부 상태 로직 | `MissionExecutor.tick()` |

따라서 독립 예제는 PX4 연결을 처음 확인할 때 사용할 수 있고, 통합 노드는 자연어와 다단계 미션을 실행할 때 사용한다.

---

## 3. 모듈 간 직접 상호작용 목록

| 호출하는 코드 | 호출받는 코드 | 상호작용 내용 |
|---|---|---|
| `drone_unified_node.py` | `qwen_engine.py` | 상태 문자열과 자연어를 전달하고 Tool Call을 받음 |
| `drone_unified_node.py` | `mission_executor.py` | 검증된 Tool Call 큐를 전달하고 timer tick 실행 |
| `drone_unified_node.py` | `px4_interface.py` | Node 객체와 기체 ID를 주입해 publisher 생성 |
| `qwen_engine.py` | `schema.py` | 시스템 프롬프트와 허용 Tool 이름 사용 |
| `qwen_engine.py` | `utils.py` | JSON 추출, Tool 정규화, 숫자 검증 |
| `mission_executor.py` | `px4_interface.py` | ARM, OFFBOARD, LAND, heartbeat, setpoint 요청 |
| `mission_executor.py` | `utils.py` | yaw 정규화와 숫자 변환 |
| `px4_interface.py` | `utils.py` | yaw 정규화 |
| `drone_unified_node.py` | PX4 출력 토픽 | 위치, 상태, 착륙 감지 수신 |
| `offboard_ctrl_example.py` | PX4 입력/출력 토픽 | 독립적으로 heartbeat, setpoint, command 발행 및 상태 수신 |

---

## 4. 대표 실행 시나리오

### 예시: “3m 이륙하고 앞으로 2m 이동한 뒤 착륙해”

1. 사용자가 터미널에 자연어 명령을 입력한다.
2. `DroneUnifiedNode.user_input_loop()`가 입력을 받는다.
3. `process_command()`가 현재 pose와 history를 상태 문자열로 만든다.
4. `QwenToolCaller.build_prompt()`가 시스템 지침과 사용자 명령을 결합한다.
5. Qwen이 다음과 유사한 두 Tool Call을 생성한다.

```text
<tool_call>{"name":"takeoff","arguments":{"altitude":3}}</tool_call>
<tool_call>{"name":"move","arguments":{"dx":2,"dy":0,"dz":0,"d_yaw":0}}</tool_call>
<tool_call>{"name":"land","arguments":{}}</tool_call>
```

6. `parse_tool_calls()`가 호출을 추출한다.
7. `validate_all()`이 안전 한계를 확인한다.
8. `MissionExecutor.enqueue_tool_calls()`가 순서대로 큐에 저장한다.
9. `start_next_mission()`이 `takeoff` 목표를 만들고 `PX4Interface.engage_offboard()`와 `arm()`을 호출한다.
10. timer가 반복적으로 heartbeat와 이륙 setpoint를 발행한다.
11. 목표 고도에 도달하면 첫 Tool이 완료되고 `move`가 시작된다.
12. `move`는 현재 yaw를 기준으로 목표 x/y/z를 계산한다.
13. 목표에 연속 도착하면 `land`가 시작된다.
14. PX4의 `VehicleLandDetected.landed`가 참이 되면 미션이 종료된다.

---

## 5. 설계상 장점

1. **책임 분리**: LLM, 미션 로직, PX4 통신, 순수 유틸리티가 분리되어 있다.
2. **안전 검증 지점 명확화**: 모델 출력은 실행 전에 `validate_all()`을 통과해야 한다.
3. **테스트 용이성**: `MissionExecutor`와 `utils.py`는 ROS 2 없이 mock으로 테스트할 수 있다.
4. **PX4 통신 감사 용이성**: 실제 명령 발행 코드는 `px4_interface.py`에 집중되어 있다.
5. **독립적인 하드웨어 확인**: `offboard_ctrl_example.py`로 LLM 계층과 무관하게 PX4 연결을 검증할 수 있다.
6. **동시성 보호**: ROS callback, timer, 사용자 입력 스레드 사이의 상태 접근을 `RLock`으로 보호한다.

## 6. 결론

이 시스템은 `자연어 해석`, `Tool 검증`, `미션 상태 관리`, `PX4 메시지 발행`, `텔레메트리 수신`을 계층으로 나눈 구조다.

- `schema.py`가 명령의 문법과 범위를 정한다.
- `qwen_engine.py`가 자연어를 구조화된 명령으로 바꾼다.
- `mission_executor.py`가 명령을 실제 목표와 순서로 변환한다.
- `px4_interface.py`가 PX4 메시지를 발행한다.
- `drone_unified_node.py`가 모든 계층을 ROS 2 환경에서 조립한다.
- `utils.py`가 계층 사이에서 공통 변환을 제공한다.
- `offboard_ctrl_example.py`는 통합 구조와 독립된 PX4 기본 동작 검증용 예제다.

결과적으로 Qwen은 의사결정과 명령 해석을 담당하고, 실제 비행 안전과 좌표 계산 및 PX4 제어는 결정론적인 Python 상태 머신과 통신 계층이 담당한다.
