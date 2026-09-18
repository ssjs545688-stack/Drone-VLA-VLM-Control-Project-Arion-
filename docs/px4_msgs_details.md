# PX4 핵심 ROS 2 uORB 메시지 상세 분석 보고서

본 문서는 `flight_controller.py`에서 PX4 Autopilot과 통신할 때 사용하는 6개 핵심 메시지(`px4_msgs.msg`)의 필드 구조, 데이터 타입, 의미 및 실제 제어 활용법을 상세히 정리한 문서입니다.

---

## 1. 개요 및 메시지 방향

PX4와 ROS 2 간의 uORB 메시지 통신은 크게 **입력(명령 송신: `/fmu/in/...`)**과 **출력(상태 수신: `/fmu/out/...`)**으로 나뉩니다.

| 메시지 이름 | 토픽 방향 | 기본 토픽명 | 핵심 역할 |
|---|---|---|---|
| **OffboardControlMode** | 송신 (Node ➔ PX4) | `/fmu/in/offboard_control_mode` | 오프보드 제어 방식(위치/속도 등) 선언 및 생존 하트비트 |
| **TrajectorySetpoint** | 송신 (Node ➔ PX4) | `/fmu/in/trajectory_setpoint` | 목표 위치(x, y, z), 속도, 가속도, Yaw 각도 전달 |
| **VehicleCommand** | 송신 (Node ➔ PX4) | `/fmu/in/vehicle_command` | MAVLink 기반 명령 (모터 시동, 모드 변경, 착륙 등) |
| **VehicleLocalPosition** | 수신 (PX4 ➔ Node) | `/fmu/out/vehicle_local_position` | EKF2 추정 로컬 NED 위치, 속도, 가속도, 헤딩 피드백 |
| **VehicleStatus** | 수신 (PX4 ➔ Node) | `/fmu/out/vehicle_status` | 기체 시동 상태(ARM/DISARM), 현재 비행 모드, Failsafe 상태 |
| **VehicleLandDetected** | 수신 (PX4 ➔ Node) | `/fmu/out/vehicle_land_detected` | 착륙 감지 상태 머신 (지면 접촉, 착륙 완료 여부) |

---

## 2. 메시지별 상세 스펙

---

### ① `OffboardControlMode.msg`
> **역할**: PX4에게 외부 제어 노드가 살아있음을 알리는 **하트비트(Heartbeat)**이자, **어떤 물리량(위치, 속도, 가속도 등)으로 드론을 제어할 것인지**를 지정하는 플래그 메시지입니다.

#### 필드 정의
```text
uint64 timestamp             # 시스템 시작 이후 시간 (microseconds)
bool position                # True: 위치(Position) 제어 활성화
bool velocity                # True: 속도(Velocity) 제어 활성화
bool acceleration            # True: 가속도(Acceleration) 제어 활성화
bool attitude                # True: 자세(Attitude) 제어 활성화
bool body_rate               # True: 각속도(Body Rate) 제어 활성화
bool thrust_and_torque       # True: 추력 및 토크 직접 제어
bool direct_actuator         # True: 모터/서보 개별 제어
```

#### 비행 제어기(`flight_controller.py`)에서의 활용
- `msg.position = True`로 설정하고 나머지를 `False`로 설정하여 **위치 기반 제어**를 선언합니다.
- **주의점**: PX4는 이 메시지가 10Hz 이상의 주기로 지속 수신되지 않으면 Offboard 모드를 즉시 해제하고 안전 Failsafe 모드(예: POSCTL 또는 Land)로 진입합니다.

---

### ② `TrajectorySetpoint.msg`
> **역할**: 드론이 이동해야 하는 목표 공간 좌표와 방향을 전달합니다. PX4 내부의 PID 위치/자세 제어기의 입력으로 사용됩니다.

#### 필드 정의
```text
uint64 timestamp             # 시스템 시작 이후 시간 (microseconds)

# NED 로컬 좌표계 기준 (단위: m, m/s, m/s^2)
float32[3] position          # 목표 위치 [North, East, Down] (m)
float32[3] velocity          # 목표 속도 [Vx, Vy, Vz] (m/s)
float32[3] acceleration      # 목표 가속도 [Ax, Ay, Az] (m/s^2)
float32[3] jerk              # 저크 (m/s^3, 로깅 전용)

float32 yaw                  # 목표 헤딩 Euler Yaw 각도 (단위: radian, -π ~ +π)
float32 yawspeed             # 목표 Yaw 각속도 (단위: rad/s)
```

#### 비행 제어기(`flight_controller.py`)에서의 활용
- `position = [x, y, z]`를 채워 보냅니다.
- **NED 좌표계 특성**:
  - `x`: 북쪽(전진 +)
  - `y`: 동쪽(우측 +)
  - `z`: 아래쪽(지면 + / **상승은 음수 `-z`**)
- 예: 2m 상승 이륙 시 `position = [0.0, 0.0, -2.0]`, `yaw = 0.0`을 발행합니다.
- 제어하지 않는 항목은 `NaN`(Not a Number)을 넣으면 해당 축 제어가 무시됩니다.

---

### ③ `VehicleCommand.msg`
> **역할**: MAVLink 프로토콜의 `COMMAND_LONG` / `COMMAND_INT`와 1:1로 대응되는 범용 기체 명령 메시지입니다. 모터 시동/정지, 비행 모드 변경, 비상 정지 등 모든 명령을 이 메시지로 수행합니다.

#### 필드 정의
```text
uint64 timestamp             # 시스템 시작 이후 시간 (microseconds)

float32 param1               # 명령별 파라미터 1
float32 param2               # 명령별 파라미터 2
float32 param3               # 명령별 파라미터 3
float32 param4               # 명령별 파라미터 4
float64 param5               # 명령별 파라미터 5 (위도 등 정밀값용)
float64 param6               # 명령별 파라미터 6 (경도 등 정밀값용)
float32 param7               # 명령별 파라미터 7 (고도 등)

uint32 command               # 명령 ID (상수값)
uint8 target_system          # 명령 수신 대상 시스템 ID (기본값: 1)
uint8 target_component       # 명령 수신 대상 컴포넌트 ID (기본값: 1)
uint8 source_system          # 송신측 시스템 ID (기본값: 1)
uint16 source_component      # 송신측 컴포넌트 ID
uint8 confirmation           # 재전송 확인 횟수 (0: 첫 전송)
bool from_external           # 외부 명령 여부 (True)
```

#### 주요 명령어 상수 (Constants)
- `VEHICLE_CMD_COMPONENT_ARM_DISARM = 400`: 모터 시동/해제
  - `param1 = 1.0` (ARM: 시동 켜기)
  - `param1 = 0.0` (DISARM: 시동 끄기)
- `VEHICLE_CMD_DO_SET_MODE = 176`: 비행 모드 변경
  - `param1 = 1.0` (커스텀 모드 플래그)
  - `param2 = 6.0` (`PX4_CUSTOM_MAIN_MODE_OFFBOARD` ➔ 오프보드 모드 전환)
- `VEHICLE_CMD_NAV_LAND = 21`: 자동 착륙 시작
- `VEHICLE_CMD_NAV_TAKEOFF = 22`: 자동 이륙 시작

---

### ④ `VehicleLocalPosition.msg`
> **역할**: 드론의 센서들(IMU, GPS, 기압계 등)을 EKF2(확장 칼만 필터)로 융합하여 계산한 **로컬 3차원 위치/속도/헤딩 추정치**를 제공합니다.

#### 필드 정의
```text
uint64 timestamp             # 시스템 시작 이후 시간 (microseconds)
uint64 timestamp_sample      # 원시 센서 데이터 타임스탬프

# 유효성 플래그
bool xy_valid                # 수평 위치 (x, y) 유효 여부
bool z_valid                 # 수직 고도 (z) 유효 여부
bool v_xy_valid              # 수평 속도 (vx, vy) 유효 여부
bool v_z_valid               # 수직 속도 (vz) 유효 여부

# 로컬 NED 좌표 (원점: EKF2 시동 위치)
float32 x                    # North (m)
float32 y                    # East (m)
float32 z                    # Down (m) (고도가 높아질수록 음수)

# 로컬 속도 및 가속도
float32 vx                   # North 속도 (m/s)
float32 vy                   # East 속도 (m/s)
float32 vz                   # Down 속도 (m/s)
float32 ax, ay, az           # 축별 가속도 (m/s^2)

# 기수 방향(Heading)
float32 heading              # NED 기준 Euler Yaw 각도 (단위: rad, -π ~ +π)

# 지면과의 상대 거리 (거리 센서/라이다 연동 시)
float32 dist_bottom          # 지면과의 실제 거리 (m)
bool dist_bottom_valid       # 지면 거리 유효 여부

# 오차 추정 (표준편차)
float32 eph                  # 수평 위치 오차 (m)
float32 epv                  # 수직 고도 오차 (m)
```

#### 비행 제어기(`flight_controller.py`)에서의 활용
- `msg.z`를 수신하여 현재 기체의 실시간 고도를 모니터링합니다.
- 초기 이륙 시 `ground_z = current_z`로 지면을 기록하고, `target_z = ground_z - alt`와 비교하여 드론이 목표 고도에 도달했는지 오차(`abs(current_z - target_z) < 0.1`)를 판정합니다.

---

### ⑤ `VehicleStatus.msg`
> **역할**: PX4의 비행 제어 총괄 모듈인 Commander가 발행하는 **기체의 전체 상태(시동 상태, 현재 활성화된 비행 모드, 페일세이프 여부 등)**를 나타냅니다.

#### 필드 정의 및 주요 상수
```text
uint64 timestamp             # 시스템 시작 이후 시간 (microseconds)
uint64 armed_time            # 시동 걸린 시간
uint64 takeoff_time          # 이륙 시간

# 1. 시동 상태
uint8 arming_state           # 현재 시동 상태
uint8 ARMING_STATE_DISARMED = 1
uint8 ARMING_STATE_ARMED    = 2

# 2. 비행 모드 (Navigation State)
uint8 nav_state              # 현재 실행 중인 비행 모드
uint8 NAVIGATION_STATE_MANUAL = 0         # 수동 모드
uint8 NAVIGATION_STATE_ALTCTL = 1         # 고도 제어 모드
uint8 NAVIGATION_STATE_POSCTL = 2         # 위치 제어 모드
uint8 NAVIGATION_STATE_AUTO_MISSION = 3    # 미션 비행
uint8 NAVIGATION_STATE_AUTO_RTL = 5        # 복귀(RTL)
uint8 NAVIGATION_STATE_OFFBOARD = 14       # 오프보드 제어 모드 (ROS2 제어 필수 모드)
uint8 NAVIGATION_STATE_AUTO_LAND = 18      # 자동 착륙 모드

# 3. 안전 및 고장 감지
bool failsafe                # True면 시스템 페일세이프 발동 중
uint16 failure_detector_status # 고장 감지 비트마스크 (롤, 피치, 배터리, 모터 등)
bool pre_flight_checks_pass  # 시동 전 검사 통과 여부 (True여야 ARM 가능)
```

#### 비행 제어기(`flight_controller.py`)에서의 활용
- `msg.arming_state == VehicleStatus.ARMING_STATE_ARMED`를 검사하여 모터가 실제로 돌고 있는지 안전 상태를 확인합니다.
- `msg.nav_state`가 실제로 `NAVIGATION_STATE_OFFBOARD (14)`로 전환되었는지 검증할 때 사용합니다.

---

### ⑥ `VehicleLandDetected.msg`
> **역할**: PX4의 착륙 감지 알고리즘이 가속도계, 고도계, 스로틀 상태 등을 종합 분석하여 **드론이 공중에 떠있는지, 하강 중인지, 지면에 착륙했는지 단계별로 판단한 상태**를 알려줍니다.

#### 필드 정의
```text
uint64 timestamp             # 시스템 시작 이후 시간 (microseconds)

bool freefall                # True: 자유낙하(추락) 상태 감지
bool ground_contact          # True: 지면에 바퀴/다리가 닿음 (1단계 감지)
bool maybe_landed            # True: 스로틀이 낮아지고 지면에 안착한 것으로 추정 (2단계)
bool landed                  # True: 완전히 지면에 정지하여 착륙 완료 (3단계 최종 판정)

bool in_ground_effect        # 지면 효과 영역 진입 여부
bool in_descend              # 하강 비행 중 여부
bool has_low_throttle        # 최소 스로틀 상태 여부
bool vertical_movement       # 수직 움직임 감지 여부
bool horizontal_movement     # 수평 움직임 감지 여부
bool rotational_movement     # 회전 움직임 감지 여부
bool at_rest                 # 기체가 완전 정지 상태인지 여부
```

#### 비행 제어기(`flight_controller.py`)에서의 활용
- 착륙 명령(`land`)을 보낸 후 드론이 바닥에 닿았는지 판정할 때 `msg.landed == True`를 확인합니다.
- `msg.landed`가 True가 되고 시동이 꺼지면 비행 상태를 `IDLE`로 안전하게 복귀시킵니다.

---

## 3. 요약: 데이터 상호작용 구조

```text
[ ROS 2 Control Node ]
       │
       ├──── 1. OffboardControlMode ────► [ PX4 FMU ] : "위치 제어 모드로 통신 유지(하트비트)"
       ├──── 2. TrajectorySetpoint  ────► [ PX4 FMU ] : "목표 좌표 [0, 0, -2.0] 로 이동하라"
       ├──── 3. VehicleCommand      ────► [ PX4 FMU ] : "시동(ARM) 켜라 / Offboard 모드로 바꿔라"
       │
       ◄──── 4. VehicleLocalPosition ─── [ PX4 FMU ] : "현재 내 위치는 [x, y, z] 이다"
       ◄──── 5. VehicleStatus       ─── [ PX4 FMU ] : "현재 시동 켜짐(ARMED), Offboard 모드 작동 중이다"
       ◄──── 6. VehicleLandDetected ─── [ PX4 FMU ] : "현재 공중 비행 중이다 (landed=False)"
```
