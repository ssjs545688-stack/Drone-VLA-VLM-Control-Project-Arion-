# PX4 드론 비행 제어 테스트 명령어 가이드

본 문서는 `flight_controller` 노드의 이륙(`takeoff`), 3차원 상대 이동(`move`), 착륙(`land`) 기능을 테스트하기 위한 ROS 2 명령어 모음입니다.

---

## 📌 테스트 전 주의사항 (`--once` 옵션을 사용하는 이유)

> [!IMPORTANT]
> `ros2 topic pub` 명령어에 **`--once`**를 붙이지 않으면 ROS 2는 기본적으로 **1초마다 동일한 명령을 무한 반복 발행(1Hz)**합니다.
> 실제 LLM(`llm_service.py`)도 사용자가 명령할 때 **단 1회(One-shot)**만 토픽을 발행하므로, 동일한 동작 환경을 만들고 드론이 끝없이 날아가는 폭주를 방지하기 위해 **반드시 `--once`를 포함**해야 합니다.

---

## 🛠️ 사전 준비 (시뮬레이터, 브릿지 및 노드 실행)

시뮬레이션 및 비행 제어를 위해 아래 순서대로 각각의 터미널 창에서 실행합니다.

### 🖥️ 터미널 1: QGroundControl (QGC) 실행
지상 관제 및 드론 상태/고도를 모니터링합니다. (QGC 파일이 있는 경로에서 실행)
```bash
./QGroundControl-x86_64.AppImage
```

### 🌉 터미널 2: MicroXRCEAgent 실행
PX4 uORB 메시지와 ROS 2 토픽(`px4_msgs`) 간의 통신 브릿지를 연결합니다.
```bash
MicroXRCEAgent udp4 -p 8888
```

### 🚁 터미널 3: PX4 SITL & Gazebo 시뮬레이터 실행
가상 드론 기체와 월드 환경을 띄웁니다.
```bash
cd ~/PX4-Autopilot
PX4_SYS_AUTOSTART=4010 \
PX4_SIM_MODEL=gz_x500_mono_cam \
PX4_GZ_MODEL_POSE="0,0,0.1,0,0,1.57" \
PX4_GZ_WORLD=default \
~/PX4-Autopilot/build/px4_sitl_default/bin/px4
```
#. PX4_GZ_MODEL_POSE="0,0,0.1,0,0,1.57"
#. PX4:NED 북쪽이 0도, 가제보:ENU 동쪽이 0도, π/2~=1.57

### 🎮 터미널 4: ROS 2 빌드 및 `flight_controller` 실행
패키지를 빌드하고 비행 제어 노드를 구동하여 명령 대기 상태로 만듭니다.
```bash
cd ~/Drone-VLA-VLM-Control-Project-Arion-/ros2_ws
colcon build --packages-select llm_drone_control
source install/setup.bash
ros2 run llm_drone_control flight_controller
```

---

## 🎮 비행 테스트 시나리오별 명령어

새 터미널(**터미널 5**)을 열고 환경 설정 후 아래 명령어를 순서대로 실행해 보세요.

```bash
source /opt/ros/humble/setup.bash
source ~/Tae_ws/Drone-VLA-VLM-Control-Project-Arion-/ros2_ws/install/setup.bash
```


### 1단계: 이륙 (Takeoff)

#### 목표 고도 2.0m로 이륙
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"takeoff\", \"arguments\": {\"altitude\": 2.0}}</tool_call>'}"
```
* **결과 확인**: 모터 시동 ➔ Offboard 전환 ➔ 2m 상승 후 상태가 `HOVER`로 변경됨.

---

### 2단계: 고도 변경 (상승 / 하강)

#### ⬆️ 위로 1.0m 추가 상승 (고도 2.0m ➔ 3.0m)
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 0.0, \"dz\": 1.0, \"d_yaw\": 0.0}}</tool_call>'}"
```

#### ⬇️ 아래로 0.5m 하강 (고도 3.0m ➔ 2.5m)
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 0.0, \"dz\": -0.5, \"d_yaw\": 0.0}}</tool_call>'}"
```

---

### 3단계: 수평 이동 (전진 / 후진 / 좌우)

> 드론이 현재 바라보는 머리 방향(기수, Yaw) 기준입니다.

#### ⬆️ 앞으로 2.0m 전진
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 2.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
```

#### ⬇️ 뒤로 1.0m 후진
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": -1.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
```

#### ⬅️ 왼쪽으로 1.5m 이동
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 1.5, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
```

#### ➡️ 오른쪽으로 1.5m 이동
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": -1.5, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
```

---

### 4단계: 제자리 회전 및 복합 입체 이동

#### 🔄 반시계방향(좌측)으로 90도 회전
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": 90.0}}</tool_call>'}"
```

#### 🔄 시계방향(우측)으로 45도 회전
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": -45.0}}</tool_call>'}"
```

#### 🚀 복합 이동 (앞으로 2m 가면서 1m 상승)
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 2.0, \"dy\": 0.0, \"dz\": 1.0, \"d_yaw\": 0.0}}</tool_call>'}"
```

---

### 5단계: 착륙 (Land)

#### 현재 위치에서 지면으로 자동 착륙
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"land\", \"arguments\": {}}</tool_call>'}"
```
* **결과 확인**: 하강 ➔ 지면 접촉 감지(`landed == True`) ➔ 모터 정지 ➔ 상태가 `IDLE`로 복귀.

---

### 6단계: 복합 연속 명령 및 미션 큐 (이륙 ➔ 이동 ➔ 착륙)

한 번의 응답에 여러 개의 `<tool_call>`이 포함되어 있을 때, 비행 제어기(`flight_controller`)가 미션 큐(Queue)를 통해 순서대로 자동 실행하는지 테스트합니다.

#### 🚀 4.0m 이륙 ➔ 복합 입체 이동 (앞 1.0m, 왼쪽 2.0m, 상승 1.0m) ➔ 자동 착륙
```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"takeoff\", \"arguments\": {\"altitude\": 4.0}}</tool_call>\n<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 1.0, \"dy\": 2.0, \"dz\": 1.0, \"d_yaw\": 0.0}}</tool_call>\n<tool_call>{\"name\": \"land\", \"arguments\": {}}</tool_call>'}"
```
* **동작 흐름 및 결과 확인**:
  1. `ARMING` ➔ `TAKEOFF` 수행: 목표 고도 **4.0m**까지 수직 상승 후 1초간 호버링 안정화 확인.
  2. 고도 도달 즉시 대기열에서 다음 명령 꺼냄 ➔ `MOVE` 수행:
     * 현재 기수(머리) 기준으로 **앞으로 1.0m (`dx: 1.0`)**, **왼쪽으로 2.0m (`dy: 2.0`)** 이동.
     * 동시에 고도를 **1.0m 추가 상승 (`dz: 1.0`)**하여 **총 고도 5.0m**에 도달 및 호버링 안정화 확인.
  3. 입체 이동 완료 즉시 대기열에서 착륙 명령 꺼냄 ➔ `LAND` 수행:
     * 현재 5.0m 공중 위치에서 지면으로 자동 하강 ➔ 착륙 센서 감지(`landed == True`) ➔ 모터 시동 꺼짐(Disarm) ➔ 상태가 `IDLE`로 복귀.

---


## 🛡️ 안전 가드 작동 테스트 (이륙 전 이동 시도)

드론이 착륙해 있는 상태(`IDLE`)에서 이동 명령을 보내면 안전 가드가 정상 동작하는지 확인해 봅니다:

```bash
ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 2.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
```
* **기대 결과**:
  ```text
  [WARN] [flight_controller]: ⚠️ [명령 거부] 기체가 비행 중이 아닙니다 (현재 상태: IDLE). 먼저 이륙(takeoff)을 수행하세요!
  ```

---

## 💡 [참고] 실제 LLM 서비스 연동 시 자연어 명령 호출법

`llm_service` 노드를 띄운 상태라면 토픽 직접 발행 대신 아래처럼 자연어로 요청할 수 있습니다:

```bash
ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '고도 2미터로 이륙해줘'}"
ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '앞으로 2미터 이동하고 1미터 올라가줘'}"
ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '이제 착륙해줘'}"
```
