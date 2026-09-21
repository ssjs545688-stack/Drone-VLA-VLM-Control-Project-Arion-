# 🚁 PX4 드론 비행 제어 및 LLM 연동 테스트 명령어 가이드

본 문서는 `flight_controller` 노드의 이륙(`takeoff`), 3차원 상대 이동 및 회전(`move`), 과거 위치 복귀(`goto_history`), 착륙(`land`) 기능을 검증하기 위한 ROS 2 테스트 명령어 모음입니다.

모든 테스트 단계는 다음 **두 가지 방식**을 모두 지원하도록 작성되었습니다:
1. **직접 토픽 발행 (`ros2 topic pub --once /llm_response`)**: LLM 모델 추론 없이 비행 제어기의 모터/좌표 제어 로직만 빠르게 단위 검증
2. **LLM 자연어 서비스 호출 (`ros2 service call /llm`)**: Qwen3 + LoRA 모델이 자연어 문장을 실시간 해석하여 비행까지 이어지는 전체 파이프라인 통합 검증

---

## 📌 테스트 전 주의사항 (`--once` 옵션을 사용하는 이유)

> [!IMPORTANT]
> `ros2 topic pub` 명령어에 **`--once`**를 붙이지 않으면 ROS 2는 기본적으로 **1초마다 동일한 명령을 무한 반복 발행(1Hz)**합니다.
> 실제 LLM(`llm_service.py`)도 사용자가 명령할 때 **단 1회(One-shot)**만 토픽을 발행하므로, 동일한 동작 환경을 만들고 드론이 끝없이 날아가는 폭주를 방지하기 위해 **반드시 `--once`를 포함**해야 합니다.

---

## 🛠️ 사전 준비 (시뮬레이터, 브릿지 및 노드 실행)

시뮬레이션 및 비행 제어를 위해 아래 순서대로 각각의 터미널 창에서 실행합니다.

### 🖥️ 터미널 1: QGroundControl (QGC) 실행
지상 관제 및 드론 비행 상태/고도를 모니터링합니다. (QGC 파일이 있는 경로에서 실행)
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
> *PX4_GZ_MODEL_POSE="0,0,0.1,0,0,1.57"는 PX4(NED, 북쪽 0도)와 Gazebo(ENU, 동쪽 0도) 간의 헤딩 오프셋(π/2 ≈ 1.57 rad)을 보정합니다.*

### 🎮 터미널 4: ROS 2 통합 런치 실행 (LLM + 제어기 + STT)
패키지를 빌드하고 메인 노드들을 구동합니다.
```bash
cd ~/Drone-VLA-VLM-Control-Project-Arion-/ros2_ws
colcon build --symlink-install --packages-select llm_drone_control
source install/setup.bash
ros2 launch llm_drone_control drone_control.launch.py
```

---

## 🎮 비행 테스트 시나리오별 명령어

새 터미널(**터미널 5**)을 열고 환경 설정 후 아래 명령어를 순서대로 실행해 보세요.

```bash
source /opt/ros/humble/setup.bash
source ~/Drone-VLA-VLM-Control-Project-Arion-/ros2_ws/install/setup.bash
```

---

### 1단계: 이륙 (Takeoff)

#### 목표 고도 2.0m로 수직 이륙
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"takeoff\", \"arguments\": {\"altitude\": 2.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '2m 높이로 이륙해줘'}"
  ```
* **결과 확인**: 모터 시동 ➔ Offboard 전환 ➔ 2.0m 상승 후 상태가 `HOVER`로 안착. 최초 출발점 `Waypoint #0 (X=0.00m, Y=0.00m)` 등록 확인.

---

### 2단계: 고도 변경 (상승 / 하강)

#### ⬆️ 위로 1.0m 추가 상승 (고도 2.0m ➔ 3.0m)
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 0.0, \"dz\": 1.0, \"d_yaw\": 0.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '위로 1미터 더 올라가줘'}"
  ```

#### ⬇️ 아래로 0.5m 하강 (고도 3.0m ➔ 2.5m)
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 0.0, \"dz\": -0.5, \"d_yaw\": 0.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '아래로 50센티미터 내려가'}"
  ```

---

### 3단계: 수평 이동 (전진 / 후진 / 좌우)

> 드론이 현재 바라보는 머리 방향(기수, Yaw) 기준 상대 이동입니다.

#### ⬆️ 앞으로 2.0m 전진
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 2.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '앞으로 2미터 전진해'}"
  ```

#### ⬇️ 뒤로 1.0m 후진
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": -1.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '뒤로 1m 가줘'}"
  ```

#### ⬅️ 왼쪽으로 1.5m 이동
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 1.5, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '왼쪽으로 1.5미터 이동해줘'}"
  ```

#### ➡️ 오른쪽으로 1.5m 이동
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": -1.5, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '오른쪽으로 1.5미터 가'}"
  ```

---

### 4단계: 제자리 회전 및 복합 입체 이동

#### 🔄 반시계방향(좌측)으로 90도 회전
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": 90.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '반시계 방향으로 90도 틀어줘'}"
  ```

#### 🔄 시계방향(우측)으로 45도 회전
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 0.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": -45.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '시계 방향으로 45도 회전해'}"
  ```

#### 🚀 복합 입체 이동 (앞으로 2m 가면서 1m 상승)
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 2.0, \"dy\": 0.0, \"dz\": 1.0, \"d_yaw\": 0.0}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '앞으로 2미터 가면서 위로 1미터 올라가줘'}"
  ```

---

### 5단계: 과거 위치 직선 복귀 (goto_history) ⭐

스택(Stack) 기반으로 기록된 과거 비행 좌표를 거슬러 올라가거나 최초 출발지(Home)로 복귀합니다.

#### 🔙 [복귀 A] 직전 위치 복귀 1회차 (방금 전 머물렀던 위치로 복귀)
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"goto_history\", \"arguments\": {\"recall\": \"previous\"}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '방금 전 위치로 돌아가'}"
  ```
* **확인**: 스택의 최상단(현재 위치)이 Pop되고 직전 안착 위치로 직선 비행 후 안착.

#### 🔙 [복귀 B] 직전 위치 복귀 2회차 (연속 복귀 ➔ 그 이전 위치/출발지 복귀 ⭐)
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"goto_history\", \"arguments\": {\"recall\": \"previous\"}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '직전 위치로 복귀해줘'}"
  ```
* **확인**: 실행할 때마다 한 단계씩 과거로 이동하여 최초 출발지까지 계속 거슬러 올라감 확인.

#### 🏠 [복귀 C] 최초 출발지 즉시 직선 복귀 (recall: "first") ⭐
드론이 여러 차례 이동해 있더라도, 중간 단계를 모두 무시하고 최초 이륙 지점(Home)으로 단번에 복귀합니다.
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"goto_history\", \"arguments\": {\"recall\": \"first\"}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '처음 출발했던 곳으로 돌아가'}"
  # 또는
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '원점으로 바로 복귀해라'}"
  ```
* **확인**: 최초 출발점`(X=0.00m, Y=0.00m)`으로 즉시 직선 복귀 후 스택을 `[원점]` 하나로 리셋.

---

### 6단계: 착륙 (Land)

#### 현재 위치에서 지면으로 자동 착륙
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"land\", \"arguments\": {}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '안전하게 착륙해줘'}"
  # 또는
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '지금 위치에 착륙해'}"
  ```
* **결과 확인**: 하강 ➔ 지면 접촉 감지(`landed == True`) ➔ 모터 정지 ➔ 상태가 `IDLE`로 복귀.

---

### 7단계: 복합 연속 명령 및 미션 큐 (이륙 ➔ 이동 ➔ 착륙)

한 번의 응답에 여러 개의 Tool Call이 포함되어 있을 때, 비행 제어기(`flight_controller`)가 미션 큐(Queue)를 통해 순서대로 자동 실행하는지 테스트합니다.

#### 🚀 4.0m 이륙 ➔ 복합 입체 이동 (앞 1.0m, 왼쪽 2.0m, 상승 1.0m) ➔ 자동 착륙
* **[방법 A] 직접 토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"takeoff\", \"arguments\": {\"altitude\": 4.0}}</tool_call>\n<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 1.0, \"dy\": 2.0, \"dz\": 1.0, \"d_yaw\": 0.0}}</tool_call>\n<tool_call>{\"name\": \"land\", \"arguments\": {}}</tool_call>'}"
  ```
* **[방법 B] LLM 모델 연동 자연어 명령:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '4미터 높이로 이륙해서 앞으로 1미터 왼쪽으로 2미터 가면서 1미터 올라간 다음 착륙해줘'}"
  ```
* **동작 흐름 및 결과 확인**:
  1. `ARMING` ➔ `TAKEOFF`: 목표 고도 **4.0m**까지 수직 상승 후 1초간 호버링 안정화.
  2. 큐에서 다음 명령 팝 ➔ `MOVE`: 전진 1.0m, 좌측 2.0m, 고도 +1.0m(총 5.0m) 도달 및 호버링 안정화.
  3. 큐에서 다음 명령 팝 ➔ `LAND`: 현재 위치에서 지면으로 자동 하강 ➔ 착륙 감지 ➔ 모터 시동 종료(`IDLE`).

---

## 🛡️ 안전 가드 및 예외 처리 테스트

### 1) 이륙 전 이동 명령 차단
드론이 착륙해 있는 상태(`IDLE`)에서 이동 명령을 보내면 안전 가드가 정상 동작하는지 확인합니다:
* **토픽 발행:**
  ```bash
  ros2 topic pub --once /llm_response std_msgs/msg/String "{data: '<tool_call>{\"name\": \"move\", \"arguments\": {\"dx\": 2.0, \"dy\": 0.0, \"dz\": 0.0, \"d_yaw\": 0.0}}</tool_call>'}"
  ```
* **기대 결과**:
  ```text
  [WARN] [flight_controller]: ⚠️ [명령 거부] 기체가 비행 중이 아닙니다 (현재 상태: IDLE). 먼저 이륙(takeoff)을 수행하세요!
  ```

### 2) 드론 제어와 무관한 일상 대화 거부 (LLM 네거티브 필터링)
비행 제어와 상관없는 질문을 했을 때 오작동하지 않고 거부 메시지가 나오는지 확인합니다:
* **자연어 요청:**
  ```bash
  ros2 service call /llm guide_interfaces/srv/GuideLLM "{prompt: '오늘 날씨 어때?'}"
  ```
* **기대 결과**:
  ```text
  response: '지원하지 않는 명령입니다.'
  ```
  *(비행 제어기로 아무런 `<tool_call>`도 전달되지 않음)*
