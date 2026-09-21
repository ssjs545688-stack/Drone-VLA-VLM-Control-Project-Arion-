#!/usr/bin/env python3
"""
flight_controller.py

/llm_response 토픽을 구독하여 <tool_call> 파싱 후,
PX4 Autopilot을 Offboard 모드로 전환하여 이륙(takeoff) 및 착륙(land)을 수행하는 노드.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String

# PX4 uORB 메시지 인터페이스
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
    VehicleLandDetected
)

from collections import deque
import json
import re
import math


class FlightController(Node):
    def __init__(self):
        super().__init__("flight_controller")

        # -------------------------------------------------------------
        # 1. QoS 프로파일 설정 (PX4 통신용 Best Effort QoS)
        # -------------------------------------------------------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # -------------------------------------------------------------
        # 2. PX4 퍼블리셔 (명령 및 제어 신호 전송)
        # -------------------------------------------------------------
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_profile
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_profile
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_profile
        )

        # -------------------------------------------------------------
        # 3. PX4 서브스크라이버 (상태 및 피드백 수신)
        # -------------------------------------------------------------
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.position_callback, qos_profile
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status", self.status_callback, qos_profile
        )
        self.create_subscription(
            VehicleLandDetected, "/fmu/out/vehicle_land_detected", self.land_detected_callback, qos_profile
        )

        # -------------------------------------------------------------
        # 4. LLM 응답 토픽 구독
        # -------------------------------------------------------------
        self.create_subscription(
            String, "/llm_response", self.llm_response_callback, 10
        )

        # -------------------------------------------------------------
        # 5. 내부 상태 변수 관리
        # -------------------------------------------------------------
        # 소프트웨어 원점 보정용 (초기 센서 EKF 오차 -0.2m~-0.3m 제거)
        self.origin_x = None
        self.origin_y = None
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.current_yaw = 0.0
        self.is_armed = False
        self.nav_state = 0
        self.is_landed = True
        self.pre_flight_checks_pass = False
        self.failsafe = False
        self.vehicle_status_received = False

        # 비행 제어 상태 머신: IDLE, ARMING, TAKEOFF, HOVER, MOVE, LAND
        self.flight_state = "IDLE"
        self.target_altitude = 0.0  # 양수 (단위: m)
        self.home_z = None          # 지면 기준 고도 (Home Z, 상대 고도 계산용)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0         # 실제 계산된 목표 NED z 좌표
        self.target_yaw = 0.0       # 목표 헤딩 각도 (단위: rad)
        self.heartbeat_counter = 0
        self.arrival_counter = 0    # 목표 도달 연속 카운터

        # 미션 대기열 (Queue)
        self.mission_queue = deque()

        # =============================================================
        # 🧭 과거 위치 히스토리 관리 (스택 자료구조)
        # =============================================================
        # 완료된 안착 위치를 (x, y, z, yaw) 튜플 형태로 저장하는 스택 리스트
        self.waypoint_history = []
        # 현재 수행 중인 이동이 과거 복귀(goto_history) 이동인지 나타내는 플래그
        self.is_recovering = False

        # 10Hz 주기적 제어 루프 타이머
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("✅ [FlightController] 노드가 준비되었습니다. /llm_response 대기 중...")

    # =================================================================
    # 콜백 함수들
    # =================================================================
    def position_callback(self, msg: VehicleLocalPosition):
        """현재 드론의 로컬 NED 좌표, 속도 및 헤딩(Yaw)"""
        # 시뮬레이션/노드 시작 시 EKF2 초기 원점 오차를 (0, 0)으로 영점 캘리브레이션
        if self.origin_x is None:
            if not math.isnan(msg.x) and not math.isnan(msg.y):
                self.origin_x = msg.x
                self.origin_y = msg.y
                self.get_logger().info(
                    f"🎯 [초기 영점 보정 완료] 센서 원점 오차(X: {self.origin_x:+.2f}m, Y: {self.origin_y:+.2f}m)를 기준 원점(0.00m, 0.00m)으로 보정했습니다."
                )

        # 보정된 로컬 좌표 계산 (출발 지점이 정확히 0.00m가 됨)
        self.current_x = msg.x - (self.origin_x if self.origin_x is not None else 0.0)
        self.current_y = msg.y - (self.origin_y if self.origin_y is not None else 0.0)
        self.current_z = msg.z
        self.current_vx = msg.vx
        self.current_vy = msg.vy
        self.current_vz = msg.vz
        self.current_yaw = msg.heading

        # 지면에 착지해 있거나 대기 중(IDLE)일 때는 QGC와 동일하게 0.00m로 고정
        if self.flight_state == "IDLE" or self.home_z is None:
            current_altitude = 0.0
        else:
            # 비행 중에는 시동 걸린 바닥(home_z) 기준 상대 고도 계산 (QGC와 100% 일치)
            current_altitude = self.home_z - self.current_z
        yaw_deg = math.degrees(self.current_yaw)
        self.get_logger().info(
            f"📍 [드론 위치 | 상태: {self.flight_state}] X: {self.current_x:6.2f}m | Y: {self.current_y:6.2f}m | "
            f"고도: {current_altitude:5.2f}m | Yaw: {yaw_deg:6.1f}°",
            throttle_duration_sec=0.5
        )

    def status_callback(self, msg: VehicleStatus):
        """기체 시동 상태 및 비행 모드 확인"""
        self.vehicle_status_received = True
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.nav_state = msg.nav_state
        self.pre_flight_checks_pass = getattr(msg, "pre_flight_checks_pass", True)
        self.failsafe = getattr(msg, "failsafe", False)

    def land_detected_callback(self, msg: VehicleLandDetected):
        """착륙 여부 감지"""
        self.is_landed = msg.landed

    def llm_response_callback(self, msg: String):
        """LLM이 발행한 응답 문자열 파싱 후 미션 큐에 적재"""
        text = msg.data.strip()
        self.get_logger().info(f"📩 수신된 LLM 응답: {text}")

        # 정규표현식으로 모든 <tool_call> 태그 추출
        matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if not matches:
            self.get_logger().warn("⚠️ 응답에서 <tool_call> 태그를 찾지 못했습니다.")
            return

        added_count = 0
        for json_str in matches:
            try:
                call_data = json.loads(json_str)
                self.mission_queue.append(call_data)
                added_count += 1
            except Exception as e:
                self.get_logger().error(f"❌ Tool Call JSON 파싱 실패: {e}")

        self.get_logger().info(f"📥 {added_count}개의 미션이 대기열에 추가되었습니다. (현재 큐 크기: {len(self.mission_queue)})")

        # 드론이 대기 중(IDLE 또는 HOVER)이면 즉시 첫 번째 미션 시작
        if self.flight_state in ["IDLE", "HOVER"]:
            self.execute_next_mission()

    def execute_next_mission(self):
        """미션 큐에서 다음 명령을 꺼내어 실행"""
        if not self.mission_queue:
            return

        call_data = self.mission_queue.popleft()
        func_name = call_data.get("name")
        args = call_data.get("arguments", {})

        self.get_logger().info(f"▶️ [미션 시작] {func_name} (남은 대기열: {len(self.mission_queue)})")

        # 명령에 따른 동작 분기
        if func_name == "takeoff":
            # 🛡️ 안전 가드 1: 이미 비행 중인 경우 중복 이륙 건너뛰고 다음 미션 수행
            if self.flight_state in ["TAKEOFF", "HOVER", "MOVE"]:
                self.get_logger().warn(f"⚠️ [명령 건너뜀] 이미 비행 중입니다 (현재 상태: {self.flight_state}).")
                self.execute_next_mission()
                return

            # 🛡️ 안전 가드 2: PX4 기체 상태 수신 여부 확인
            if not self.vehicle_status_received:
                self.get_logger().error("❌ [이륙 거부] PX4 기체 상태(VehicleStatus)를 아직 수신하지 못했습니다.")
                self.mission_queue.clear()
                return

            # 🛡️ 안전 가드 3: 사전 점검(Pre-flight Checks) 통과 여부 확인
            if not self.pre_flight_checks_pass:
                self.get_logger().error(
                    "❌ [이륙 거부] PX4 사전 안전 점검(Pre-flight Checks) 불합격! "
                    "(센서 타임아웃, 나침반 오류, 배터리 방전 등 기체 상태를 점검하세요)"
                )
                self.mission_queue.clear()
                return

            # 🛡️ 안전 가드 4: Failsafe 상태 여부 확인
            if self.failsafe:
                self.get_logger().error("❌ [이륙 거부] 기체가 Failsafe 상태입니다. 비행을 시작할 수 없습니다.")
                self.mission_queue.clear()
                return

            alt = float(args.get("altitude", 2.0))
            self.target_altitude = alt
            # 이륙 시점의 실제 바닥(current_z)을 기준으로 목표 고도 계산 (PX4 NED: 고도 상승은 -Z 방향)
            self.home_z = self.current_z
            self.target_z = self.home_z - abs(alt)
            self.target_x = self.current_x
            self.target_y = self.current_y
            # ⚠️ target_yaw는 바닥의 틀어진 각도로 덮어쓰지 않고,
            # 사용자가 설정한 기준 헤딩(self.target_yaw, 기본 0.0°)을 그대로 유지하여 스스로 정렬하도록 함
            self.heartbeat_counter = 0
            self.arrival_counter = 0
            self.flight_state = "ARMING"
            yaw_deg = math.degrees(self.target_yaw)
            self.get_logger().info(
                f"🚀 [명령 수신] 이륙 명령: 목표 상대 고도 {alt:.2f}m (바닥 Z: {self.home_z:.2f}m ➔ 목표 NED Z: {self.target_z:.2f}m, 목표 헤딩: {yaw_deg:.1f}°)"
            )

        elif func_name == "move":
            # 🛡️ 안전 가드: 공중에 안정적으로 비행 중(HOVER 또는 MOVE)이 아니면 이동 거부 및 큐 초기화
            if self.flight_state not in ["HOVER", "MOVE"]:
                self.get_logger().warn(
                    f"⚠️ [명령 거부] 기체가 비행 중이 아닙니다 (현재 상태: {self.flight_state}). "
                    f"먼저 이륙(takeoff)을 수행하세요!"
                )
                self.mission_queue.clear()
                return

            dx = float(args.get("dx", 0.0))
            dy = float(args.get("dy", 0.0))
            dz = float(args.get("dz", 0.0))
            d_yaw_deg = float(args.get("d_yaw", 0.0))

            # 스키마 좌표계 -> PX4 FRD 바디 좌표계 변환
            # 스키마: 전진(+)/후진(-), 좌(+)/우(-), 상승(+)/하강(-), 반시계(+)/시계(-)
            # PX4: Forward(+)/Back(-), Right(+)/Left(-), Down(+)/Up(-), CW(+)/CCW(-)
            body_x = dx
            body_y = -dy  # 스키마의 '좌(+)'를 PX4의 '좌(-Y)'로 변환

            # 드론 목표 헤딩(Yaw) 기준 회전 변환 (기체 좌표계 -> NED 좌표계)
            # 일시적인 센서 흔들림(current_yaw) 대신 안정적인 목표 헤딩(target_yaw) 기준으로 전후좌우 변환
            yaw = self.target_yaw
            delta_x_ned = body_x * math.cos(yaw) - body_y * math.sin(yaw)
            delta_y_ned = body_x * math.sin(yaw) + body_y * math.cos(yaw)

            # 새로운 목표 좌표 및 헤딩 계산 (센서 노이즈/바람 밀림 오차가 누적되지 않도록 기존 목표 위치 기준으로 증분)
            self.target_x = self.target_x + delta_x_ned
            self.target_y = self.target_y + delta_y_ned
            self.target_z = self.target_z - dz  # 고도 상승(+)은 NED에서 -Z 방향

            # d_yaw 회전 명령이 있을 때만 목표 헤딩을 갱신 (d_yaw=0이면 기존 목표 헤딩 완벽 유지)
            if abs(d_yaw_deg) > 1e-3:
                new_yaw = self.target_yaw - math.radians(d_yaw_deg)  # 스키마의 '반시계(+)'를 PX4 '반시계(-Yaw)'로 변환
                self.target_yaw = math.atan2(math.sin(new_yaw), math.cos(new_yaw))

            self.arrival_counter = 0
            self.flight_state = "MOVE"
            self.get_logger().info(
                f"🚶 [명령 수신] 이동 명령: 전후 {dx:+.2f}m, 좌우 {dy:+.2f}m, 고도변화 {dz:+.2f}m, 회전 {d_yaw_deg:+.1f}° ➔ "
                f"목표(NED): X={self.target_x:.2f}m, Y={self.target_y:.2f}m, Z={self.target_z:.2f}m"
            )

        elif func_name == "_goto_waypoint":
            # [내부 전용 명령] 저장된 절대 로컬 좌표(NED)로 직접 이동
            if self.flight_state not in ["HOVER", "MOVE"]:
                self.get_logger().warn(f"⚠️ [명령 거부] 기체가 비행 중이 아닙니다 (상태: {self.flight_state}).")
                self.mission_queue.clear()
                return

            self.target_x = float(args.get("x", self.target_x))
            self.target_y = float(args.get("y", self.target_y))
            self.target_z = float(args.get("z", self.target_z))
            self.target_yaw = float(args.get("yaw", self.target_yaw))

            self.is_recovering = True  # 복귀 이동 중 플래그 유지 (스택 재기록 차단)
            self.arrival_counter = 0
            self.flight_state = "MOVE"

            yaw_deg = math.degrees(self.target_yaw)
            self.get_logger().info(
                f"🧭 [경로 복귀 비행] 웨이포인트 이동: X={self.target_x:.2f}m, Y={self.target_y:.2f}m, "
                f"Z={self.target_z:.2f}m, Yaw={yaw_deg:.1f}° (남은 복귀 미션: {len(self.mission_queue)}개)"
            )

        elif func_name == "goto_history":
            # 🛡️ 안전 가드: 공중에 비행 중일 때만 복귀 명령 허용
            if self.flight_state not in ["HOVER", "MOVE"]:
                self.get_logger().warn(
                    f"⚠️ [명령 거부] 기체가 비행 중이 아닙니다 (현재 상태: {self.flight_state})."
                )
                self.mission_queue.clear()
                return

            recall_type = args.get("recall", "previous")
            self.get_logger().info(f"🧭 [명령 수신] 과거 위치 복귀 명령 (모드: {recall_type})")

            # 기록된 히스토리가 없는 경우 방어
            if not self.waypoint_history:
                self.get_logger().warn("⚠️ [복귀 불가] 기록된 과거 비행 위치가 없습니다.")
                self.execute_next_mission()
                return

            if recall_type == "previous":
                # 스택에 최소 2개 이상의 좌표가 있어야 '직전 지점'으로 복귀 가능
                # (1개만 있다면 그 자리가 바로 최초 출발지임)
                if len(self.waypoint_history) <= 1:
                    self.get_logger().warn(
                        "⚠️ [복귀 불가] 이미 최초 출발 위치에 있거나 직전 위치가 없습니다."
                    )
                    self.execute_next_mission()
                    return

                # 1. 현재 머물고 있는 최상단 지점 Pop (버림)
                current_top = self.waypoint_history.pop()
                self.get_logger().info(
                    f"🗑️ [스택 Pop] 현재 위치 폐기: X={current_top[0]:.2f}m, Y={current_top[1]:.2f}m"
                )

                # 2. 이제 최상단에 남은 지점이 바로 '직전 위치'
                target_point = self.waypoint_history[-1]
                self.target_x, self.target_y, self.target_z, self.target_yaw = target_point
                self.is_recovering = True  # 복귀 플래그 설정 (도착 시 재등록 방지)
                self.arrival_counter = 0
                self.flight_state = "MOVE"

                yaw_deg = math.degrees(self.target_yaw)
                self.get_logger().info(
                    f"🚀 [직전 위치 복귀 시작] 목표: X={self.target_x:.2f}m, Y={self.target_y:.2f}m, "
                    f"Z={self.target_z:.2f}m, Yaw={yaw_deg:.1f}° (남은 스택: {len(self.waypoint_history)})"
                )

            elif recall_type == "first":
                # 스택의 인덱스 0이 최초 출발점(Home)
                target_point = self.waypoint_history[0]
                self.target_x, self.target_y, self.target_z, self.target_yaw = target_point

                # 최초 출발지로 완전히 복귀하므로, 스택을 [원점]만 남기고 리셋
                self.waypoint_history = [target_point]
                self.is_recovering = True
                self.arrival_counter = 0
                self.flight_state = "MOVE"

                yaw_deg = math.degrees(self.target_yaw)
                self.get_logger().info(
                    f"🏠 [최초 출발지 복귀 시작] 목표: X={self.target_x:.2f}m, Y={self.target_y:.2f}m, "
                    f"Z={self.target_z:.2f}m, Yaw={yaw_deg:.1f}° (스택 원점 리셋 완료)"
                )

            else:
                self.get_logger().warn(f"⚠️ 알 수 없는 recall 인자입니다: {recall_type}")
                self.execute_next_mission()

        elif func_name == "reverse_plan":
            # 🛡️ 안전 가드: 공중에 비행 중일 때만 복귀 명령 허용
            if self.flight_state not in ["HOVER", "MOVE"]:
                self.get_logger().warn(
                    f"⚠️ [명령 거부] 기체가 비행 중이 아닙니다 (현재 상태: {self.flight_state})."
                )
                self.mission_queue.clear()
                return

            # 스택에 최소 2개 이상의 좌표가 있어야 거슬러 올라갈 경로가 있음
            if len(self.waypoint_history) <= 1:
                self.get_logger().warn("⚠️ [경로 복귀 불가] 이미 최초 출발 위치에 있거나 이동 경로가 없습니다.")
                self.execute_next_mission()
                return

            # 1. 현재 머물고 있는 지점을 제외하고, 지나온 이전 지점들을 시간 역순으로 추출
            backtrack_points = list(reversed(self.waypoint_history[:-1]))
            total_steps = len(backtrack_points)

            self.get_logger().info(
                f"🔄 [경로 역순 복귀 시작] 총 {total_steps}개의 이전 경유지를 순서대로 되짚어 출발지로 복귀합니다!"
            )

            # 2. 기존 대기열 비우기 (복귀 우선권 부여)
            self.mission_queue.clear()

            # 3. 역순 웨이포인트들을 미션 큐에 순차적으로 적재
            for i, pt in enumerate(backtrack_points):
                self.mission_queue.append({
                    "name": "_goto_waypoint",
                    "arguments": {
                        "x": pt[0],
                        "y": pt[1],
                        "z": pt[2],
                        "yaw": pt[3]
                    }
                })

            # 4. 🛬 [핵심] 최초 출발지 복귀 즉시 자동으로 안전 착륙(land)까지 수행하도록 큐 맨 끝에 land 추가!
            self.mission_queue.append({
                "name": "land",
                "arguments": {}
            })
            self.get_logger().info("🛬 [착륙 미션 연계] 최초 출발지 복귀 후 자동 착륙(land) 미션이 큐의 마지막에 등록되었습니다.")

            # 5. 스택을 원점(최초 출발지) 하나로 선제적 초기화 (복귀 완료 후 깨끗한 상태 유지)
            home_point = self.waypoint_history[0]
            self.waypoint_history = [home_point]
            self.is_recovering = True

            # 6. 첫 번째 역순 경유지로 즉시 이동 시작!
            self.execute_next_mission()

        elif func_name == "land":
            # 착륙 시작 시 비행 안전을 위해 대기열 비움
            self.mission_queue.clear()
            self.get_logger().info("🛬 [명령 수신] 착륙 명령을 시작합니다.")
            self.flight_state = "LAND"
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        else:
            self.get_logger().warn(f"⚠️ 현재 버전에서 지원하지 않는 명령입니다: {func_name}")
            # 지원하지 않는 명령이면 다음 미션 실행
            self.execute_next_mission()


    # =================================================================
    # 주기적 제어 루프 (10Hz)
    # =================================================================
    def timer_callback(self):
        # 1. 항상 Offboard Heartbeat 발행 (위치 제어 모드 유지)
        self.publish_offboard_heartbeat()

        # 2. 비행 상태 머신 처리
        if self.flight_state == "ARMING":
            # 지면 대기 중 기체 밀림을 방지하기 위해 실시간 지면 위치와 고도를 동기화
            self.target_x = self.current_x
            self.target_y = self.current_y
            self.home_z = self.current_z
            self.target_z = self.home_z - abs(self.target_altitude)

            # PX4 규칙: Offboard 진입 전 최소 10회 이상의 Setpoint/Heartbeat 필요
            self.publish_position_setpoint(self.target_x, self.target_y, self.target_z, self.target_yaw)
            self.heartbeat_counter += 1

            if self.heartbeat_counter == 15:
                self.get_logger().info("⚙️ Offboard 모드 전환 요청...")
                self.send_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    param1=1.0,  # Custom mode
                    param2=6.0   # PX4_CUSTOM_MAIN_MODE_OFFBOARD
                )

            if self.heartbeat_counter == 20:
                self.get_logger().info("⚡ 시동(ARM) 요청...")
                self.send_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=1.0   # 1.0 = ARM
                )

            # 시동 요청(20틱) 이후 실제 시동(is_armed) 체결 확인
            if self.heartbeat_counter > 20:
                if self.is_armed:
                    # 모터가 돌아 실제 땅에서 뜨는 바로 그 순간 최종 이륙 좌표 확정!
                    self.target_x = self.current_x
                    self.target_y = self.current_y
                    self.home_z = self.current_z
                    self.target_z = self.home_z - abs(self.target_altitude)
                    yaw_deg = math.degrees(self.target_yaw)
                    self.get_logger().info(
                        f"✅ [시동 체결 확인] 모터 시동 확인됨! 이륙(TAKEOFF)을 시작합니다. "
                        f"(이륙위치: X={self.target_x:.2f}m, Y={self.target_y:.2f}m, 목표고도={self.target_altitude:.2f}m, 목표헤딩={yaw_deg:.1f}°)"
                    )
                    self.flight_state = "TAKEOFF"
                elif self.heartbeat_counter >= 50:  # 3초(30틱) 동안 시동이 걸리지 않음 (Arming denied 등)
                    self.get_logger().error(
                        "❌ [시동 실패] PX4가 시동(ARM)을 거부하여 타임아웃되었습니다! "
                        "(Arming denied: 기체 센서/배터리 상태를 확인하세요)"
                    )
                    self.flight_state = "IDLE"
                    self.mission_queue.clear()

        elif self.flight_state == "TAKEOFF":
            # 비행 중 모터가 갑자기 꺼진 경우 안전 처리
            if not self.is_armed:
                self.get_logger().error("⚠️ [비행 중단] 이륙 중 모터 시동이 꺼졌습니다! (상태: IDLE 복귀)")
                self.flight_state = "IDLE"
                self.mission_queue.clear()
                return

            # 목표 고도로 지속적 Setpoint 발행
            self.publish_position_setpoint(self.target_x, self.target_y, self.target_z, self.target_yaw)

            # 고도 오차 0.15m 이내 & 수직 상승 속도 0.15m/s 이하로 감속/안정화되었을 때 카운트
            altitude_error = abs(self.current_z - self.target_z)
            vertical_speed = abs(self.current_vz)
            if altitude_error < 0.15 and vertical_speed < 0.15:
                self.arrival_counter += 1
                if self.arrival_counter >= 10:  # 1.0초 동안 고도 및 속도가 안정 유지됨
                    current_alt = (self.home_z - self.current_z) if self.home_z is not None else -self.current_z
                    # ⚠️ 주의: target_x, target_y는 밀려난 순간의 current_x로 덮어쓰지 않고 본래 목표 좌표를 그대로 유지
                    self.get_logger().info(
                        f"🎯 [이륙 완료] 목표 고도 안착 완료! (상대 고도: {current_alt:.2f}m, 수직속도: {vertical_speed:.2f}m/s)"
                    )
                    self.flight_state = "HOVER"
                    self.arrival_counter = 0

                    # 🧭 최초 이륙 안착 지점을 히스토리 스택의 원점(first)으로 등록
                    self.waypoint_history.clear()
                    self.waypoint_history.append((self.target_x, self.target_y, self.target_z, self.target_yaw))
                    self.get_logger().info(
                        f"📌 [히스토리 등록 - 최초 출발지] Waypoint #0 저장: "
                        f"X={self.target_x:.2f}m, Y={self.target_y:.2f}m, Z={self.target_z:.2f}m (스택 크기: {len(self.waypoint_history)})"
                    )

                    if self.mission_queue:
                        self.get_logger().info("📋 다음 대기 미션 실행...")
                        self.execute_next_mission()
                    else:
                        self.get_logger().info("➔ 대기열 비어있음: 호버링 유지")
            else:
                self.arrival_counter = 0

        elif self.flight_state == "MOVE":
            # 비행 중 모터가 갑자기 꺼진 경우 안전 처리
            if not self.is_armed:
                self.get_logger().error("⚠️ [비행 중단] 이동 중 모터 시동이 꺼졌습니다! (상태: IDLE 복귀)")
                self.flight_state = "IDLE"
                self.mission_queue.clear()
                return

            # 이동 목표 위치 및 헤딩 Setpoint 발행
            self.publish_position_setpoint(self.target_x, self.target_y, self.target_z, self.target_yaw)

            # 3차원 위치 오차 및 속도 계산
            dist_error = math.sqrt(
                (self.current_x - self.target_x) ** 2 +
                (self.current_y - self.target_y) ** 2 +
                (self.current_z - self.target_z) ** 2
            )
            speed = math.sqrt(
                self.current_vx ** 2 +
                self.current_vy ** 2 +
                self.current_vz ** 2
            )
            # 위치 오차 0.2m 이내 & 기체 잔류 속도 0.2m/s 이하로 감속/안착되었을 때 카운트
            if dist_error < 0.20 and speed < 0.20:
                self.arrival_counter += 1
                if self.arrival_counter >= 10:  # 1.0초 동안 목표 지점에 안정 정지
                    self.get_logger().info(f"🎯 [이동 완료] 목표 위치 안착 완료! (위치 오차: {dist_error:.2f}m, 잔류 속도: {speed:.2f}m/s)")
                    self.flight_state = "HOVER"
                    self.arrival_counter = 0

                    # 🧭 일반 이동 완료 시에만 히스토리 스택에 저장 (복귀 이동 시에는 핑퐁 방지를 위해 스킵)
                    if not self.is_recovering:
                        self.waypoint_history.append((self.target_x, self.target_y, self.target_z, self.target_yaw))
                        self.get_logger().info(
                            f"📌 [히스토리 등록] Waypoint #{len(self.waypoint_history)-1} 저장: "
                            f"X={self.target_x:.2f}m, Y={self.target_y:.2f}m, Z={self.target_z:.2f}m (스택 크기: {len(self.waypoint_history)})"
                        )
                    else:
                        # 대기열에 더 이상 복귀 미션이 없으면 최종 출발지 복귀 완료!
                        if not self.mission_queue:
                            self.is_recovering = False
                            self.get_logger().info(
                                f"🏁 [경로 복귀 완료] 최종 목표 지점에 안전하게 도착했습니다. (스택 원점 유지)"
                            )
                        else:
                            self.get_logger().info(
                                f"📍 [경유지 도달] 다음 역순 경유지로 계속 비행합니다... (남은 복귀 경로: {len(self.mission_queue)}개)"
                            )

                    if self.mission_queue:
                        self.get_logger().info("📋 다음 대기 미션 실행...")
                        self.execute_next_mission()
                    else:
                        self.get_logger().info("➔ 대기열 비어있음: 호버링 유지")
            else:
                self.arrival_counter = 0

        elif self.flight_state == "HOVER":
            # 비행 중 모터가 갑자기 꺼진 경우 안전 처리
            if not self.is_armed:
                self.get_logger().error("⚠️ [비행 중단] 호버링 중 모터 시동이 꺼졌습니다! (상태: IDLE 복귀)")
                self.flight_state = "IDLE"
                self.mission_queue.clear()
                return

            # 현재 목표 위치에 그대로 머무르도록 Setpoint 유지
            self.publish_position_setpoint(self.target_x, self.target_y, self.target_z, self.target_yaw)

        elif self.flight_state == "LAND":
            # 착륙이 완료되어 지면에 닿았는지 확인
            if self.is_landed and not self.is_armed:
                self.home_z = self.current_z  # 새로운 지면 높이를 home_z로 갱신 (재이륙 시 0m 기준)
                self.get_logger().info("🏁 착륙 및 모터 정지 완료! IDLE 상태로 복귀합니다.")
                self.flight_state = "IDLE"

    # =================================================================
    # PX4 저수준 메시지 발행 함수들
    # =================================================================
    def publish_offboard_heartbeat(self):
        """PX4에게 외부 제어기가 살아있음을 알리는 Heartbeat"""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0):
        """PX4 NED 로컬 좌표 목표 위치 발행 (단위: m, rad)"""
        msg = TrajectorySetpoint()
        # 소프트웨어 기준 좌표계 -> PX4 실제 물리 EKF 좌표계로 복원하여 전송
        raw_x = x + (self.origin_x if self.origin_x is not None else 0.0)
        raw_y = y + (self.origin_y if self.origin_y is not None else 0.0)
        msg.position = [raw_x, raw_y, z]
        msg.yaw = yaw
        msg.velocity = [float("nan"), float("nan"), float("nan")]
        msg.acceleration = [float("nan"), float("nan"), float("nan")]
        msg.yawspeed = float("nan")
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.setpoint_pub.publish(msg)

    def send_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
        """PX4 제어 명령(ARM, 모드 전환, 착륙 등) 발행"""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FlightController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 노드가 종료되었습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
