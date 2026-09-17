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

import json
import re


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
        self.current_z = 0.0
        self.is_armed = False
        self.nav_state = 0
        self.is_landed = True

        # 비행 제어 상태 머신: IDLE, ARMING, TAKEOFF, HOVER, LAND
        self.flight_state = "IDLE"
        self.target_altitude = 0.0  # 양수 (단위: m)
        self.ground_z = 0.0         # 이륙 시점 지면 고도 기준점
        self.target_z = 0.0         # 실제 계산된 목표 NED z 좌표
        self.heartbeat_counter = 0
        self.arrival_counter = 0    # 목표 고도 연속 도달 카운터

        # 10Hz 주기적 제어 루프 타이머
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("✅ [FlightController] 노드가 준비되었습니다. /llm_response 대기 중...")

    # =================================================================
    # 콜백 함수들
    # =================================================================
    def position_callback(self, msg: VehicleLocalPosition):
        """현재 드론의 로컬 NED 좌표 (z: 고도 반대 방향, 위로 갈수록 음수)"""
        self.current_z = msg.z

    def status_callback(self, msg: VehicleStatus):
        """기체 시동 상태 및 비행 모드 확인"""
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.nav_state = msg.nav_state

    def land_detected_callback(self, msg: VehicleLandDetected):
        """착륙 여부 감지"""
        self.is_landed = msg.landed

    def llm_response_callback(self, msg: String):
        """LLM이 발행한 응답 문자열 파싱"""
        text = msg.data.strip()
        self.get_logger().info(f"📩 수신된 LLM 응답: {text}")

        # 정규표현식으로 <tool_call> 태그 추출
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if not match:
            self.get_logger().warn("⚠️ 응답에서 <tool_call> 태그를 찾지 못했습니다.")
            return

        json_str = match.group(1)
        try:
            call_data = json.loads(json_str)
            func_name = call_data.get("name")
            args = call_data.get("arguments", {})
        except Exception as e:
            self.get_logger().error(f"❌ Tool Call JSON 파싱 실패: {e}")
            return

        # 명령에 따른 동작 분기
        if func_name == "takeoff":
            alt = float(args.get("altitude", 2.0))
            self.target_altitude = alt
            # 이륙 명령 시점의 지면 고도를 기록하고 지면 기준 상대 고도로 target_z 계산
            self.ground_z = self.current_z
            self.target_z = self.ground_z - alt
            self.heartbeat_counter = 0
            self.arrival_counter = 0
            self.flight_state = "ARMING"
            self.get_logger().info(
                f"🚀 [명령 수신] 이륙 명령: {alt}m (지면 Z: {self.ground_z:.2f}m ➔ 목표 Z: {self.target_z:.2f}m)"
            )

        elif func_name == "land":
            self.get_logger().info("🛬 [명령 수신] 착륙 명령을 시작합니다.")
            self.flight_state = "LAND"
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

        else:
            self.get_logger().warn(f"⚠️ 현재 버전에서 지원하지 않는 명령입니다: {func_name}")

    # =================================================================
    # 주기적 제어 루프 (10Hz)
    # =================================================================
    def timer_callback(self):
        # 1. 항상 Offboard Heartbeat 발행 (위치 제어 모드 유지)
        self.publish_offboard_heartbeat()

        # 2. 비행 상태 머신 처리
        if self.flight_state == "ARMING":
            # PX4 규칙: Offboard 진입 전 최소 10회 이상의 Setpoint/Heartbeat 필요
            self.publish_position_setpoint(0.0, 0.0, self.target_z)
            self.heartbeat_counter += 1

            if self.heartbeat_counter == 10:
                self.get_logger().info("⚙️ Offboard 모드 전환 요청...")
                self.send_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    param1=1.0,  # Custom mode
                    param2=6.0   # PX4_CUSTOM_MAIN_MODE_OFFBOARD
                )

            if self.heartbeat_counter == 15:
                self.get_logger().info("⚡ 시동(ARM) 요청...")
                self.send_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=1.0   # 1.0 = ARM
                )
                self.flight_state = "TAKEOFF"

        elif self.flight_state == "TAKEOFF":
            # 지면 기준 상대 목표 고도로 지속적 Setpoint 발행
            self.publish_position_setpoint(0.0, 0.0, self.target_z)

            # 고도 오차가 0.1m 이내로 들어오고, 1초(10회) 동안 안정적으로 유지될 때 호버링 전환
            altitude_error = abs(self.current_z - self.target_z)
            if altitude_error < 0.1:
                self.arrival_counter += 1
                if self.arrival_counter >= 10:
                    actual_agl = self.ground_z - self.current_z
                    self.get_logger().info(
                        f"🎯 목표 고도 도달 완료! (지면 기준 실상승: {actual_agl:.2f}m, NED Z: {self.current_z:.2f}m) ➔ 호버링 유지"
                    )
                    self.flight_state = "HOVER"
            else:
                self.arrival_counter = 0

        elif self.flight_state == "HOVER":
            # 현재 목표 위치에 그대로 머무르도록 Setpoint 유지
            self.publish_position_setpoint(0.0, 0.0, self.target_z)

        elif self.flight_state == "LAND":
            # 착륙이 완료되어 지면에 닿았는지 확인
            if self.is_landed and not self.is_armed:
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
        msg.position = [x, y, z]
        msg.yaw = yaw
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
