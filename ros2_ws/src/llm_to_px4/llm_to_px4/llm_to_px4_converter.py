#!/usr/bin/env python3
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import String
from guide_interfaces.srv import GuideLLM
from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleLocalPosition,
    VehicleStatus,
)


class LLMToPX4Converter(Node):
    """
    LLM의 텍스트 응답(TAKEOFF, LAND, MOVE, HOVER)을 수신하여
    PX4 커스텀 메시지(VehicleCommand, TrajectorySetpoint, OffboardControlMode)로
    변환 및 발행하는 노드입니다.
    """

    def __init__(self):
        super().__init__("llm_to_px4_converter")

        # PX4 uORB 토픽용 QoS 설정 (Best Effort, Volatile)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # -------------------------------------------------------------
        # 1. PX4 Publishers
        # -------------------------------------------------------------
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos_profile
        )
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos_profile
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos_profile
        )

        # -------------------------------------------------------------
        # 2. PX4 Subscribers (드론 현재 상태 및 위치 피드백)
        # -------------------------------------------------------------
        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            qos_profile,
        )
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.vehicle_status_callback,
            qos_profile,
        )

        # -------------------------------------------------------------
        # 3. LLM Response Subscriber
        # -------------------------------------------------------------
        self.llm_response_sub = self.create_subscription(
            String, "llm_response", self.llm_response_callback, 10
        )

        # -------------------------------------------------------------
        # 4. LLM Service Client & Command Service Server
        # -------------------------------------------------------------
        self.llm_client = self.create_client(GuideLLM, "llm")
        self.exec_srv = self.create_service(
            GuideLLM, "execute_llm_command", self.handle_execute_command_srv
        )

        # -------------------------------------------------------------
        # 5. 드론 상태 변수
        # -------------------------------------------------------------
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_heading = 0.0  # yaw in radians (-pi..pi)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        # 목표 위치 (NED 좌표계)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_yaw = 0.0

        # Offboard 제어 주기 타이머 (10Hz - 100ms)
        self.is_offboard_active = False
        self.offboard_setpoint_counter = 0
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info("LLM to PX4 Converter Node Initialized successfully.")

    def get_timestamp_us(self) -> int:
        """현재 시간을 마이크로초(us) 단위로 반환합니다."""
        return int(self.get_clock().now().nanoseconds / 1000)

    # -----------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------
    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        """PX4로부터 로컬 위치(NED) 및 헤딩각 수신"""
        self.current_x = msg.x
        self.current_y = msg.y
        self.current_z = msg.z
        self.current_heading = msg.heading

    def vehicle_status_callback(self, msg: VehicleStatus):
        """PX4로부터 비행 상태 및 아밍 상태 수신"""
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def llm_response_callback(self, msg: String):
        """llm_response 토픽으로 들어온 문자열 명령 처리"""
        raw_cmd = msg.data.strip()
        self.get_logger().info(f"[LLM Response 수신] '{raw_cmd}'")
        self.process_command(raw_cmd)

    def handle_execute_command_srv(self, request, response):
        """
        Service를 통해 직접 자연어 프롬프트를 받아 LLM 호출 후 PX4 명령으로 변환
        """
        prompt = request.prompt
        self.get_logger().info(f"[Execute Service 호출] 프롬프트: {prompt}")

        if not self.llm_client.wait_for_service(timeout_sec=2.0):
            response.response = "ERROR: LLM Service (/llm) is not available."
            self.get_logger().error(response.response)
            return response

        llm_req = GuideLLM.Request()
        llm_req.prompt = prompt

        # 동기 호출 대기
        future = self.llm_client.call_async(llm_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is not None:
            llm_output = future.result().response.strip()
            self.get_logger().info(f"[LLM 추론 결과] {llm_output}")
            self.process_command(llm_output)
            response.response = f"SUCCESS: Executed '{llm_output}'"
        else:
            response.response = "ERROR: Failed to call LLM service."
            self.get_logger().error(response.response)

        return response

    # -----------------------------------------------------------------
    # Command Processing & Conversion to PX4 Messages
    # -----------------------------------------------------------------
    def process_command(self, cmd_text: str):
        """
        LLM 출력 포맷 분석 및 PX4 메시지 변환
        - TAKEOFF <고도>
        - LAND
        - MOVE <x> <y> <z>
        - HOVER
        """
        # 혹시 남아있을 수 있는 불필요한 따옴표나 공백 정리
        cmd_text = cmd_text.replace('"', '').replace("'", "").strip()
        tokens = cmd_text.split()

        if not tokens:
            self.get_logger().warning("빈 명령어가 수신되었습니다.")
            return

        action = tokens[0].upper()

        if action == "TAKEOFF":
            if len(tokens) >= 2:
                try:
                    altitude = float(tokens[1])
                except ValueError:
                    altitude = 2.5
            else:
                altitude = 2.5
            self.execute_takeoff(altitude)

        elif action == "LAND":
            self.execute_land()

        elif action == "MOVE":
            if len(tokens) >= 4:
                try:
                    dx = float(tokens[1])
                    dy = float(tokens[2])
                    dz = float(tokens[3])
                except ValueError:
                    self.get_logger().error(f"MOVE 좌표 파싱 실패: {cmd_text}")
                    return
                self.execute_move(dx, dy, dz)
            else:
                self.get_logger().error(f"MOVE 명령 인자 부족: {cmd_text}")

        elif action == "HOVER":
            self.execute_hover()

        else:
            self.get_logger().warning(f"알 수 없는 명령 '{action}', HOVER로 대체합니다.")
            self.execute_hover()

    # -----------------------------------------------------------------
    # PX4 Command Executors
    # -----------------------------------------------------------------
    def execute_takeoff(self, altitude: float):
        """
        TAKEOFF <고도>:
        상대 고도(m) 만큼 이륙 명령 전송
        """
        self.get_logger().info(f"[PX4 변환] TAKEOFF 고도: {altitude}m")
        # 1. Arming 명령 전송
        self.arm()

        # 2. VehicleCommand (VEHICLE_CMD_NAV_TAKEOFF) 발행
        # param7: 이륙 고도 (m)
        self.publish_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF,
            param7=float(altitude),
        )

        # Offboard 모드로 전환하여 특정 위치를 유지할 수도 있도록 세팅
        self.target_x = self.current_x
        self.target_y = self.current_y
        self.target_z = self.current_z - altitude  # NED z축은 아래 방향이므로 -altitude
        self.target_yaw = self.current_heading
        self.is_offboard_active = True

    def execute_land(self):
        """
        LAND:
        착륙 모드 명령 전송
        """
        self.get_logger().info("[PX4 변환] LAND 착륙 명령 전송")
        self.is_offboard_active = False
        self.publish_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_NAV_LAND
        )

    def execute_move(self, dx: float, dy: float, dz: float):
        """
        MOVE <x> <y> <z>:
        현재 드론 위치/헤딩 기준 상대 이동
        +x: 전진, -x: 후진
        +y: 우측, -y: 좌측
        +z: 상승, -z: 하강
        """
        self.get_logger().info(f"[PX4 변환] MOVE 상대 이동: dx={dx}m, dy={dy}m, dz={dz}m")

        # 드론 현재 헤딩(yaw)을 반영하여 Body 프레임 -> NED 프레임 변환
        psi = self.current_heading
        dx_ned = dx * math.cos(psi) - dy * math.sin(psi)
        dy_ned = dx * math.sin(psi) + dy * math.cos(psi)
        dz_ned = -dz  # NED z축은 아래가 양수이므로 상승(+z)은 -dz

        # 목표 위치 갱신
        self.target_x = self.current_x + dx_ned
        self.target_y = self.current_y + dy_ned
        self.target_z = self.current_z + dz_ned
        self.target_yaw = self.current_heading

        self.get_logger().info(
            f"목표 NED 좌표: North={self.target_x:.2f}, East={self.target_y:.2f}, Down={self.target_z:.2f}"
        )

        # Offboard 모드 활성화 및 스트리밍 시작
        self.is_offboard_active = True
        self.set_offboard_mode()
        self.arm()

    def execute_hover(self):
        """
        HOVER:
        현재 위치에서 정지 비행
        """
        self.get_logger().info("[PX4 변환] HOVER 현재 위치 유지")
        self.target_x = self.current_x
        self.target_y = self.current_y
        self.target_z = self.current_z
        self.target_yaw = self.current_heading

        self.is_offboard_active = True
        self.set_offboard_mode()

    # -----------------------------------------------------------------
    # PX4 Message Helpers
    # -----------------------------------------------------------------
    def arm(self):
        """드론 시동 (Arming)"""
        self.publish_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,
        )
        self.get_logger().info("PX4 Arming 명령 전송 완료")

    def disarm(self):
        """드론 시동 끄기 (Disarming)"""
        self.publish_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,
        )
        self.get_logger().info("PX4 Disarming 명령 전송 완료")

    def set_offboard_mode(self):
        """Offboard 모드 진입 명령"""
        self.publish_vehicle_command(
            command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,  # custom mode
            param2=6.0,  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        )
        self.get_logger().info("PX4 Offboard Mode 전환 명령 전송 완료")

    def publish_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ):
        """VehicleCommand 메시지 생성 및 발행"""
        msg = VehicleCommand()
        msg.timestamp = self.get_timestamp_us()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.confirmation = 0

        self.vehicle_command_pub.publish(msg)

    def publish_offboard_control_mode(self):
        """OffboardControlMode 하트비트 메시지 발행"""
        msg = OffboardControlMode()
        msg.timestamp = self.get_timestamp_us()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False

        self.offboard_control_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        """TrajectorySetpoint 위치 제어 메시지 발행"""
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_timestamp_us()
        msg.position = [
            float(self.target_x),
            float(self.target_y),
            float(self.target_z),
        ]
        msg.velocity = [float("nan"), float("nan"), float("nan")]
        msg.acceleration = [float("nan"), float("nan"), float("nan")]
        msg.jerk = [float("nan"), float("nan"), float("nan")]
        msg.yaw = float(self.target_yaw)
        msg.yawspeed = float("nan")

        self.trajectory_setpoint_pub.publish(msg)

    def timer_callback(self):
        """
        10Hz 주기 타이머 콜백:
        PX4 Offboard 모드 유지를 위해 지속적으로 OffboardControlMode와
        TrajectorySetpoint를 스트리밍합니다.
        """
        if self.is_offboard_active:
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint()

            # Offboard 진입 전 사전 setpoint 전송 (PX4 요구사항: 10개 이상 발행 후 모드 전환)
            if self.offboard_setpoint_counter < 11:
                self.offboard_setpoint_counter += 1
                if self.offboard_setpoint_counter == 10:
                    self.set_offboard_mode()
                    self.arm()


def main(args=None):
    rclpy.init(args=args)
    node = LLMToPX4Converter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
