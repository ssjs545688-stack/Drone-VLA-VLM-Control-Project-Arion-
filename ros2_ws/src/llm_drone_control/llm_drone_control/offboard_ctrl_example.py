#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode and receiving camera feed."""

    def __init__(self) -> None:
        """PX4 offboard 이륙·착륙 예제에 필요한 토픽과 타이머를 초기화한다."""
        super().__init__('offboard_control_takeoff_and_land')

        # 이 파일은 분리된 모듈 구조를 이해하기 위한 독립 예제다.
        # timer가 heartbeat와 초기 setpoint를 보내고 OFFBOARD/ARM -> 유지 -> LAND를 실행한다.

        # ── 하드코딩 제거: 토픽/기체 ID/이륙고도/주기를 ROS2 파라미터로 선언 ──
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('takeoff_height', -1.5)
        self.declare_parameter('timer_period', 0.1)
        self.declare_parameter('target_system', 1)
        self.declare_parameter('target_component', 1)
        self.declare_parameter('source_system', 1)
        self.declare_parameter('source_component', 1)
        self.declare_parameter('offboard_engage_setpoints', 10)

        camera_topic = self.get_parameter('camera_topic').value
        self.takeoff_height = self.get_parameter('takeoff_height').value
        timer_period = self.get_parameter('timer_period').value
        self._target_system = self.get_parameter('target_system').value
        self._target_component = self.get_parameter('target_component').value
        self._source_system = self.get_parameter('source_system').value
        self._source_component = self.get_parameter('source_component').value
        self._offboard_engage_setpoints = self.get_parameter('offboard_engage_setpoints').value

        # PX4 예제와 같은 QoS로 입력/출력 메시지의 전달 정책을 맞춘다.
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        self.cv_bridge = CvBridge()
        self.image_subscriber = self.create_subscription(
            Image,
            camera_topic,  # 더 이상 코드에 고정되지 않고 파라미터로 지정
            self.image_callback,
            10)

        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()

        self.timer = self.create_timer(timer_period, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """PX4가 보낸 최신 로컬 위치를 저장한다."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """PX4의 현재 비행 상태를 저장한다."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """기체 시동 명령을 발행한다."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """기체 시동 해제 명령을 발행한다."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """PX4를 offboard 제어 모드로 전환한다."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """PX4에 착륙 모드 전환 명령을 발행한다."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """offboard 모드 유지를 위해 위치 제어 heartbeat를 주기적으로 보낸다."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float):
        """PX4에 목표 NED 위치 setpoint를 발행한다."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = 0.
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """대상 기체 식별 정보와 함께 PX4 vehicle command를 발행한다."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        # ── 버그 수정: target/source 필드가 누락되어 있었음(기존엔 브로드캐스트 취급될 위험) ──
        msg.target_system = self._target_system
        msg.target_component = self._target_component
        msg.source_system = self._source_system
        msg.source_component = self._source_component
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def timer_callback(self) -> None:
        """초기 setpoint를 충분히 보낸 뒤 offboard 이륙과 착륙을 순서대로 수행한다."""
        # heartbeat를 먼저 보내고 초기 setpoint 횟수가 충족되면 모드 전환/ARM을 요청한다.
        self.publish_offboard_control_heartbeat_signal()

        if self.offboard_setpoint_counter == self._offboard_engage_setpoints:
            self.engage_offboard_mode()
            self.arm()

        if self.vehicle_local_position.z > self.takeoff_height and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.publish_position_setpoint(0.0, 0.0, self.takeoff_height)

        elif self.vehicle_local_position.z <= self.takeoff_height:
            self.land()
            exit(0)

        if self.offboard_setpoint_counter < self._offboard_engage_setpoints + 1:
            self.offboard_setpoint_counter += 1

    def image_callback(self, msg):
        """ROS 이미지 메시지를 OpenCV 영상으로 변환해 화면에 표시한다."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow('Drone Camera View', cv_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")


def main(args=None) -> None:
    """ROS 2 노드를 실행하고 종료 시 카메라 창과 노드를 정리한다."""
    rclpy.init(args=args)
    offboard_control = OffboardControl()

    print('Starting offboard control & camera node...')
    try:
        rclpy.spin(offboard_control)
    except KeyboardInterrupt:
        pass
    finally:
        offboard_control.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
