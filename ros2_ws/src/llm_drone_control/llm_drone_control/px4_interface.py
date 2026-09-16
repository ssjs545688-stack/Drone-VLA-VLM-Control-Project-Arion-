"""
px4_interface.py

PX4 저수준 명령 발행만 전담하는 클래스.
offboard_ctrl_example.py의 publish_* 함수들과 동일한 패턴을 그대로 따르되,
ROS2 Node 자체가 아니라 "Node 하나를 받아서 publisher만 만들어 쓰는" 형태로 분리했다.
미션 로직(mission_executor.py)이나 메인 노드가 이 클래스를 통해서만 PX4와 통신하게 해서,
"드론에 실제 명령을 내리는 지점"을 한 파일로 모아 감사(audit)하기 쉽게 만드는 것이 목적이다.
"""

from dataclasses import dataclass

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from .utils import clamp_angle


@dataclass
class VehicleIds:
    target_system: int = 1
    target_component: int = 1
    source_system: int = 1
    source_component: int = 1

    
def default_pub_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )


def default_sub_qos() -> QoSProfile:
    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
    )


class PX4Interface:
    """PX4 입력 토픽으로 명령을 변환해 발행하는 단일 경계.

    상위 MissionExecutor는 좌표와 미션 상태만 결정하고, 이 클래스가
    OffboardControlMode, TrajectorySetpoint, VehicleCommand 메시지를 만든다.
    """

    def __init__(self, node: Node, vehicle_ids: VehicleIds, qos: QoSProfile = None) -> None:
        self._node = node
        self._ids = vehicle_ids
        qos = qos or default_pub_qos()

        self.offboard_control_mode_pub = node.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos
        )
        self.trajectory_setpoint_pub = node.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos
        )
        self.vehicle_command_pub = node.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", qos
        )

    def _now_us(self) -> int:
        return int(self._node.get_clock().now().nanoseconds / 1000)

    # ------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------
    # 사람이 읽기 쉬운 비행 동작을 PX4 VehicleCommand로 변환한다.

    def arm(self) -> None:
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self._node.get_logger().info("⚡ ARM")

    def disarm(self) -> None:
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self._node.get_logger().info("💤 DISARM")

    def engage_offboard(self) -> None:
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self._node.get_logger().info("🎮 OFFBOARD")

    def land(self) -> None:
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self._node.get_logger().info("🛬 LAND")

    # ------------------------------------------------------------
    # Low-level publish
    # ------------------------------------------------------------
    # heartbeat와 setpoint는 timer 주기로 반복 발행하며 timestamp도 매번 갱신한다.

    def publish_offboard_heartbeat(self) -> None:
        msg = OffboardControlMode()
        msg.position = True
        msg.timestamp = self._now_us()
        self.offboard_control_mode_pub.publish(msg)

    def publish_setpoint(self, x: float, y: float, z: float, yaw: float) -> None:
        msg = TrajectorySetpoint()
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = float(clamp_angle(yaw))
        msg.timestamp = self._now_us()
        self.trajectory_setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command: int, **params: float) -> None:
        """명령 파라미터와 대상/발신 기체 ID를 채워 vehicle_command를 발행한다."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(params.get("param1", 0.0))
        msg.param2 = float(params.get("param2", 0.0))
        msg.param3 = float(params.get("param3", 0.0))
        msg.param4 = float(params.get("param4", 0.0))
        msg.target_system = self._ids.target_system
        msg.target_component = self._ids.target_component
        msg.source_system = self._ids.source_system
        msg.source_component = self._ids.source_component
        msg.from_external = True
        msg.timestamp = self._now_us()
        self.vehicle_command_pub.publish(msg)
