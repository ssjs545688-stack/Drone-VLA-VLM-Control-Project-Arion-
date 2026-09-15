#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy,DurabilityPolicy
from std_msgs.msg import String
from px4_msgs.msg import OffboardControlMode,TrajectorySetpoint,VehicleCommand,VehicleLocalPosition,VehicleStatus


class LLMToPX4Controller(Node):
    def __init__(self):
        super().__init__("llm_px4_controller")

        qos=QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboard_pub=self.create_publisher(OffboardControlMode,"/fmu/in/offboard_control_mode",qos)
        self.setpoint_pub=self.create_publisher(TrajectorySetpoint,"/fmu/in/trajectory_setpoint",qos)
        self.command_pub=self.create_publisher(VehicleCommand,"/fmu/in/vehicle_command",qos)

        self.pos_sub=self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.position_callback,
            qos
        )

        self.status_sub=self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.status_callback,
            qos
        )

        self.llm_sub=self.create_subscription(
            String,
            "/px4_command",
            self.command_callback,
            10
        )

        self.current_x=0.0
        self.current_y=0.0
        self.current_z=0.0
        self.current_yaw=0.0

        self.target_x=0.0
        self.target_y=0.0
        self.target_z=0.0
        self.target_yaw=0.0

        self.nav_state=VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state=VehicleStatus.ARMING_STATE_DISARMED

        self.offboard_active=False
        self.offboard_setpoint_counter=0

        self.timer=self.create_timer(0.1,self.timer_callback)

        self.get_logger().info("LLM → PX4 Controller started")

    # --------------------------------------------------
    # PX4 callbacks
    # --------------------------------------------------

    def position_callback(self,msg):
        self.current_x=msg.x
        self.current_y=msg.y
        self.current_z=msg.z
        self.current_yaw=msg.heading

    def status_callback(self,msg):
        self.nav_state=msg.nav_state
        self.arming_state=msg.arming_state

    # --------------------------------------------------
    # LLM command
    # --------------------------------------------------

    def command_callback(self,msg):
        command=msg.data.strip()
        self.get_logger().info(f"Validated command: {command}")
        self.process_command(command)

    def process_command(self,command):
        command=command.replace('"',"").replace("'","").strip()
        tokens=command.split()

        if not tokens:
            return

        action=tokens[0].upper()

        if action=="TAKEOFF":
            altitude=float(tokens[1]) if len(tokens)>1 else 2.0
            self.takeoff(altitude)

        elif action=="LAND":
            self.land()

        elif action=="MOVE":
            if len(tokens)<4:
                self.get_logger().error("MOVE requires x y z")
                return

            try:
                x=float(tokens[1])
                y=float(tokens[2])
                z=float(tokens[3])
            except ValueError:
                self.get_logger().error("Invalid MOVE arguments")
                return

            self.move(x,y,z)

        elif action=="HOVER":
            self.hover()

        else:
            self.get_logger().warning(f"Unknown command: {action}")

    # --------------------------------------------------
    # Command functions
    # --------------------------------------------------

    def takeoff(self,altitude):
        self.get_logger().info(f"TAKEOFF {altitude}m")

        self.target_x=self.current_x
        self.target_y=self.current_y

        # PX4 local position은 NED이므로 상승은 음수
        self.target_z=self.current_z-abs(altitude)

        self.target_yaw=self.current_yaw

        self.start_offboard()

    def move(self,dx,dy,dz):
        self.get_logger().info(
            f"MOVE dx={dx}, dy={dy}, dz={dz}"
        )

        # Body frame → NED frame
        yaw=self.current_yaw

        dx_ned=dx*math.cos(yaw)-dy*math.sin(yaw)
        dy_ned=dx*math.sin(yaw)+dy*math.cos(yaw)

        # 사용자 기준 +Z = 상승
        dz_ned=-dz

        self.target_x=self.current_x+dx_ned
        self.target_y=self.current_y+dy_ned
        self.target_z=self.current_z+dz_ned
        self.target_yaw=self.current_yaw

        self.get_logger().info(
            f"Target NED: "
            f"x={self.target_x:.2f}, "
            f"y={self.target_y:.2f}, "
            f"z={self.target_z:.2f}"
        )

        self.start_offboard()

    def hover(self):
        self.get_logger().info("HOVER")

        self.target_x=self.current_x
        self.target_y=self.current_y
        self.target_z=self.current_z
        self.target_yaw=self.current_yaw

        self.start_offboard()

    def land(self):
        self.get_logger().info("LAND")

        self.offboard_active=False
        self.offboard_setpoint_counter=0

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_NAV_LAND
        )

    # --------------------------------------------------
    # Offboard
    # --------------------------------------------------

    def start_offboard(self):
        self.offboard_active=True
        self.offboard_setpoint_counter=0

    def arm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0
        )
        self.get_logger().info("ARM command sent")

    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=0.0
        )
        self.get_logger().info("DISARM command sent")

    def engage_offboard(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0
        )
        self.get_logger().info("OFFBOARD command sent")

    # --------------------------------------------------
    # PX4 publishers
    # --------------------------------------------------

    def publish_offboard_control_mode(self):
        msg=OffboardControlMode()
        msg.timestamp=self.timestamp()

        msg.position=True
        msg.velocity=False
        msg.acceleration=False
        msg.attitude=False
        msg.body_rate=False
        msg.thrust_and_torque=False
        msg.direct_actuator=False

        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self):
        msg=TrajectorySetpoint()
        msg.timestamp=self.timestamp()

        msg.position=[
            float(self.target_x),
            float(self.target_y),
            float(self.target_z)
        ]

        msg.velocity=[float("nan")]*3
        msg.acceleration=[float("nan")]*3
        msg.jerk=[float("nan")]*3

        msg.yaw=float(self.target_yaw)
        msg.yawspeed=float("nan")

        self.setpoint_pub.publish(msg)

    def publish_vehicle_command(self,command,**params):
        msg=VehicleCommand()

        msg.timestamp=self.timestamp()
        msg.command=command

        msg.param1=params.get("param1",0.0)
        msg.param2=params.get("param2",0.0)
        msg.param3=params.get("param3",0.0)
        msg.param4=params.get("param4",0.0)
        msg.param5=params.get("param5",0.0)
        msg.param6=params.get("param6",0.0)
        msg.param7=params.get("param7",0.0)

        msg.target_system=1
        msg.target_component=1
        msg.source_system=1
        msg.source_component=1
        msg.from_external=True

        self.command_pub.publish(msg)

    # --------------------------------------------------
    # Timer
    # --------------------------------------------------

    def timer_callback(self):
        if not self.offboard_active:
            return

        # Offboard heartbeat
        self.publish_offboard_control_mode()

        # 목표 위치 계속 발행
        self.publish_trajectory_setpoint()

        # PX4가 요구하는 사전 setpoint
        if self.offboard_setpoint_counter<11:
            self.offboard_setpoint_counter+=1

            if self.offboard_setpoint_counter==10:
                self.engage_offboard()
                self.arm()

    def timestamp(self):
        return int(self.get_clock().now().nanoseconds/1000)


def main(args=None):
    rclpy.init(args=args)

    node=LLMToPX4Controller()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__=="__main__":
    main()