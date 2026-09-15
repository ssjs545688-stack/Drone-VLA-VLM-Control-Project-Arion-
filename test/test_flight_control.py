#!/usr/bin/env python3
"""
test_flight_control.py
- 로컬 Qwen3-0.6B Function Calling 추론
- 자연어 명령("드론 고도 2m로 띄워줘") -> takeoff(altitude=2.0) 추출
- ROS 2 (px4_msgs) 노드를 통해 PX4 Offboard 제어로 2m 이륙 수행
"""

import sys
import time
import json
import re
from pathlib import Path

# AI / PyTorch
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ROS 2 & PX4 Msgs
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition


# ==============================================================================
# 1. LLM 추론기 (Qwen3-0.6B + Tool Calling)
# ==============================================================================
class LLMCommandParser:
    def __init__(self):
        model_path = Path.home() / "Drone-VLA-VLM-Control-Project-Arion-" / "models" / "Qwen3-0.6B"
        print("🧠 [LLM] 로컬 Qwen3-0.6B 모델 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype="auto", device_map="auto"
        )
        print("✅ [LLM] 모델 로딩 완료!")

        # 함수 스키마 정의
        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "takeoff",
                    "description": "이륙: 드론을 지정한 목표 고도로 수직 이륙시킵니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "altitude": {
                                "type": "number",
                                "description": "이륙 목표 고도 (단위: 미터)"
                            }
                        },
                        "required": ["altitude"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "land",
                    "description": "착륙: 드론을 현재 위치에서 지면으로 안전하게 착륙시킵니다.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

    def parse(self, user_prompt: str):
        """자연어를 분석하여 호출할 함수명과 인자 딕셔너리 반환"""
        print(f"\n🗣️ [사용자 명령]: \"{user_prompt}\"")
        messages = [{"role": "user", "content": user_prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools_schema,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=128, do_sample=False)

        raw_output = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        print(f"🤖 [LLM 출력 원문]:\n{raw_output.strip()}")

        # <tool_call> 태그 파싱
        match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', raw_output, re.DOTALL)
        if match:
            call_json = json.loads(match.group(1))
            return call_json.get("name"), call_json.get("arguments", {})
        
        # 태그가 없는 순수 JSON 형태일 경우 대비
        try:
            call_json = json.loads(raw_output.strip())
            return call_json.get("name"), call_json.get("arguments", {})
        except Exception:
            return None, {}


# ==============================================================================
# 2. PX4 ROS 2 Offboard 제어기 노드
# ==============================================================================
class PX4TakeoffController(Node):
    def __init__(self, target_altitude: float = 2.0):
        super().__init__('px4_takeoff_controller')
        self.target_altitude = float(target_altitude)
        # PX4 NED 좌표계: 고도 상승은 -Z
        self.target_z = -abs(self.target_altitude)
        self.callback_group = ReentrantCallbackGroup()

        # QoS 설정 (PX4 uORB 규격 맞춤)
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1
        )
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=5
        )

        # Publishers
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_pub)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_pub)
        self.vehicle_cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_pub)

        # Subscribers
        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self.pos_callback, qos_sub, callback_group=self.callback_group
        )

        # 내부 상태
        self.current_z = 0.0
        self.position_received = False
        self.heartbeat_counter = 0
        self.is_armed = False
        self.is_offboard = False
        self.takeoff_complete = False

        # 10Hz (0.1초 간격) 제어 루프 타이머 가동
        self.timer = self.create_timer(
            0.1, self.control_loop, callback_group=self.callback_group
        )
        self.get_logger().info(f"🎯 목표 이륙 고도: {self.target_altitude}m (NED Z: {self.target_z}m)")

    def pos_callback(self, msg: VehicleLocalPosition):
        self.current_z = msg.z
        if not self.position_received:
            self.position_received = True
            self.get_logger().info(f"📡 [PX4 위치 수신 확인]: 현재 Z={msg.z:.2f}m (고도 {-msg.z:.2f}m)")

    def control_loop(self):
        # 1. Heartbeat 지속 발행 (PX4 Offboard 유지 필수)
        self.publish_offboard_heartbeat()

        # 2. 목표 위치 Setpoint 지속 발행 (현재 X=0, Y=0, Z=-2.0 유지)
        self.publish_setpoint(0.0, 0.0, self.target_z, 0.0)

        # 3. 안전을 위해 1초간(10회) Setpoint를 먼저 보낸 뒤 순차 실행
        if self.heartbeat_counter < 10:
            self.heartbeat_counter += 1
            return

        # 4. [중요] 1단계: 모터 시동 (ARM) 먼저 실행! (Disarmed 상태에서는 Offboard 진입이 거부됨)
        if not self.is_armed:
            self.get_logger().info("⚡ [PX4] 1단계: 모터 시동 (ARM) 실행...")
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
            self.is_armed = True
            return

        # 5. [중요] 2단계: 시동 후 Offboard 모드로 전환!
        if not self.is_offboard:
            self.get_logger().info("🎮 [PX4] 2단계: Offboard 모드 전환 실행...")
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
            self.is_offboard = True
            return

        # 6. 현재 고도 모니터링 및 2m 도달 검사
        curr_alt = -self.current_z
        self.get_logger().info(f"🚁 현재 실시간 고도: {curr_alt:.2f} m / 목표: {self.target_altitude:.2f} m", throttle_duration_sec=1.0)

        if abs(curr_alt - self.target_altitude) < 0.2 and not self.takeoff_complete:
            self.takeoff_complete = True
            self.get_logger().info(f"🎉 드론이 목표 고도 {self.target_altitude}m에 성공적으로 도달하여 호버링(Hovering) 중입니다!")
            self.get_logger().info("ℹ️ QGroundControl 화면에서 고도와 Offboard 모드 상태를 확인하세요.")

    def publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_pub.publish(msg)

    def publish_setpoint(self, x: float, y: float, z: float, yaw: float):
        sp = TrajectorySetpoint()
        sp.position = [float(x), float(y), float(z)]
        sp.yaw = float(yaw)
        sp.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(sp)

    def send_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        cmd = VehicleCommand()
        cmd.command = command
        cmd.param1 = float(param1)
        cmd.param2 = float(param2)
        cmd.param7 = float(param7)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_cmd_pub.publish(cmd)


# ==============================================================================
# 3. 메인 실행 함수
# ==============================================================================
def main():
    # 1. 자연어 명령 파싱
    parser = LLMCommandParser()
    user_prompt = "드론 고도 2m로 띄워줘"
    func_name, args = parser.parse(user_prompt)

    print(f"\n📋 [파싱 결과]: 함수 = '{func_name}', 인자 = {args}")

    if func_name != "takeoff":
        print(f"❌ 지원되지 않는 명령입니다: {func_name}")
        return

    target_alt = args.get("altitude", 2.0)
    print(f"🚀 [실행 시작]: 목표 고도 {target_alt}m 이륙 프로세스를 시작합니다.\n")

    # 2. ROS 2 멀티스레드 노드 기동 및 비행 시작
    rclpy.init()
    node = PX4TakeoffController(target_altitude=target_alt)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 비행 제어가 중단되었습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
