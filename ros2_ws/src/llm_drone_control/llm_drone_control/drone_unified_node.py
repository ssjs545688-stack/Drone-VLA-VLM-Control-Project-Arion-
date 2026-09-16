#!/usr/bin/env python3
"""
drone_unified_node.py

PX4 ROS 2 + local Qwen3 Tool Calling 통합 드론 제어 노드 (조립 지점).

이 파일은 ex.py(1372줄, 단일 파일)를 역할별로 나눈 뒤 마지막에 조립만 하는
얇은 진입점이다. 각 조각의 책임은 다음과 같다.

    schema.py           Tool 스키마 / 시스템 프롬프트 정의
    utils.py            각도 정규화, 안전 변환, JSON 파싱 유틸 (순수 함수)
    qwen_engine.py       로컬 Qwen3 로딩 / 추론 / Tool Call 파싱·검증
    px4_interface.py    PX4 저수준 명령 발행 (ARM/OFFBOARD/LAND/setpoint)
    mission_executor.py 미션 큐 / 도착 판정 / history / reverse_plan 상태 머신
    drone_unified_node.py(현재 파일)  위 조각들을 ROS2 Node로 조립 + 진입점

핵심 흐름 (기존과 동일)
    사용자 자연어
        -> Qwen3-0.6B (local, qwen_engine.QwenToolCaller)
        -> <tool_call>{"name": ..., "arguments": {...}}</tool_call>
        -> mission_executor.MissionExecutor (mission queue)
        -> px4_interface.PX4Interface -> PX4 Offboard / VehicleCommand

주의
    - Qwen은 "무엇을 할지"만 결정한다.
    - 실제 좌표 계산, History, 역경로 생성, PX4 제어는 ROS 2 노드(이 파일 + 분리된 모듈)가 담당한다.
"""

import threading

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image

from px4_msgs.msg import (
    VehicleLandDetected,
    VehicleLocalPosition,
    VehicleStatus,
)

from .mission_executor import ArrivalThresholds, MissionExecutor
from .px4_interface import PX4Interface, VehicleIds, default_sub_qos
from .qwen_engine import QwenToolCaller, ValidationLimits
from .utils import safe_float


class DroneUnifiedNode(Node):
    # 수동 자연어 입력을 거치지 않고 정해진 순서의 비행을 시험할 때 사용하는 시나리오다.
    def start_basic_flight_test(self):

        test_missions = [
            # 이륙
            {
                "arm": True,
                "offboard": True,
                "x": 0.0,
                "y": 0.0,
                "z": -2.0,
                "yaw": 0.0
            },

            # 전진 1m
            {
                "x": 1.0,
                "y": 0.0,
                "z": -2.0,
                "yaw": 0.0
            },

            # 오른쪽 1m
            {
                "x": 1.0,
                "y": 1.0,
                "z": -2.0,
                "yaw": 0.0
            },

            # 제자리 유지
            {
                "x": 1.0,
                "y": 1.0,
                "z": -2.0,
                "yaw": 0.0
            },

            # 착륙
            {
                "land": True
            }
        ]

        self.setpoint_queue.clear()

        for mission in test_missions:
            self.setpoint_queue.append(mission)

        self.in_landing = False
        self.has_target = False

        self.get_logger().info(
            "🧪 기본 비행 테스트 시작"
        )

        self._consume_next()

    def __init__(self) -> None:
        super().__init__("drone_unified")
        self.callback_group = ReentrantCallbackGroup()
        self.state_lock = threading.RLock()

        # ROS 콜백, 타이머, 사용자 입력 스레드가 동시에 상태를 읽고 쓰므로
        # 비행 상태와 미션 큐를 하나의 재진입 락으로 보호한다.

        # --------------------------------------------------------
        # ROS parameters: 토픽, 모델, 안전 한계, 도착 판정 설정을 외부에서 조정한다.
        # --------------------------------------------------------
        self.declare_parameter("camera_topic", "/camera")
        self.declare_parameter(
            "model_path",
            "~/Drone-VLA-VLM-Control-Project-Arion-/models/Qwen3-0.6B",
        )
        self.declare_parameter("max_new_tokens", 256)
        self.declare_parameter("target_system", 1)
        self.declare_parameter("target_component", 1)
        self.declare_parameter("source_system", 1)
        self.declare_parameter("source_component", 1)
        self.declare_parameter("arrival_threshold", 0.30)
        self.declare_parameter("arrival_confirm_count", 10)
        self.declare_parameter("yaw_arrival_threshold", 0.10)
        self.declare_parameter("timer_period", 0.10)
        self.declare_parameter("max_takeoff_alt", 10.0)
        self.declare_parameter("max_horizontal_move", 20.0)
        self.declare_parameter("max_vertical_move", 10.0)
        self.declare_parameter("max_yaw_change", 360.0)
        self.declare_parameter("history_size", 100)

        camera_topic = str(self.get_parameter("camera_topic").value)
        model_path = str(self.get_parameter("model_path").value)
        max_new_tokens = int(self.get_parameter("max_new_tokens").value)
        timer_period = float(self.get_parameter("timer_period").value)
        history_size = int(self.get_parameter("history_size").value)

        vehicle_ids = VehicleIds(
            target_system=int(self.get_parameter("target_system").value),
            target_component=int(self.get_parameter("target_component").value),
            source_system=int(self.get_parameter("source_system").value),
            source_component=int(self.get_parameter("source_component").value),
        )
        thresholds = ArrivalThresholds(
            distance_m=float(self.get_parameter("arrival_threshold").value),
            confirm_count=int(self.get_parameter("arrival_confirm_count").value),
            yaw_rad=float(self.get_parameter("yaw_arrival_threshold").value),
        )
        limits = ValidationLimits(
            max_takeoff_alt=float(self.get_parameter("max_takeoff_alt").value),
            max_horizontal_move=float(self.get_parameter("max_horizontal_move").value),
            max_vertical_move=float(self.get_parameter("max_vertical_move").value),
            max_yaw_change=float(self.get_parameter("max_yaw_change").value),
        )

        # --------------------------------------------------------
        # PX4 subscribers: 위치/상태/착륙 감지는 이 노드가 받고,
        # 명령 publisher는 PX4Interface가 소유한다.
        # --------------------------------------------------------
        qos_sub = default_sub_qos()
        self.local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.land_detected = VehicleLandDetected()
        self.position_received = False

        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position",
            self.position_callback, qos_sub, callback_group=self.callback_group,
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status",
            self.status_callback, qos_sub, callback_group=self.callback_group,
        )
        self.create_subscription(
            VehicleLandDetected, "/fmu/out/vehicle_land_detected",
            self.land_detected_callback, qos_sub, callback_group=self.callback_group,
        )

        # --------------------------------------------------------
        # Camera: ROS Image를 OpenCV 화면으로 변환해 현재 미션 정보를 표시한다.
        # --------------------------------------------------------
        self.cv_bridge = CvBridge()
        self.create_subscription(
            Image, camera_topic, self.image_callback, 10,
            callback_group=self.callback_group,
        )

        # --------------------------------------------------------
        # 조립: 자연어 해석(Qwen), 미션 상태 머신, PX4 발행 계층을 연결한다.
        # --------------------------------------------------------
        self.px4 = PX4Interface(self, vehicle_ids)

        self.qwen = QwenToolCaller(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            limits=limits,
            logger=self.get_logger(),
        )

        self.mission = MissionExecutor(
            px4=self.px4,
            get_local_pose=self._get_local_pose,
            thresholds=thresholds,
            history_size=history_size,
            logger=self.get_logger(),
        )

        # --------------------------------------------------------
        # Timer: 고정 주기로 heartbeat, setpoint, 도착 판정을 수행한다.
        # --------------------------------------------------------
        self.timer = self.create_timer(
            timer_period, self.timer_callback, callback_group=self.callback_group
        )

        self.get_logger().info("🦾🧠 Qwen3 Tool Calling 드론 노드 시작")
        self.get_logger().info(f"🤖 Qwen model: {self.qwen.model_path}")
        self.get_logger().info("🧰 Tools: takeoff / move / land / goto_history / reverse_plan")

        threading.Thread(
            target=self.user_input_loop, daemon=True, name="drone-command-input"
        ).start()

    # ========================================================
    # Pose helper (mission_executor가 콜백으로 사용)
    # ========================================================

    def _get_local_pose(self):
        """PX4의 최신 NED pose를 미션 executor가 사용할 형식으로 반환한다."""
        with self.state_lock:
            x = safe_float(self.local_position.x)
            y = safe_float(self.local_position.y)
            z = safe_float(self.local_position.z)
            yaw = safe_float(getattr(self.local_position, "heading", 0.0))
            return x, y, z, yaw

    def _build_state_text(self) -> str:
        x, y, z, yaw = self._get_local_pose()
        import math
        return (
            "[현재 드론 상태 NED]\n"
            f"x={x:.4f}, y={y:.4f}, z={z:.4f}\n"
            f"현재 고도={-z:.2f}m\n"
            f"yaw={yaw:.4f}rad ({math.degrees(yaw):.1f}°)\n"
            f"history 크기={len(self.mission.position_history)}\n"
            f"완료된 move 수={len(self.mission.forward_move_history)}"
        )

    # ========================================================
    # User input / command processing
    # ========================================================

    def user_input_loop(self) -> None:
        """별도 스레드에서 명령을 받고 ROS 상태와 충돌하지 않게 처리한다."""
        print("\n🗣️ 명령 (exit으로 종료, '멈춰'로 현재 이동 정지)")
        print("예: '5m 이륙해', '앞으로 3m', '오른쪽으로 2m 가'")
        print("예: '이전 위치로', '처음 위치로', '왔던 길로 돌아가'\n")

        while rclpy.ok():
            try:
                user_input = input("\n🗣️ 명령: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            lower = user_input.lower()
            if lower in ("exit", "quit"):
                rclpy.shutdown()
                break

            if lower in ("멈춰", "정지", "stop"):
                with self.state_lock:
                    self.mission.stop_current_mission()
                continue

            if not self.position_received:
                print("⚠️ PX4 위치 미수신")
                continue

            self.process_command(user_input)

    def process_command(self, user_input: str) -> None:
        """자연어 -> Qwen Tool Call -> mission queue."""
        # Qwen은 계획만 만들고, 실제 실행과 안전 상태 변경은 MissionExecutor가 담당한다.
        with self.state_lock:
            busy = (
                self.mission.active_tool is not None
                or self.mission.mission_queue
                or self.mission.in_landing
            )
        if busy:
            print("⚠️ 현재 임무가 실행 중입니다. 완료 후 다음 명령을 입력하세요.")
            return

        try:
            print("\n🤔 Qwen3 호출 중...")
            state_text = self._build_state_text()
            raw, tool_calls = self.qwen.infer_tool_calls(state_text, user_input)
            print(f"\n🧠 Qwen raw output:\n{raw}")

            if not tool_calls:
                print("❌ Tool Call을 찾지 못했습니다.")
                return

            ok, reason, valid_calls = self.qwen.validate_all(tool_calls)
            if not ok:
                print(f"❌ Tool Call 거부: {reason}")
                return

            print(f"\n📋 Tool Calls ({len(valid_calls)}개):")
            for i, call in enumerate(valid_calls, 1):
                print(f"  {i}. {call}")

            with self.state_lock:
                self.mission.enqueue_tool_calls(valid_calls)
            print("✅ Tool Call 실행 시작")

        except Exception as exc:
            print(f"❌ 명령 처리 오류: {exc}")

    # ========================================================
    # Timer
    # ========================================================

    def timer_callback(self) -> None:
        """착륙 여부를 읽어 미션 상태 머신을 한 주기 진행시킨다."""
        with self.state_lock:
            is_landed = bool(getattr(self.land_detected, "landed", False))
            self.mission.tick(is_landed)

    # ========================================================
    # ROS callbacks
    # ========================================================
    # 아래 콜백들은 수신 메시지를 최신 상태로 저장하고 timer_callback이 사용하게 한다.

    def position_callback(self, msg: VehicleLocalPosition) -> None:
        with self.state_lock:
            self.local_position = msg
            if not self.position_received:
                self.position_received = True
                self.get_logger().info(
                    f"📡 PX4 위치 수신: X:{msg.x:.2f}, Y:{msg.y:.2f}, Z:{msg.z:.2f}"
                )

    def status_callback(self, msg: VehicleStatus) -> None:
        with self.state_lock:
            self.vehicle_status = msg

    def land_detected_callback(self, msg: VehicleLandDetected) -> None:
        with self.state_lock:
            self.land_detected = msg

    def image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            pos_str = (
                f"Pos:({self.local_position.x:.1f},"
                f"{self.local_position.y:.1f},{self.local_position.z:.1f})"
            )
            tgt_str = (
                f"Tgt:({self.mission.target_x:.1f},"
                f"{self.mission.target_y:.1f},{self.mission.target_z:.1f})"
            )
            cv2.putText(
                cv_image,
                f"Q:{len(self.mission.mission_queue)} | "
                f"Hist:{len(self.mission.position_history)} | "
                f"Moves:{len(self.mission.forward_move_history)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.putText(cv_image, pos_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(cv_image, tgt_str, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imshow("Drone View", cv_image)
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().debug(f"camera display error: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DroneUnifiedNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
