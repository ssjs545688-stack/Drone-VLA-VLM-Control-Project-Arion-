"""
mission_executor.py

Tool Call 큐 관리, 도착 판정, 위치 history, reverse_plan(경로 역산) 로직을
ROS2 콜백/구독과 분리한 순수 상태 머신으로 분리했다.

이 클래스는 rclpy Node를 직접 상속하지 않는다. 대신:
  - PX4Interface: 실제 명령 발행
  - get_local_pose(): 현재 (x, y, z, yaw)를 반환하는 콜백
  - logger: rclpy logger 또는 표준 logging.Logger

만 주입받는다. 그래서 시뮬레이터 없이도(=콜백을 mock으로 대체하면) 미션 로직만
따로 단위 테스트할 수 있다.
"""

import copy
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .px4_interface import PX4Interface
from .utils import clamp_angle, safe_float

Pose = Tuple[float, float, float, float]  # (x, y, z, yaw)


@dataclass
class ArrivalThresholds:
    distance_m: float = 0.30
    confirm_count: int = 10
    yaw_rad: float = 0.10


class MissionExecutor:
    """Tool Call을 순차 실행하는 비ROS 상태 머신.

    enqueue_tool_calls()가 큐를 만들고 start_next_mission()이 현재 작업을
    시작한다. 이후 tick()이 setpoint를 발행하고 도착을 확인하면
    on_tool_completed()를 거쳐 다음 작업으로 넘어간다.
    """
    def __init__(
        self,
        px4: PX4Interface,
        get_local_pose: Callable[[], Pose],
        thresholds: Optional[ArrivalThresholds] = None,
        history_size: int = 100,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._px4 = px4
        self._get_pose = get_local_pose
        self._th = thresholds or ArrivalThresholds()
        self._log = logger or logging.getLogger("mission_executor")

        self.mission_queue: deque = deque()
        self.active_tool: Optional[Dict[str, Any]] = None
        self.current_mission_name: Optional[str] = None
        self.mission_state = "STANDBY"

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_yaw = 0.0
        self.has_target = False
        self.in_landing = False
        self.arrival_count = 0

        # 실제 명령 시작 위치 기록
        self.position_history: deque = deque(maxlen=history_size)

        # 실제 완료된 forward move 기록. reverse_plan은 여기서 역순으로 생성한다.
        self.forward_move_history: List[Dict[str, Any]] = []
        self.executing_reverse = False
        self.reverse_source_count = 0

    # ========================================================
    # Enqueue / dispatch
    # ========================================================

    def enqueue_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """검증된 Tool 목록을 실행 큐로 복사하고 첫 작업을 즉시 시작한다."""
        self.mission_queue.clear()
        for call in tool_calls:
            self.mission_queue.append(copy.deepcopy(call))
        self.executing_reverse = False
        self.start_next_mission()

    def start_next_mission(self) -> None:
        """큐에서 다음 Tool을 꺼내 실행 상태를 만든다.

        각 Tool의 _start_* 메서드가 목표 좌표와 상태를 준비하고,
        PX4Interface를 통해 필요한 명령만 보낸다.
        """
        if not self.mission_queue:
            self.active_tool = None
            self.current_mission_name = None
            self.has_target = False
            self.arrival_count = 0
            self.mission_state = "HOVER"

            if self.executing_reverse:
                self._log.info("✅ reverse_plan 완료")
                self.forward_move_history.clear()
                self.executing_reverse = False

            self._log.info("✅ 모든 임무 완료! HOVER 대기 중")
            return

        cmd = self.mission_queue.popleft()
        name = cmd["name"]
        args = cmd.get("arguments", {})

        self.active_tool = copy.deepcopy(cmd)
        self.current_mission_name = name
        self.arrival_count = 0
        self.has_target = False
        self.in_landing = False

        current_x, current_y, current_z, current_yaw = self._get_pose()

        if name == "takeoff":
            self._start_takeoff(args, current_x, current_y, current_z, current_yaw)
        elif name == "move":
            self._start_move(args, current_x, current_y, current_z, current_yaw)
        elif name == "land":
            self._start_land()
        elif name == "goto_history":
            self._start_goto_history(args)
        elif name == "reverse_plan":
            self._start_reverse_plan()
        else:
            self._log.error(f"❌ 알 수 없는 Tool: {name}")
            self.active_tool = None
            self.current_mission_name = None
            self.start_next_mission()

    # ----------------------------------------------------
    # 1. takeoff
    # ----------------------------------------------------

    def _start_takeoff(self, args, current_x, current_y, current_z, current_yaw):
        """현재 x/y에서 지정 고도까지 이동할 목표를 만들고 ARM/OFFBOARD를 요청한다."""
        altitude = float(args["altitude"])

        # 명령 시작 직전 위치를 history에 기록
        self.record_position(current_x, current_y, current_z, current_yaw)

        self.target_x = current_x
        self.target_y = current_y
        self.target_z = -altitude
        self.target_yaw = current_yaw
        self.has_target = True
        self.mission_state = "EXECUTING"

        # Offboard setpoint를 먼저 활성화하고 ARM/Offboard를 요청한다.
        self._px4.engage_offboard()
        self._px4.arm()

        self._log.info(
            f"🚀 takeoff → altitude={altitude:.2f}m target_z={self.target_z:.2f}"
        )

    # ----------------------------------------------------
    # 2. move
    # ----------------------------------------------------

    def _start_move(self, args, current_x, current_y, current_z, current_yaw):
        """body-frame 이동량을 현재 yaw로 world-frame 목표 좌표로 변환한다."""
        dx_body = float(args["dx"])
        dy_body = float(args["dy"])
        dz_up = float(args["dz"])
        dyaw_deg = float(args["d_yaw"])

        # 현재 실제 yaw 기준 body -> world 변환
        delta_x = dx_body * math.cos(current_yaw) - dy_body * math.sin(current_yaw)
        delta_y = dx_body * math.sin(current_yaw) + dy_body * math.cos(current_yaw)

        self.target_x = current_x + delta_x
        self.target_y = current_y + delta_y
        self.target_z = current_z - dz_up  # PX4 NED
        self.target_yaw = clamp_angle(current_yaw + math.radians(dyaw_deg))
        self.has_target = True
        self.mission_state = "EXECUTING"

        # reverse 실행 중에는 원래 history를 오염시키지 않는다.
        if not self.executing_reverse:
            self.record_position(current_x, current_y, current_z, current_yaw)

        self._log.info(
            f"📍 move → dx={dx_body:.2f}, dy={dy_body:.2f}, dz={dz_up:.2f}, "
            f"d_yaw={dyaw_deg:.1f}° | target=({self.target_x:.2f}, "
            f"{self.target_y:.2f}, {self.target_z:.2f}, "
            f"{math.degrees(self.target_yaw):.1f}°)"
        )

    # ----------------------------------------------------
    # 3. land
    # ----------------------------------------------------

    def _start_land(self):
        """목표 도착 판정 대신 PX4의 landed 신호를 기다리는 상태로 전환한다."""
        self.mission_state = "LANDING"
        self.in_landing = True
        self.has_target = False
        self._px4.land()
        self._log.info("🛬 land 요청")

    # ----------------------------------------------------
    # 4. goto_history
    # ----------------------------------------------------

    def _start_goto_history(self, args):
        """기록된 시작 위치를 찾아 직접 이동 목표로 설정한다."""
        recall = args["recall"]
        target_pos = self.resolve_history(recall)

        if target_pos is None:
            self._log.error(
                f"❌ goto_history 실패: recall={recall}, "
                f"history={len(self.position_history)}"
            )
            self.active_tool = None
            self.current_mission_name = None
            self.start_next_mission()
            return

        tx, ty, tz, t_yaw = target_pos
        self.target_x = tx
        self.target_y = ty
        self.target_z = tz
        self.target_yaw = t_yaw
        self.has_target = True
        self.mission_state = "EXECUTING"

        # History jump는 단순 move와 같은 경로가 아니므로
        # 이후 reverse_plan의 forward path를 깨끗하게 초기화한다.
        self.forward_move_history.clear()

        label = "출발지(원점)" if recall == "first" else "직전 위치"
        self._log.info(
            f"⏪ goto_history({recall}) → {label} "
            f"target=({tx:.2f}, {ty:.2f}, {tz:.2f}, {math.degrees(t_yaw):.1f}°)"
        )

    # ----------------------------------------------------
    # 5. reverse_plan
    # ----------------------------------------------------

    def _start_reverse_plan(self):
        """완료된 move 기록을 역순 보상 이동으로 바꿔 큐에 넣는다."""
        if not self.forward_move_history:
            self._log.error("❌ reverse_plan 실패: 완료된 move 이력이 없음")
            self.active_tool = None
            self.current_mission_name = None
            self.start_next_mission()
            return

        reverse_steps = self.build_reverse_plan(self.forward_move_history)
        if not reverse_steps:
            self._log.error("❌ reverse_plan 생성 실패")
            self.active_tool = None
            self.current_mission_name = None
            self.start_next_mission()
            return

        self.executing_reverse = True
        self.reverse_source_count = len(self.forward_move_history)
        self.mission_queue.clear()
        for step in reverse_steps:
            self.mission_queue.append(step)

        self._log.info(
            f"↩️ reverse_plan → {len(self.forward_move_history)}개 move "
            f"를 역순 {len(reverse_steps)}개 step으로 변환"
        )

        # 현재 reverse_plan tool 자체를 active state에서 제외하고
        # 첫 번째 역이동을 실행한다.
        self.active_tool = None
        self.current_mission_name = None
        self.start_next_mission()

    # ========================================================
    # Stop
    # ========================================================

    def stop_current_mission(self) -> None:
        """큐를 비우고 현재 실제 위치에서 정지 setpoint를 유지한다."""
        self.mission_queue.clear()
        self.active_tool = None
        self.current_mission_name = None
        self.executing_reverse = False
        self.in_landing = False
        self.mission_state = "HOVER"

        x, y, z, yaw = self._get_pose()
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self.target_yaw = yaw
        self.has_target = True
        self.arrival_count = 0

        self._log.warning("⏸ STOP → 현재 위치 setpoint 유지 / mission queue 초기화")

    # ========================================================
    # History / reverse plan helpers
    # ========================================================

    def record_position(self, x: float, y: float, z: float, yaw: float) -> None:
        """새 forward 작업 직전의 실제 pose를 history에 저장한다."""
        pos = (float(x), float(y), float(z), float(yaw))
        self.position_history.append(pos)
        idx = len(self.position_history) - 1
        self._log.info(
            f"📌 history[{idx}] 기록 → X:{x:.2f}, Y:{y:.2f}, Z:{z:.2f}, "
            f"Yaw:{math.degrees(yaw):.1f}°"
        )

    def resolve_history(self, recall: str) -> Optional[Pose]:
        """first/previous 요청을 history의 첫 항목 또는 마지막 항목으로 해석한다."""
        if not self.position_history:
            return None
        if recall == "first":
            return self.position_history[0]
        if recall == "previous":
            # position_history에는 각 forward command 시작 시점이 저장된다.
            # 가장 최근 기록이 바로 직전 command 시작 위치다.
            return self.position_history[-1]
        return None

    def build_reverse_plan(self, move_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        완료된 move 이력을 정확히 역순으로 되짚는 Tool Call 목록을 만든다.

        원래 move가 translate(dx,dy,dz) + rotate(d_yaw) 라면,
        복귀는 rotate(-d_yaw) -> translate(-dx,-dy,-dz) 순서로 만든다.
        이렇게 해야 원래 translation이 실행된 body frame으로 돌아갈 수 있다.
        """
        reverse_steps: List[Dict[str, Any]] = []

        for original in reversed(move_history):
            args = original.get("arguments", {})
            dx = safe_float(args.get("dx"))
            dy = safe_float(args.get("dy"))
            dz = safe_float(args.get("dz"))
            d_yaw = safe_float(args.get("d_yaw"))

            if abs(d_yaw) > 1e-9:
                reverse_steps.append({
                    "name": "move",
                    "arguments": {"dx": 0.0, "dy": 0.0, "dz": 0.0, "d_yaw": -d_yaw},
                    "_reverse_generated": True,
                })

            if abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dz) > 1e-9:
                reverse_steps.append({
                    "name": "move",
                    "arguments": {"dx": -dx, "dy": -dy, "dz": -dz, "d_yaw": 0.0},
                    "_reverse_generated": True,
                })

        return reverse_steps

    def on_tool_completed(self) -> None:
        """현재 active tool이 도착 완료된 시점의 후처리."""
        if not self.active_tool:
            return

        name = self.active_tool.get("name")

        if name == "move" and not self.executing_reverse:
            # 실제로 완료된 원래 forward move만 기록한다.
            self.forward_move_history.append(copy.deepcopy(self.active_tool))
            self._log.info(f"📝 forward_move_history={len(self.forward_move_history)}")

        self.active_tool = None
        self.current_mission_name = None
        self.has_target = False
        self.arrival_count = 0

    # ========================================================
    # Timer tick — 매 주기 ROS 노드의 timer_callback에서 호출
    # ========================================================

    def tick(self, is_landed: bool) -> None:
        """heartbeat -> setpoint -> 도착 확인 순서로 타이머 한 주기를 실행한다.

        일반 이동은 confirm_count회 연속으로 오차 범위 안에 있어야 완료되고,
        착륙 중에는 위치 판정 없이 PX4의 landed 신호만 사용한다.
        """
        self._px4.publish_offboard_heartbeat()

        if self.in_landing:
            if is_landed:
                self.in_landing = False
                self.mission_state = "LANDED"
                self.has_target = False
                self.active_tool = None
                self.current_mission_name = None
                self.mission_queue.clear()
                self.forward_move_history.clear()
                self._log.info("✅ 착륙 감지 완료")
            return

        if not self.has_target:
            return

        self._px4.publish_setpoint(self.target_x, self.target_y, self.target_z, self.target_yaw)

        cur_x, cur_y, cur_z, cur_yaw = self._get_pose()
        dist = math.sqrt(
            (cur_x - self.target_x) ** 2
            + (cur_y - self.target_y) ** 2
            + (cur_z - self.target_z) ** 2
        )
        yaw_diff = abs(clamp_angle(cur_yaw - self.target_yaw))

        if dist < self._th.distance_m and yaw_diff < self._th.yaw_rad:
            self.arrival_count += 1
            if self.arrival_count >= self._th.confirm_count:
                self._log.info(
                    f"✔ 도착 완료 | dist={dist:.3f}m yaw_diff={math.degrees(yaw_diff):.2f}°"
                )
                self.on_tool_completed()

                if self.mission_queue:
                    self.start_next_mission()
                else:
                    self.mission_state = "HOVER"
                    if self.executing_reverse:
                        self._log.info("✅ reverse_plan 전체 경로 복귀 완료")
                        self.forward_move_history.clear()
                        self.executing_reverse = False
                    self._log.info("✅ 모든 setpoint/tool 완료")
        else:
            self.arrival_count = 0
