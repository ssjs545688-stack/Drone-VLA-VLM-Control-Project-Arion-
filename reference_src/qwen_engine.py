"""
qwen_engine.py

로컬 Qwen3 모델 로딩, 프롬프트 생성, 추론, Tool Call 파싱/검증을 담당한다.
이 클래스는 rclpy에 의존하지 않는다 — ROS 노드는 현재 드론 상태를 문자열로
만들어 build_prompt()에 넘겨주기만 하면 되고, 그 외의 로직(모델 관리, 파싱,
스키마 검증)은 전부 여기서 독립적으로 처리한다.
독립적이기 때문에 ROS 없이도 단위 테스트/CLI 스크립트에서 재사용 가능하다.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .schema import SYSTEM_PROMPT, TOOL_NAMES
from .utils import extract_balanced_json, is_finite_number, normalize_tool_call


@dataclass
class ValidationLimits:
    """move/takeoff 등 Tool 인자의 물리적 한계값."""
    max_takeoff_alt: float = 10.0
    max_horizontal_move: float = 20.0
    max_vertical_move: float = 10.0
    max_yaw_change: float = 360.0


class QwenToolCaller:
    """로컬 Qwen3 Tool Calling 엔진.

    모델은 자연어를 직접 PX4 명령으로 바꾸지 않는다. 구조화된 Tool Call을
    생성하고 파싱 및 물리 한계 검증을 통과한 호출만 실행 계층으로 넘긴다.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        limits: Optional[ValidationLimits] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_path = os.path.expanduser(model_path)
        self.max_new_tokens = max_new_tokens
        self.limits = limits or ValidationLimits()
        self._log = logger or logging.getLogger("qwen_engine")

        self.tokenizer, self.model = self._load_qwen(self.model_path)

    # ========================================================
    # Model loading
    # ========================================================

    def _load_qwen(self, model_path: str):
        """로컬 Qwen3 모델을 로드한다."""
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Qwen 모델 디렉터리가 없습니다: {model_path}\n"
                "ROS2 실행 전에 model_path 또는 QWEN_MODEL_PATH를 확인하세요."
            )

        config_path = os.path.join(model_path, "config.json")
        tokenizer_path = os.path.join(model_path, "tokenizer_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"로컬 모델의 config.json이 없습니다: {config_path}"
            )
        if not os.path.isfile(tokenizer_path):
            self._log.warning(
                f"⚠ tokenizer_config.json이 없습니다: {tokenizer_path}"
            )

        self._log.info("📦 Qwen tokenizer 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        self._log.info("📦 Qwen model 로딩 중...")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        load_kwargs: Dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.eval()

        self._log.info(
            f"✅ Qwen 로딩 완료 | CUDA={torch.cuda.is_available()} | dtype={dtype}"
        )
        return tokenizer, model

    # ========================================================
    # Prompt / inference
    # ========================================================
    # 상태 문자열과 사용자 명령을 chat prompt로 만들고, 입력 prompt 이후의
    # 새 토큰만 모델 출력으로 반환한다.

    def build_prompt(self, state_text: str, user_input: str) -> str:
        """
        state_text: 호출 측(ROS 노드)이 만든 "현재 드론 상태" 설명 문자열.
        이 클래스는 x/y/z/yaw 같은 실제 텔레메트리 필드를 알 필요가 없다.
        """
        user_text = f"{state_text}\n\n사용자 명령: {user_input}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # 오래된 transformers에서 enable_thinking 인자를 모를 경우
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def invoke(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")

        try:
            input_device = next(self.model.parameters()).device
        except StopIteration:
            input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return text.strip()

    def infer_tool_calls(self, state_text: str, user_input: str) -> Tuple[str, List[Dict[str, Any]]]:
        """편의 메서드: 프롬프트 생성 -> 추론 -> 파싱까지 한 번에 수행."""
        prompt = self.build_prompt(state_text, user_input)
        raw = self.invoke(prompt)
        return raw, self.parse_tool_calls(raw)

    # ========================================================
    # Tool call parsing
    # ========================================================
    # <tool_call> 형식을 우선 처리하고, 실패하면 balanced JSON과 plain JSON을 시도한다.

    def parse_tool_calls(self, raw_text: str) -> List[Dict[str, Any]]:
        """Qwen 출력에서 1개 이상의 Tool Call을 추출한다."""
        text = raw_text.strip()

        # thinking 블록 제거
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        calls: List[Dict[str, Any]] = []
        tagged = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            text,
            flags=re.DOTALL,
        )

        for body in tagged:
            try:
                obj = json.loads(body.strip())
            except json.JSONDecodeError:
                candidate = extract_balanced_json(body)
                if not candidate:
                    continue
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    continue

            call = normalize_tool_call(obj)
            if call:
                calls.append(call)

        if calls:
            return calls

        # fallback: plain JSON object / array
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("tool_calls"), list):
                    for item in parsed["tool_calls"]:
                        call = normalize_tool_call(item)
                        if call:
                            calls.append(call)
                else:
                    call = normalize_tool_call(parsed)
                    if call:
                        calls.append(call)
            elif isinstance(parsed, list):
                for item in parsed:
                    call = normalize_tool_call(item)
                    if call:
                        calls.append(call)
        except json.JSONDecodeError:
            candidate = extract_balanced_json(text)
            if candidate:
                try:
                    obj = json.loads(candidate)
                    call = normalize_tool_call(obj)
                    if call:
                        calls.append(call)
                except json.JSONDecodeError:
                    pass

        return calls

    # ========================================================
    # Tool call validation
    # ========================================================
    # 파싱 이후 허용된 Tool, 필수 인자, 숫자 형식, 이동/고도/yaw 한계를 확인한다.

    def validate_tool_call(self, call: Dict[str, Any]) -> Tuple[bool, str]:
        name = call.get("name")
        args = call.get("arguments", {})

        if name not in TOOL_NAMES:
            return False, f"허용되지 않은 Tool: {name}"
        if not isinstance(args, dict):
            return False, f"arguments가 object가 아님: {args}"

        if name == "takeoff":
            if "altitude" not in args:
                return False, "takeoff에는 altitude가 필요합니다."
            if not is_finite_number(args["altitude"]):
                return False, "takeoff altitude가 숫자가 아닙니다."
            altitude = float(args["altitude"])
            if altitude <= 0.0 or altitude > self.limits.max_takeoff_alt:
                return False, (
                    f"takeoff altitude 범위 오류: {altitude}m "
                    f"(0 < altitude <= {self.limits.max_takeoff_alt})"
                )

        elif name == "move":
            required = ("dx", "dy", "dz", "d_yaw")
            for key in required:
                if key not in args:
                    return False, f"move에 {key}가 없습니다."
                if not is_finite_number(args[key]):
                    return False, f"move {key}가 숫자가 아닙니다."

            dx = float(args["dx"])
            dy = float(args["dy"])
            dz = float(args["dz"])
            d_yaw = float(args["d_yaw"])

            horizontal = (dx ** 2 + dy ** 2) ** 0.5
            if horizontal > self.limits.max_horizontal_move:
                return False, (
                    f"move 수평 이동량 초과: {horizontal:.2f}m "
                    f"> {self.limits.max_horizontal_move:.2f}m"
                )
            if abs(dz) > self.limits.max_vertical_move:
                return False, (
                    f"move 수직 이동량 초과: {abs(dz):.2f}m "
                    f"> {self.limits.max_vertical_move:.2f}m"
                )
            if abs(d_yaw) > self.limits.max_yaw_change:
                return False, (
                    f"move yaw 변화량 초과: {abs(d_yaw):.2f}° "
                    f"> {self.limits.max_yaw_change:.2f}°"
                )

        elif name == "goto_history":
            recall = args.get("recall")
            if recall not in ("first", "previous"):
                return False, (
                    f"goto_history recall 오류: {recall}. "
                    "허용값은 first / previous 입니다."
                )

        elif name in ("land", "reverse_plan"):
            pass  # no args required

        return True, ""

    def validate_all(self, calls: List[Dict[str, Any]]) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """리스트 전체를 검증하고, 첫 실패 지점에서 멈춘다 (기존 ex.py 동작과 동일)."""
        valid: List[Dict[str, Any]] = []
        for call in calls:
            ok, reason = self.validate_tool_call(call)
            if not ok:
                return False, reason, valid
            valid.append(call)
        return True, "", valid
