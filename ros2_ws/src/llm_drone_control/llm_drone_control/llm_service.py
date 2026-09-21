from pathlib import Path
from datetime import datetime
import yaml
import torch
import rclpy
import re

from rclpy.node import Node
from std_msgs.msg import String

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from guide_interfaces.srv import GuideLLM
from ament_index_python.packages import get_package_share_directory

from llm_drone_control.schema import DEFINE_SCHEMA, SYSTEM_PROMPT


class LLMService(Node):

    def __init__(self):

        super().__init__("llm_service")

        # -------------------------------------------------------------
        # 로그 파일 설정
        # -------------------------------------------------------------

        log_dir = Path.cwd() / "llm_logs"
        log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        current_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.log_file_path = (
            log_dir /
            f"llm_log_{current_time}.log"
        )

        self.get_logger().info(
            f"📁 로그 파일 생성 경로: {self.log_file_path}"
        )


        # -------------------------------------------------------------
        # 모델 및 설정 로드
        # -------------------------------------------------------------

        config_path = (
            Path(
                get_package_share_directory(
                    "llm_drone_control"
                )
            )
            / "config"
            / "model.yaml"
        )

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:

            config = yaml.safe_load(f) or {}


        model_config = config.get(
            "model",
            {}
        )


        model_path = Path(
            model_config.get(
                "path",
                "~/Drone-VLA-VLM-Control-Project-Arion-/models/Qwen3-1.7B"
            )
        ).expanduser()


        lora_path = Path(
            model_config.get(
                "lora_path",
                "~/Drone-VLA-VLM-Control-Project-Arion-/models/finetuned_qwen3-1.7B_drone_lora"
            )
        ).expanduser()


        max_new_tokens = model_config.get(
            "max_new_tokens",
            64
        )


        do_sample = model_config.get(
            "do_sample",
            False
        )


        torch_dtype = model_config.get(
            "torch_dtype",
            "auto"
        )


        device_map = model_config.get(
            "device_map",
            "auto"
        )


        if not model_path.exists():

            raise FileNotFoundError(
                f"모델 경로가 존재하지 않습니다: {model_path}"
            )


        # -------------------------------------------------------------
        # Tokenizer
        # -------------------------------------------------------------

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
        )


        # -------------------------------------------------------------
        # Model
        # -------------------------------------------------------------

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
        )


        # -------------------------------------------------------------
        # LoRA
        # -------------------------------------------------------------

        if lora_path:

            if not lora_path.exists():

                raise FileNotFoundError(
                    f"LoRA 경로가 존재하지 않습니다: {lora_path}"
                )


            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path,
                local_files_only=True
            )


        self.model.eval()


        # -------------------------------------------------------------
        # Generation 설정
        # -------------------------------------------------------------

        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None


        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample


        # =============================================================
        # ROS2 Publisher
        # =============================================================

        self.response_pub = self.create_publisher(
            String,
            "llm_response",
            10
        )


        # =============================================================
        # 기존 GuideLLM Service
        # =============================================================

        self.srv = self.create_service(
            GuideLLM,
            "llm",
            self.callback
        )


        # =============================================================
        # 스마트폰 음성 명령 Subscriber
        # =============================================================

        self.voice_sub = self.create_subscription(
            String,
            "/voice_command",
            self.voice_command_callback,
            10
        )


        # -------------------------------------------------------------
        # 완료 로그
        # -------------------------------------------------------------

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            "Qwen3-1.7B + LoRA LLM Service Ready"
        )

        self.get_logger().info(
            "GuideLLM Service : /llm"
        )

        self.get_logger().info(
            "Smartphone Topic : /voice_command"
        )

        self.get_logger().info(
            "LLM Response     : /llm_response"
        )

        self.get_logger().info(
            "=========================================="
        )


    # ================================================================
    # 기존 STT → /llm Service
    # ================================================================

    def callback(
        self,
        request,
        response
    ):

        result = self.process_prompt(
            request.prompt
        )

        response.response = result

        return response


    # ================================================================
    # 스마트폰 → /voice_command
    # ================================================================

    def voice_command_callback(
        self,
        msg
    ):

        prompt = msg.data.strip()


        if not prompt:

            self.get_logger().warn(
                "⚠️ 스마트폰에서 빈 음성 명령이 들어왔습니다."
            )

            return


        self.get_logger().info(
            f"📱 스마트폰 음성 명령 수신: {prompt}"
        )


        # ------------------------------------------------------------
        # 기존 STT와 동일한 LLM 처리 함수 사용
        # ------------------------------------------------------------

        result = self.process_prompt(
            prompt
        )


        self.get_logger().info(
            f"📤 스마트폰 명령 LLM 처리 완료: {result}"
        )


    # ================================================================
    # 공통 LLM 처리
    # ================================================================

    def process_prompt(
        self,
        prompt
    ):

        self.get_logger().info(
            f"🤖 LLM 입력: {prompt}"
        )


        # ------------------------------------------------------------
        # Chat Template
        # ------------------------------------------------------------

        messages = [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": prompt
            }

        ]


        inputs = (
            self.tokenizer.apply_chat_template(

                messages,

                tools=DEFINE_SCHEMA,

                add_generation_prompt=True,

                enable_thinking=False,

                return_tensors="pt",

                return_dict=True
            )
            .to(self.model.device)
        )


        # ------------------------------------------------------------
        # LLM Generate
        # ------------------------------------------------------------

        with torch.inference_mode():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=self.max_new_tokens,

                do_sample=self.do_sample

            )


        # ------------------------------------------------------------
        # Decode
        # ------------------------------------------------------------

        result = self.tokenizer.decode(

            outputs[
                0
            ][
                inputs["input_ids"].shape[-1]:
            ],

            skip_special_tokens=True

        )


        # ------------------------------------------------------------
        # Tool Call 추출
        # ------------------------------------------------------------

        matches = re.findall(

            r"<tool_call>\s*(.*?)\s*</tool_call>",

            result,

            re.DOTALL

        )


        cleaned_tool_calls = []


        for m in matches:

            s = re.sub(
                r"\s+",
                " ",
                m.strip()
            )

            cleaned_tool_calls.append(
                f"<tool_call>{s}</tool_call>"
            )


        # ------------------------------------------------------------
        # 최종 결과
        # ------------------------------------------------------------

        if cleaned_tool_calls:

            response_text = "\n".join(
                cleaned_tool_calls
            )

        else:

            response_text = result.strip()


        # =============================================================
        # /llm_response Publish
        # =============================================================

        msg = String()

        msg.data = response_text

        self.response_pub.publish(
            msg
        )


        # -------------------------------------------------------------
        # 로그
        # -------------------------------------------------------------

        self.get_logger().info(
            f"질문: {prompt}"
        )

        self.get_logger().info(
            f"답변: {response_text}"
        )


        log_timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )


        log_entry = (

            f"[{log_timestamp}]\n"

            f"[질문] {prompt}\n"

            f"[답변] {response_text}\n"

            f"{'=' * 60}\n"

        )


        try:

            with open(
                self.log_file_path,
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    log_entry
                )

        except Exception as e:

            self.get_logger().error(
                f"로그 파일 저장 중 오류 발생: {e}"
            )


        return response_text


# ====================================================================
# Main
# ====================================================================

def main():

    rclpy.init()

    node = LLMService()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
