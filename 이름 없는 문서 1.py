"""공통 시스템 프롬프트로 Qwen LLM ROS2 서비스를 실행한다."""

import time
from pathlib import Path
from typing import Any

import rclpy
from llama_cpp import Llama
from rclpy.node import Node

from drone_command_interface.prompts import SYSTEM_PROMPT
from llm_ros2.srv import AskLLM


DEFAULT_MODEL_PATH = "model/v0.4-Q4_K_M.gguf"
DEFAULT_CONTEXT_SIZE = 16384
DEFAULT_THREAD_COUNT = 4
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120


def resolve_model_path(model_path: str) -> Path:
    """모델 경로를 실행 가능한 절대경로로 변환한다."""
    path = Path(model_path).expanduser()

    if path.is_absolute():
        return path.resolve()

    search_roots = (Path.cwd(), *Path(__file__).resolve().parents)
    for search_root in search_roots:
        candidate = search_root / path
        if candidate.is_file():
            return candidate.resolve()

    return (Path.cwd() / path).resolve()


class LLMServiceNode(Node):
    """자연어 요청을 Qwen 모델에 전달하는 ROS2 서비스 노드다."""

    def __init__(self) -> None:
        super().__init__("llm_service")

        self._declare_parameters()
        model_path = resolve_model_path(
            self.get_parameter("model_path").value,
        )
        context_size = self.get_parameter("context_size").value
        thread_count = self.get_parameter("threads").value

        if not model_path.is_file():
            error_message = f"모델 파일을 찾을 수 없습니다: {model_path}"
            self.get_logger().error(error_message)
            raise FileNotFoundError(error_message)

        self.get_logger().info(f"모델 로딩 중: {model_path}")
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=context_size,
            n_threads=thread_count,
            verbose=False,
        )
        self.get_logger().info("모델 로딩 완료")

        self._service = self.create_service(
            AskLLM,
            "ask_llm",
            self.handle_request,
        )
        self.get_logger().info("LLM Service 준비 완료: /ask_llm")

    def _declare_parameters(self) -> None:
        """YAML 파일로 덮어쓸 수 있는 기본 실행 파라미터를 선언한다."""
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("context_size", DEFAULT_CONTEXT_SIZE)
        self.declare_parameter("threads", DEFAULT_THREAD_COUNT)
        self.declare_parameter("max_tokens", DEFAULT_MAX_TOKENS)
        self.declare_parameter("temperature", DEFAULT_TEMPERATURE)
        self.declare_parameter(
            "timeout_seconds",
            DEFAULT_TIMEOUT_SECONDS,
        )

    def handle_request(self, request: Any, response: Any) -> Any:
        """사용자 명령을 모델에 전달하고 생성된 응답을 반환한다."""
        question = request.question
        self.get_logger().info(f"질문 수신: {question}")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]
        max_tokens = self.get_parameter("max_tokens").value
        temperature = self.get_parameter("temperature").value

        self.get_logger().info("LLM 응답 생성 중")
        start_time = time.monotonic()
        output = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed_time = time.monotonic() - start_time

        answer = output["choices"][0]["message"]["content"]
        response.response = answer

        self.get_logger().info(
            f"LLM 응답 생성 완료: elapsed_seconds={elapsed_time:.2f}"
        )
        self.get_logger().info(f"LLM 응답: {answer}")
        return response


def main(args: list[str] | None = None) -> None:
    """LLM 서비스 노드를 초기화하고 종료될 때까지 실행한다."""
    rclpy.init(args=args)
    node = LLMServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

