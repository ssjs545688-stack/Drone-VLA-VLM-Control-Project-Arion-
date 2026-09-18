#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stt_node.py

마이크로부터 음성을 입력받아 Google Speech Recognition API를 사용하여 텍스트로 변환한 뒤,
llm_service의 '/llm' (GuideLLM) 서비스 클라이언트로 요청을 전송하는 ROS 2 노드입니다.
"""

import threading
import rclpy
from rclpy.node import Node
import speech_recognition as sr
from guide_interfaces.srv import GuideLLM


class STTNode(Node):
    def __init__(self):
        super().__init__("stt_node")

        self.client = self.create_client(GuideLLM, "llm")

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("⏳ 'llm' 서비스 연결 대기 중...")

        self.get_logger().info("✅ 'llm' 서비스 연결 완료. STT 준비 시작...")
        
        # 💡 [추가] LLM 응답 대기용 이벤트 객체 생성
        self.llm_done_event = threading.Event()

        self.recognizer = sr.Recognizer()

        self.stt_thread = threading.Thread(target=self.speech_loop, daemon=True)
        self.stt_thread.start()

    # (speech_loop 함수는 기존과 동일하게 유지)
    def speech_loop(self):
        while rclpy.ok():
            try:
                input("\n⌨️ 엔터(Enter)키를 누르면 음성 인식을 시작합니다...")

                with sr.Microphone() as source:
                    self.get_logger().info("🎙️ 주변 소음 측정 중 (0.5초)...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                    print("\n" + "=" * 50)
                    self.get_logger().info("🗣️ [지금 말씀하세요] 명령을 입력 받는 중...")
                    print("=" * 50)

                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=10.0)
                    self.get_logger().info("🔊 음성 수신 완료. 텍스트 변환 중...")

                    text = self.recognizer.recognize_whisper(audio, language="ko", model="base")
                    self.get_logger().info(f"📝 인식된 텍스트: '{text}'")

                    self.send_llm_request(text)

            except sr.WaitTimeoutError:
                self.get_logger().warn("⏰ 대기 시간 초과: 5초 동안 아무 말씀도 하지 않으셨습니다.")
            except sr.UnknownValueError:
                self.get_logger().warn("❓ 음성을 정확히 인식하지 못했습니다.")
            except sr.RequestError as e:
                self.get_logger().error(f"🌐 Google STT API 요청 실패: {e}")
            except Exception as e:
                self.get_logger().error(f"⚠️ 오류 발생: {e}")

    def send_llm_request(self, prompt_text: str):
        """인식된 텍스트를 '/llm' 서비스로 요청하고 응답을 기다림"""
        req = GuideLLM.Request()
        req.prompt = prompt_text

        self.get_logger().info(f"🚀 LLM 서비스로 요청 전달: '{prompt_text}'")

        # 💡 [추가] 이벤트 초기화 및 응답 대기 시작
        self.llm_done_event.clear()
        
        future = self.client.call_async(req)
        future.add_done_callback(self.response_callback)
        
        self.get_logger().info("⏳ LLM이 명령을 처리하고 있습니다. 잠시만 기다려주세요...")
        
        # 💡 [추가] LLM 응답(콜백)이 올 때까지 현재 스레드(음성 루프) 멈춤
        self.llm_done_event.wait()

    def response_callback(self, future):
        """LLM 응답이 도착했을 때 실행되는 콜백"""
        try:
            response = future.result()
            self.get_logger().info(f"📥 LLM 응답 수신 완료:")
            self.get_logger().info(f"{response.response}")
            self.get_logger().info("➡️ llm_service에서 /llm_response 토픽으로 발행되어 flight_controller가 제어를 진행합니다.")
        except Exception as e:
            self.get_logger().error(f"❌ LLM 서비스 호출 중 예외 발생: {e}")
        finally:
            # 💡 [추가] 응답 처리가 끝나면 멈춰둔 스레드의 대기 상태를 해제함
            self.llm_done_event.set()


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("사용자에 의해 STT 노드가 종료되었습니다.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
