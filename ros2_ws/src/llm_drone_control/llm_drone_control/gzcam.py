import os
import time
import typing
import argparse
import threading

from gz.msgs10.image_pb2 import Image
from gz.transport13 import SubscribeOptions
from gz.transport13 import Node

import cv2
import numpy as np
import PIL
from PIL import Image as PIL_Image


# ── 하드코딩 제거: 환경변수 → 기본값 순으로 설정을 읽는다 ──
DEFAULT_TOPIC = os.environ.get("GZ_CAMERA_TOPIC", "/camera")
DEFAULT_WIDTH = int(os.environ.get("GZ_CAMERA_WIDTH", "1280"))
DEFAULT_HEIGHT = int(os.environ.get("GZ_CAMERA_HEIGHT", "960"))
DEFAULT_CHANNELS = int(os.environ.get("GZ_CAMERA_CHANNELS", "3"))  # RGB=3, RGBA=4 등


class GzCam:
    """Gazebo Transport 카메라 토픽을 OpenCV 이미지로 변환하는 수신기."""

    def __init__(self, topic_name: str, resolution: tuple, channels: int = 3):
        # Gazebo 이미지 메시지를 받을 노드와 콜백을 생성한다.
        self._res = resolution
        self._channels = channels  # 더 이상 3으로 고정하지 않음
        self._node = Node()  # gz node
        self._node.subscribe(Image, topic_name, self._cb)

        self._img = None
        self._condition = threading.Condition()

    def _cb(self, image: Image) -> None:
        # 바이트 배열을 지정된 해상도와 채널 수의 NumPy 이미지로 복원한다.
        raw_image_data = image.data
        np_image = np.frombuffer(raw_image_data, dtype=np.uint8).reshape(
            (self._res[1], self._res[0], self._channels)
        )
        if self._channels == 3:
            cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        elif self._channels == 4:
            cv2_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGR)
        else:
            cv2_image = np_image
        # 최신 프레임을 저장하고 대기 중인 소비자에게 새 프레임 도착을 알린다.
        with self._condition:
            self._img = cv2_image
            self._condition.notify_all()

    def get_next_image(self, timeout=None):
        """새 프레임을 기다린 뒤 반환하고, 반환한 프레임은 버퍼에서 제거한다."""
        with self._condition:
            if self._img is None:
                self._condition.wait_for(lambda: self._img is not None, timeout=timeout)
            ret_img = self._img
            self._img = None
            return ret_img


def parse_args():
    """카메라 토픽과 이미지 형식을 명령행 인자로 읽는다."""
    parser = argparse.ArgumentParser(description="Gazebo 카메라 뷰어")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="구독할 gz 카메라 토픽")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="이미지 가로 해상도")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="이미지 세로 해상도")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="채널 수 (3=RGB, 4=RGBA)")
    return parser.parse_args()


if __name__ == "__main__":
    # 독립 실행 시 Gazebo 카메라 영상을 OpenCV 창에 표시한다.
    args = parse_args()
    cam = GzCam(args.topic, (args.width, args.height), args.channels)
    while True:
        img = cam.get_next_image()
        if img is None:
            continue
        cv2.imshow("pic-display", img)
        cv2.waitKey(1)
