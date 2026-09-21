#!/usr/bin/env python3

import socket
import threading
import webbrowser

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import uvicorn


# ============================================================
# 설정
# ============================================================

HOST = "0.0.0.0"
PORT = 8000

DEFAULT_TOPIC = "/voice_command"


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Drone Smartphone Bridge",
    version="3.0.0"
)


# ============================================================
# ROS2 Node
# ============================================================

ros_node = None


class SmartphoneBridge(Node):

    def __init__(self):

        super().__init__("smartphone_bridge")

        self.declare_parameter(
            "topic_name",
            DEFAULT_TOPIC
        )

        self.topic_name = (
            self.get_parameter("topic_name")
            .get_parameter_value()
            .string_value
        )

        self.publisher = self.create_publisher(
            String,
            self.topic_name,
            10
        )

        self.get_logger().info(
            f"Smartphone Bridge started"
        )

        self.get_logger().info(
            f"Publishing topic: {self.topic_name}"
        )


    def publish_voice_command(self, text):

        msg = String()
        msg.data = text

        self.publisher.publish(msg)

        self.get_logger().info(
            f"[SMARTPHONE] {text}"
        )


# ============================================================
# Request Model
# ============================================================

class VoiceCommand(BaseModel):

    text: str


# ============================================================
# Web UI
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>

<html lang="ko">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Drone AI Control</title>


<style>

* {
    box-sizing: border-box;
}


body {

    margin: 0;

    padding: 20px;

    background: #111827;

    color: white;

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}


.container {

    max-width: 600px;

    margin: 0 auto;
}


h1 {

    text-align: center;

    margin-bottom: 30px;
}


.status {

    padding: 15px;

    margin-bottom: 20px;

    border-radius: 12px;

    background: #374151;

    text-align: center;

    font-size: 16px;
}


.status.connected {

    background: #065f46;
}


.status.error {

    background: #991b1b;
}


.card {

    background: #1f2937;

    border-radius: 16px;

    padding: 20px;

    margin-bottom: 20px;
}


.card-title {

    font-size: 18px;

    font-weight: bold;

    margin-bottom: 15px;
}


/* =========================================================
   마이크
   ========================================================= */

#micButton {

    width: 180px;

    height: 180px;

    border-radius: 50%;

    border: none;

    display: block;

    margin: 20px auto;

    font-size: 70px;

    background: #2563eb;

    color: white;

    cursor: pointer;

    transition: 0.2s;
}


#micButton:active {

    transform: scale(0.95);

}


#micButton.recording {

    background: #dc2626;

}


#micButton:disabled {

    background: #4b5563;

    cursor: not-allowed;

}


#voiceResult {

    min-height: 60px;

    padding: 15px;

    background: #111827;

    border-radius: 10px;

    word-break: break-word;

    text-align: center;

}


/* =========================================================
   텍스트 입력
   ========================================================= */

#textCommand {

    width: 100%;

    min-height: 120px;

    padding: 15px;

    border-radius: 10px;

    border: 1px solid #4b5563;

    background: #111827;

    color: white;

    font-size: 17px;

    resize: vertical;

}


#textCommand:focus {

    outline: 2px solid #2563eb;

}


#sendTextButton {

    width: 100%;

    margin-top: 12px;

    padding: 15px;

    border: none;

    border-radius: 10px;

    background: #2563eb;

    color: white;

    font-size: 17px;

    font-weight: bold;

    cursor: pointer;

}


#sendTextButton:active {

    transform: scale(0.98);

}


#sendTextButton:disabled {

    background: #4b5563;

    cursor: not-allowed;

}


/* =========================================================
   로그
   ========================================================= */

#log {

    margin-top: 15px;

    padding: 15px;

    background: #111827;

    border-radius: 10px;

    min-height: 50px;

    font-size: 14px;

    color: #d1d5db;

}


</style>

</head>


<body>


<div class="container">


<h1>🚁 Drone AI Control</h1>


<!-- =====================================================
     서버 상태
     ====================================================== -->

<div
    id="serverStatus"
    class="status"
>

서버 연결 확인 중...

</div>


<!-- =====================================================
     음성 입력
     ====================================================== -->

<div class="card">

<div class="card-title">

🎤 음성 명령

</div>


<button
    id="micButton"
    type="button"
>

🎤

</button>


<div id="voiceResult">

마이크 버튼을 눌러 명령하세요.

</div>

</div>


<!-- =====================================================
     텍스트 입력
     ====================================================== -->

<div class="card">

<div class="card-title">

⌨️ 텍스트 명령

</div>


<textarea
    id="textCommand"
    placeholder="예: 드론을 3미터 높이로 띄워줘"
></textarea>


<button
    id="sendTextButton"
    type="button"
>

텍스트 명령 전송

</button>


</div>


<!-- =====================================================
     로그
     ====================================================== -->

<div class="card">

<div class="card-title">

📡 전송 상태

</div>


<div id="log">

대기 중...

</div>

</div>


</div>


<script>


// ============================================================
// DOM
// ============================================================

const serverStatus =
    document.getElementById("serverStatus");

const micButton =
    document.getElementById("micButton");

const voiceResult =
    document.getElementById("voiceResult");

const textCommand =
    document.getElementById("textCommand");

const sendTextButton =
    document.getElementById("sendTextButton");

const log =
    document.getElementById("log");


// ============================================================
// 로그
// ============================================================

function setLog(message) {

    console.log(message);

    log.textContent = message;

}


// ============================================================
// 서버 상태 확인
// ============================================================

async function checkServer() {

    try {

        setLog("서버 상태 확인 중...");

        const response =
            await fetch(
                "/health",
                {
                    method: "GET",
                    cache: "no-store"
                }
            );


        if (!response.ok) {

            throw new Error(
                "HTTP " + response.status
            );

        }


        const data =
            await response.json();


        console.log(
            "Health:",
            data
        );


        if (data.status === "ok") {

            serverStatus.textContent =
                "● 서버 연결됨";

            serverStatus.className =
                "status connected";

            setLog(
                "서버 연결 완료"
            );

        } else {

            throw new Error(
                "서버 상태가 정상적이지 않습니다."
            );

        }


    } catch (error) {

        console.error(
            "Health check error:",
            error
        );


        serverStatus.textContent =
            "● 서버 연결 실패";

        serverStatus.className =
            "status error";


        setLog(
            "서버 연결 실패: " +
            error.message
        );

    }

}


// ============================================================
// ROS2 명령 전송
// ============================================================

async function sendCommand(text) {

    text = text.trim();


    if (!text) {

        setLog(
            "명령을 입력해주세요."
        );

        return false;

    }


    setLog(
        "명령 전송 중: " + text
    );


    try {

        const response =
            await fetch(
                "/voice_command",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })
                }
            );


        console.log(
            "HTTP status:",
            response.status
        );


        const data =
            await response.json();


        console.log(
            "Response:",
            data
        );


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "HTTP " + response.status
            );

        }


        setLog(
            "✓ 명령 전송 완료: " +
            text
        );


        return true;


    } catch (error) {

        console.error(
            "Command send error:",
            error
        );


        setLog(
            "✗ 명령 전송 실패: " +
            error.message
        );


        return false;

    }

}


// ============================================================
// 텍스트 명령
// ============================================================

sendTextButton.addEventListener(
    "click",
    async function() {

        const text =
            textCommand.value;


        if (!text.trim()) {

            setLog(
                "텍스트 명령을 입력해주세요."
            );

            textCommand.focus();

            return;

        }


        sendTextButton.disabled = true;


        const success =
            await sendCommand(text);


        if (success) {

            textCommand.value = "";

        }


        sendTextButton.disabled = false;

    }
);


// ============================================================
// Ctrl + Enter
// ============================================================

textCommand.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter" &&
            event.ctrlKey
        ) {

            event.preventDefault();

            sendTextButton.click();

        }

    }
);


// ============================================================
// Web Speech API
// ============================================================

let recognition = null;

let isRecording = false;


const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;


if (!SpeechRecognition) {

    console.warn(
        "Web Speech API를 지원하지 않는 브라우저입니다."
    );


    micButton.disabled = true;

    micButton.textContent = "🚫";


    voiceResult.textContent =
        "이 브라우저에서는 음성 인식을 지원하지 않습니다.";

} else {


    recognition =
        new SpeechRecognition();


    recognition.lang =
        "ko-KR";


    recognition.continuous =
        false;


    recognition.interimResults =
        true;


    recognition.maxAlternatives =
        1;


    // --------------------------------------------------------
    // 음성 인식 시작
    // --------------------------------------------------------

    recognition.onstart = function() {

        isRecording = true;

        micButton.classList.add(
            "recording"
        );

        micButton.textContent =
            "🔴";


        voiceResult.textContent =
            "듣고 있습니다...";

        setLog(
            "음성 입력 중..."
        );

    };


    // --------------------------------------------------------
    // 음성 결과
    // --------------------------------------------------------

    recognition.onresult = function(event) {

        let finalText = "";

        let interimText = "";


        for (
            let i = event.resultIndex;
            i < event.results.length;
            i++
        ) {

            const transcript =
                event.results[i][0].transcript;


            if (
                event.results[i].isFinal
            ) {

                finalText += transcript;

            } else {

                interimText += transcript;

            }

        }


        const displayText =
            finalText || interimText;


        voiceResult.textContent =
            displayText;


        // 최종 결과가 나오면 ROS2로 전송

        if (finalText.trim()) {

            sendCommand(finalText);

        }

    };


    // --------------------------------------------------------
    // 음성 인식 종료
    // --------------------------------------------------------

    recognition.onend = function() {

        isRecording = false;

        micButton.classList.remove(
            "recording"
        );

        micButton.textContent =
            "🎤";

    };


    // --------------------------------------------------------
    // 음성 인식 오류
    // --------------------------------------------------------

    recognition.onerror = function(event) {

        console.error(
            "Speech recognition error:",
            event.error
        );


        isRecording = false;

        micButton.classList.remove(
            "recording"
        );

        micButton.textContent =
            "🎤";


        voiceResult.textContent =
            "음성 인식 오류: " +
            event.error;


        setLog(
            "음성 인식 오류: " +
            event.error
        );

    };


    // --------------------------------------------------------
    // 마이크 버튼
    // --------------------------------------------------------

    micButton.addEventListener(
        "click",
        function() {

            if (!recognition) {

                return;

            }


            if (isRecording) {

                recognition.stop();

                return;

            }


            try {

                recognition.start();

            } catch (error) {

                console.error(
                    error
                );

                setLog(
                    "마이크 시작 실패: " +
                    error.message
                );

            }

        }
    );

}


// ============================================================
// 페이지 시작
// ============================================================

window.addEventListener(
    "load",
    function() {

        console.log(
            "Drone AI Control UI loaded"
        );


        checkServer();

    }
);


</script>


</body>

</html>
"""


# ============================================================
# FastAPI Routes
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def root():

    return HTML_PAGE


@app.get(
    "/voice",
    response_class=HTMLResponse
)
async def voice():

    return HTML_PAGE


@app.get("/health")
async def health():

    if ros_node is None:

        return {
            "status": "error",
            "ros2": False,
            "topic": DEFAULT_TOPIC
        }


    return {
        "status": "ok",
        "ros2": True,
        "topic": ros_node.topic_name
    }


@app.post("/voice_command")
async def voice_command(
    command: VoiceCommand
):

    if ros_node is None:

        return {
            "status": "error",
            "message": "ROS2 node is not running"
        }


    text = command.text.strip()


    if not text:

        return {
            "status": "error",
            "message": "Empty command"
        }


    ros_node.publish_voice_command(
        text
    )


    return {
        "status": "ok",
        "text": text,
        "topic": ros_node.topic_name
    }


# ============================================================
# ROS2 Spin
# ============================================================

def ros_spin():

    global ros_node

    try:

        rclpy.spin(
            ros_node
        )

    except Exception as e:

        print(
            f"[ROS2 ERROR] {e}"
        )

def get_local_ip():
    """
    Jetson이 현재 사용 중인 LAN/Wi-Fi의 IP 주소를 가져온다.
    인터넷 통신을 실제로 수행하지 않고,
    라우팅 정보를 이용해서 로컬 IP를 확인한다.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]

    except OSError:
        return "127.0.0.1"
# ============================================================
# Main
# ============================================================

def main():

    global ros_node


    print("=" * 60)
    print("Drone Smartphone Bridge")
    print("=" * 60)


    rclpy.init()


    ros_node = SmartphoneBridge()


    ros_thread = threading.Thread(
        target=ros_spin,
        daemon=True
    )

    ros_thread.start()

    # Jetson 실제 LAN IP 확인
    local_ip = get_local_ip()

    local_url = f"http://{local_ip}:{PORT}"
    localhost_url = f"http://127.0.0.1:{PORT}"


    print()
    print(f"Web server : http://0.0.0.0:{PORT}")
    print(f"ROS2 topic : {ros_node.topic_name}")
    print()
    print(f"📱 Smartphone URL : {local_url}")
    print("=" * 60)
    print()
    print("Waiting for smartphone connection...")

    # --------------------------------------------------------
    # 브라우저 자동 실행
    # --------------------------------------------------------

    def open_browser():
        import time

        time.sleep(1)

        print(f"[WEB] Opening browser: {localhost_url}")

        try:
            webbrowser.open(localhost_url)
        except Exception as e:
            print(f"[WEB] Browser open failed: {e}")

    browser_thread = threading.Thread(
        target=open_browser,
        daemon=True
    )
    browser_thread.start()


    # --------------------------------------------------------
    # FastAPI 실행
    # --------------------------------------------------------

    try:

        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            log_level="info"
        )

    except KeyboardInterrupt:

        print("\nStopping...")

    finally:

        if ros_node is not None:

            ros_node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()