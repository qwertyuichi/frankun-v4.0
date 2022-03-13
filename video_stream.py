import threading

import cv2
import numpy as np
import yaml
from flask import Flask, Response, render_template
from munch import Munch

# 設定ファイルの読み込み
with open("configuration.yml", encoding="UTF-8") as file:
    cfg = Munch.fromDict(yaml.safe_load(file))

# グローバル変数の定義
streaming_image = np.zeros([], dtype=np.uint8)
app = Flask(__name__)

# Flaskの設定
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen():
    while True:
        if streaming_image.ndim == 1:
            # streaming_imageが初期値のまま
            continue
        
        _, jpeg = cv2.imencode(".jpg", streaming_image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
        )


if __name__ == "__main__":
    # カメラの準備
    capture = cv2.VideoCapture(0)

    # Flaskの起動
    threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=cfg.debug.streaming_port,
            debug=False,
        ),
        daemon=True,
    ).start()

    try:
        while True:
            _, streaming_image = capture.read()

            cv2.imshow("Result", streaming_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
