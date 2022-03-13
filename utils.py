import datetime
import random

import cv2
import numpy as np
import yaml
from matplotlib import colors
from munch import Munch

# 設定ファイルの読み込み
with open("configuration.yml", encoding="UTF-8") as file:
    cfg = Munch.fromDict(yaml.safe_load(file))


def plot_fps(fps_camera, fps_inference, image):
    # FPSの表示
    put_outlined_text("CAM:{:00d}".format(fps_camera), (10, 20), image)
    put_outlined_text("INF:{:00d}".format(fps_inference), (10, 45), image)


def plot_current_time(image):
    # 現在時刻の表示
    current_time = datetime.datetime.now()
    put_outlined_text(current_time.strftime("%H:%M:%S"), (730, 20), image)


def put_outlined_text(text, point, image):
    # アウトライン付きの文字を描写
    cv2.putText(
        image,
        text=text,
        org=point,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(255, 255, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text=text,
        org=point,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA,
    )


def plot_one_box(frame, pt1, pt2, id, confidence, distance,  line_thickness=3):
    """
    受け取ったRGBフレームにバウンディングボックスを上書きする関数

    Args:
        image: RGBフレーム
        pt1: [x1, y1]で表されるバウンディングボックスの左上頂点配列
        pt2: [x2, y2]で表されるバウンディングボックスの右下頂点配列
        id: ターゲットのクラスID
        score: ターゲットのスコア
        distance: ターゲットまでの距離 [mm]
        line_thickness: 枠線の太さ
    """

    # バウンディングボックスの頂点座標を求める
    # pt1 = (int(box[0]), int(box[1]))  # 左上頂点
    # pt2 = (int(box[2]), int(box[3]))  # 右下頂点

    # pt1 = (int(result["x1"]), int(result["y1"]))  # 左上頂点
    # pt2 = (int(result["x2"]), int(result["y2"]))  # 右下頂点

    # クラス名と表示色を取得
    if id >= 0 and id < len(cfg.nn.class_names):
        label = cfg.nn.class_names[id]
        color = (
            np.array(colors.to_rgb(
                cfg.target_profile.colors[id])) * 255.0
        )
    else:
        label = "unknown"
        color = [random.randint(0, 255) for _ in range(3)]

    # ラベルに距離を追加
    label += ": {:.2f}: {:.2f}m".format(confidence,
                                        distance / 1000.0)

    # 描画設定
    font_thickness = max(line_thickness - 1, 1)
    text_size = cv2.getTextSize(
        label, 0, fontScale=line_thickness / 3, thickness=font_thickness
    )[0]
    _pt2 = (pt1[0] + text_size[0], pt1[1] + text_size[1] + 6)

    # 枠線と文字を描写
    cv2.rectangle(
        frame, pt1, pt2, color, thickness=line_thickness, lineType=cv2.LINE_AA
    )
    cv2.rectangle(frame, pt1, _pt2, color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (pt1[0], pt1[1] + 16),
        0,
        line_thickness / 3,
        [225, 255, 255],
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )
