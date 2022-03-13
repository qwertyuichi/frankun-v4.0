#!/usr/bin/env python3

import threading

import cv2
import depthai as dai
import numpy as np
import yaml
import sys
import utils
from munch import Munch

import video_stream as vs

# 設定ファイルの読み込み
with open("configuration.yml", encoding="UTF-8") as file:
    cfg = Munch.fromDict(yaml.safe_load(file))

# 入出力ノードの設定
pipeline = dai.Pipeline()
camera_rgb = pipeline.create(dai.node.ColorCamera)
detection_network = pipeline.create(dai.node.YoloDetectionNetwork)
xout_video = pipeline.create(dai.node.XLinkOut)
nn_out = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
nn_out.setStreamName("nn")

# カメラの設定
if cfg.camera.resolution == "1080p":
    camera_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P)
elif cfg.camera.resolution == "4k":
    camera_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_4_K)
elif cfg.camera.resolution == "12mp":
    camera_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_12_MP)
elif cfg.camera.resolution == "13mp":
    camera_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_13_MP)
else:
    print("camera→resolutionの設定値が不正です")
    sys.exit()
camera_rgb.setFps(cfg.camera.fps)
camera_rgb.setInterleaved(False)
camera_rgb.setPreviewSize(cfg.nn.input_size)
camera_rgb.setPreviewKeepAspectRatio(False)
camera_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)


# 物体検知ネットワークの設定
anchor_masks = dict(
    zip(
        ["side52", "side26", "side13"],
        [
            cfg.nn.anchor_masks.side52,
            cfg.nn.anchor_masks.side26,
            cfg.nn.anchor_masks.side13,
        ],
    )
)
detection_network.setConfidenceThreshold(cfg.nn.confidence_threshold)
detection_network.setNumClasses(cfg.nn.classes)
detection_network.setCoordinateSize(cfg.nn.coordinates)
detection_network.setAnchors(cfg.nn.anchors)
detection_network.setAnchorMasks(anchor_masks)
detection_network.setIouThreshold(cfg.nn.iou_threshold)
detection_network.setBlobPath(cfg.nn.blob_path)
detection_network.setNumInferenceThreads(2)
detection_network.input.setBlocking(False)

# XLinkの設定
camera_rgb.video.link(xout_video.input)
camera_rgb.preview.link(detection_network.input)
detection_network.out.link(nn_out.input)

# Flaskの起動
if cfg.debug.streaming_enable:
    threading.Thread(
        target=lambda: vs.app.run(
            host="0.0.0.0",
            port=cfg.debug.streaming_port,
            debug=False,
        ),
        daemon=True,
    ).start()


# デバイスに接続&pipelineの開始
with dai.Device(pipeline, usb2Mode=True) as device:

    # 画像データと推論結果を格納するためのキューを設定
    queue_frames = device.getOutputQueue(
        name="video", maxSize=4, blocking=False)
    queue_detections = device.getOutputQueue(
        name="nn", maxSize=4, blocking=False)

    while True:
        img_frame = queue_frames.get()
        img_detections = queue_detections.get()

        frame = img_frame.getCvFrame()

        if img_detections is not None:
            for detection in img_detections.detections:
                width = img_frame.getWidth()
                height = img_frame.getHeight()
                utils.plot_one_box(
                    frame=frame,
                    pt1=(int(detection.xmin*width),
                         int(detection.ymin*height)),
                    pt2=(int(detection.xmax*width),
                         int(detection.ymax*height)),
                    id=detection.label,
                    confidence=detection.confidence,
                    distance=0
                )

        # デバッグ関係
        frame_debug = cv2.resize(frame, dsize=tuple(cfg.debug.resolution))
        if cfg.debug.streaming_enable:
            vs.streaming_image = frame_debug
        if cfg.debug.display_enable:
            cv2.imshow("result", frame_debug)
            if cv2.waitKey(1) == ord("q"):
                break
