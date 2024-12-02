import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from supervision.assets import download_assets, VideoAssets
import time
import cv2

HOME = os.getcwd()
print(HOME)

download_assets(VideoAssets.VEHICLES)
SOURCE_VIDEO_PATH = f"{HOME}/vehicles.mp4"

sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

model = YOLO("yolov8x.pt")  # Load YOLO model
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)

START = sv.Point(0, 1500)
END = sv.Point(3840, 1500)
line_zone = sv.LineZone(start=START, end=END)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=2)

byte_tracker = sv.ByteTrack()
last_frame_time = time.time()
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global last_frame_time

    # Calculate FPS
    current_time = time.time()
    fps = 1.0 / (current_time - last_frame_time)
    last_frame_time = current_time


    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]

    annotated_frame = frame.copy()
    trace_annotator = sv.TraceAnnotator(thickness=4)
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    line_zone.trigger(detections)

    text_size = cv2.getTextSize(f"FPS: {fps:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
    cv2.putText(annotated_frame,f"FPS: {fps:.2f}",(10,annotated_frame.shape[0] - 10),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0, 255, 0),2,lineType=cv2.LINE_AA)

    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)


TARGET_VIDEO_PATH = f"{HOME}/count-objects-crossing-the-line-result.mp4"

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)
