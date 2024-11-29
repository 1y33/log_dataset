import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.video.sink import VideoSink
from supervision.video.dataclasses import VideoInfo
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from supervision.draw.color import ColorPalette
from tqdm import tqdm

# Paths
SOURCE_VIDEO_PATH = "./videos/HIGHWAY_1.MOV"  # Replace with your video path
TARGET_VIDEO_PATH = "results.mp4"  # Output video path

# Line settings for counting vehicles
LINE_START = (50, 500)  # Adjust coordinates for your video
LINE_END = (1280 - 50, 500)

# Classes to detect (e.g., cars, trucks) - adjust as needed
CLASS_ID = [2, 3, 5, 7]  # Example: 2 = car, 3 = motorcycle, etc.
CLASS_NAMES_DICT = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# Initialize YOLOv8 model
model = YOLO("yolov8x.pt")  # Replace with the desired YOLOv8 model weight file

# Initialize ByteTrack
class BYTETrackerArgs:
    track_thresh = 0.25
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 3.0
    min_box_area = 1.0
    mot20 = False

byte_tracker = BYTETracker(BYTETrackerArgs())

# Load video and get info
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = cv2.VideoCapture(SOURCE_VIDEO_PATH)

# Create video writer
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # Initialize annotators
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=0.5)
    line_annotator = LineCounterAnnotator(thickness=2, text_thickness=2, text_scale=0.5)

    # Process each frame
    for _ in tqdm(range(int(video_info.total_frames))):
        ret, frame = frame_generator.read()
        if not ret:
            break

        # Model predictions
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # Filter detections for desired classes
        mask = np.array([cls_id in CLASS_ID for cls_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # Update trackers
        tracks = byte_tracker.update(
            output_results=np.array(detections.xyxy),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = [track.track_id for track in tracks]
        detections.tracker_id = np.array(tracker_id)

        # Remove detections without trackers
        mask = np.array([tid is not None for tid in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # Annotate frame
        labels = [
            f"#{tid} {CLASS_NAMES_DICT[cls]} {conf:.2f}"
            for (_, conf, cls, tid) in detections
        ]
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        # Write frame to output video
        sink.write_frame(frame)

print(f"Processed video saved at: {TARGET_VIDEO_PATH}")
