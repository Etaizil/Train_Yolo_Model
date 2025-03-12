# ==============================================================================
#                                   IMPORTS
# ==============================================================================

import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
import csv

# ==============================================================================
#                               ARGUMENTS PARSING
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Path to YOLO model file", required=True)
parser.add_argument("--source", help="Image source", required=True)
parser.add_argument("--thresh", help="Confidence threshold", default=0.5, type=float)
parser.add_argument("--resolution", help="Resolution WxH", default=None)
parser.add_argument("--record", help="Save video output", action="store_true")

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print(f'ERROR: Model file "{model_path}" not found.')
    sys.exit(1)

model = YOLO(model_path, task="detect")
labels = model.names

img_ext_list = [".jpg", ".jpeg", ".png", ".bmp"]
vid_ext_list = [".avi", ".mov", ".mp4", ".mkv", ".wmv"]

# ==============================================================================
#                               INPUT HANDLING
# ==============================================================================

if os.path.isdir(img_source):
    source_type = "folder"
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    source_type = (
        "image" if ext in img_ext_list else "video" if ext in vid_ext_list else None
    )
    if source_type is None:
        print(f"ERROR: Unsupported file extension {ext}.")
        sys.exit(1)
elif "usb" in img_source:
    source_type = "usb"
    cam_idx = int(img_source[3:])
elif "laptop" in img_source:
    source_type = "laptop"
    cam_idx = 0
else:
    print(f'ERROR: Invalid source "{img_source}".')
    sys.exit(1)

resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.split("x"))
        resize = True
    except ValueError:
        print("ERROR: Invalid resolution format. Use WxH (e.g. 1280x720).")
        sys.exit(1)

if record:
    if source_type not in ["video", "usb", "laptop"]:
        print("ERROR: Recording only works for video and camera sources.")
        sys.exit(1)
    if not user_res:
        print("ERROR: Specify resolution with --resolution for recording.")
        sys.exit(1)
    recorder = cv2.VideoWriter(
        "demo1.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, (resW, resH)
    )

if source_type == "image":
    imgs_list = [img_source]
elif source_type == "folder":
    imgs_list = [
        f
        for f in glob.glob(os.path.join(img_source, "*"))
        if os.path.splitext(f)[1] in img_ext_list
    ]
elif source_type in ["video", "usb", "laptop"]:
    cap = cv2.VideoCapture(
        cam_idx if source_type in ["usb", "laptop"] else img_source, cv2.CAP_DSHOW
    )
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"ERROR: Failed to open {img_source}. Camera may be busy or unavailable.")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"ERROR: Failed to read frames from {img_source}. Exiting.")
        sys.exit(1)

    if resize:
        cap.set(3, resW)
        cap.set(4, resH)

expression_stats = {"Natural": 0, "Happy": 0, "Surprised": 0}
frame_counter = 0
stats_file = "face_expression_stats.csv"
session_name = input("Enter session name: ").strip()

existing_sessions = set()
if os.path.exists(stats_file) and os.path.getsize(stats_file) > 0:
    try:
        with open(stats_file, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if row:
                    existing_sessions.add(row[0])
    except Exception as e:
        print(f"Warning: Couldn't read existing session names: {e}")

# Make session name unique if it already exists
original_name = session_name
counter = 1
while session_name in existing_sessions:
    session_name = f"{original_name}_{counter}"
    counter += 1

if original_name != session_name:
    print(f"Session name already exists. Using '{session_name}' instead.")

if not os.path.exists(stats_file):
    with open(stats_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Session Name",
                "Natural",
                "Happy",
                "Surprised",
                "Total Expressions",
                "Duration",
                "Start time",
                "End time",
            ]
        )

# Bounding box colors
bbox_colors = [
    (164, 120, 87),
    (68, 148, 228),
    (93, 97, 209),
]

start_time = time.time()

# ==============================================================================
#                               MAIN DETECTION LOOP
# ==============================================================================

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"ERROR: Unable to read frames from {img_source}. Exiting.")
            break

        if resize and frame is not None:
            frame = cv2.resize(frame, (resW, resH))

        results = model(frame, verbose=False)
        detections = results[0].boxes

        detected_classes = set()
        for detection in detections:
            classidx = int(detection.cls.item())
            classname = labels[classidx]
            detected_classes.add(classname)

        for detected_class in detected_classes:
            if detected_class in expression_stats:
                expression_stats[detected_class] += 1

        # Draw bounding boxes
        for detection in detections:
            xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
            classidx = int(detection.cls.item())
            conf = detection.conf.item()
            if conf > min_thresh:
                cv2.rectangle(
                    frame,
                    tuple(xyxy[:2]),
                    tuple(xyxy[2:]),
                    bbox_colors[classidx % 3],
                    2,
                )
                cv2.putText(
                    frame,
                    f"{labels[classidx]}: {int(conf * 100)}%",
                    (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        # Display results
        cv2.imshow("YOLO Detection Results", frame)
        if record:
            recorder.write(frame)

        key = cv2.waitKey(5 if source_type in ["video", "usb", "laptop"] else 0)
        if key in [ord("q"), ord("Q")]:
            break

except Exception as _:
    print(f"ERROR")

# ==============================================================================
#                             SAVING STATS AND CLEANUP
# ==============================================================================

finally:
    end_time = time.time()
    session_time = end_time - start_time
    session_hours = int(session_time // 3600)
    session_minutes = int((session_time % 3600) // 60)
    session_seconds = int(session_time % 60)
    print(
        f"\nSession Time: {session_hours} hours, {session_minutes} minutes, {session_seconds} seconds"
    )
    print(f"\nFinal Expression Counts: {expression_stats}")

    total_expressions = (
        expression_stats["Natural"]
        + expression_stats["Happy"]
        + expression_stats["Surprised"]
    )

    try:
        file_exists = os.path.exists(stats_file)
        is_empty = os.stat(stats_file).st_size == 0 if file_exists else True
        with open(stats_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if is_empty:
                writer.writerow(
                    [
                        "Session Name",
                        "Natural",
                        "Happy",
                        "Surprised",
                        "Total Expressions",
                        "Duration",
                        "Start time",
                        "End time",
                    ]
                )
            writer.writerow(
                [
                    session_name,
                    expression_stats["Natural"],
                    expression_stats["Happy"],
                    expression_stats["Surprised"],
                    total_expressions,
                    f"{session_hours}h {session_minutes}m {session_seconds}s",
                    time.ctime(start_time),
                    time.ctime(end_time),
                ]
            )
        print(f"\nFinal statistics saved for session '{session_name}'.\n")
    except Exception as e:
        print(f"ERROR: Failed to save statistics. {e}")

    # Cleanup
    if source_type in ["video", "usb", "laptop"]:
        cap.release()
    if record:
        recorder.release()
    cv2.destroyAllWindows()
