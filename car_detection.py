import dataclasses
import enum
from typing import Optional
import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

# Constants
BOX_COLOR = (0, 0, 255)  # Red color in BGR
HIGHLIGHT_COLOR = (0, 255, 0)  # Green color for selected points
HIT_BOX_COLOR = (255, 255, 255)  # White color for detected objects
INTERSECT_COLOR = (0, 0, 255)  # Red color for intersections
BOX_OPACITY = 0.4
POINT_RADIUS = 5
SELECT_RADIUS = 50  # Radius around points for selection
JSON_FILE = "quadrilaterals.json"

FPS = None

# map car id to a dict containg a key for each quadrilateral and the time spent in that quadrilateral
CARS_TIME_IN_QUADRILATERALS = {}

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load existing quadrilaterals or create a new structure
if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r") as f:
        quadrilaterals = json.load(f).get("quadrilaterals", [])
else:
    quadrilaterals = []

def draw_quadrilaterals(frame):
    """Draw the quadrilaterals on the frame."""
    overlay = frame.copy()
    for quad in quadrilaterals:
        pts = np.array(quad, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], BOX_COLOR)
    cv2.addWeighted(overlay, BOX_OPACITY, frame, 1 - BOX_OPACITY, 0, frame)
    for quad in quadrilaterals:
        pts = np.array(quad, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=BOX_COLOR, thickness=2)
        for point in quad:
            cv2.circle(frame, (point[0], point[1]), POINT_RADIUS, (255, 255, 255), -1)

from ultralytics import YOLO
from supervision.detection.core import Detections
import supervision as sv


# Initialize tracker
tracker = sv.ByteTrack()

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def format_label(class_name: str, tracker_id: int) -> str:
    return f"{class_name} #{tracker_id}".capitalize()

def get_car_detections(frame: np.ndarray) -> Detections:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections: Detections = tracker.update_with_detections(detections)
    return detections

def get_cars_in_quadrilaterals(detections: Detections) -> list:
    cars = []
    for i in range(len(detections)):
        detection = detections[i]
        x1, y1, x2, y2 = map(int, detection[0].xyxy[0])
        car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        for quad in quadrilaterals:
            if cv2.pointPolygonTest(np.array(quad, dtype=np.int32), car_center, False) >= 0:
                cars.append(detection)
                break
    print("Cars in quadrilaterals:", cars)
    return cars

def get_car_label(detection: Detections) -> str:
    class_name = detection[-1]["class_name"][0]
    tracker_id = detection.tracker_id[0]
    return format_label(class_name, tracker_id)

def update_car_time_in_quadrilaterals(cars_in_quadrilaterals: list[Detections]):
    global CARS_TIME_IN_QUADRILATERALS
    try:
        for detection in cars_in_quadrilaterals:
            car_label = get_car_label(detection)
            if car_label not in CARS_TIME_IN_QUADRILATERALS:
                CARS_TIME_IN_QUADRILATERALS[car_label] = 0
            CARS_TIME_IN_QUADRILATERALS[car_label] += 1 / FPS
        car_labels = [get_car_label(detection) for detection in cars_in_quadrilaterals]
        for car_label in CARS_TIME_IN_QUADRILATERALS:
            if car_label not in car_labels:
                CARS_TIME_IN_QUADRILATERALS[car_label] = 0
        # delete any cars with time 0
        CARS_TIME_IN_QUADRILATERALS = {car_label: time for car_label, time in CARS_TIME_IN_QUADRILATERALS.items() if time > 0}
    except Exception as e:
        raise e

def remove_detection(detections: Detections, index: int) -> Detections:
    return Detections(
        xyxy=np.delete(detections.xyxy, index, axis=0),
        confidence=np.delete(detections.confidence, index, axis=0),
        class_id=np.delete(detections.class_id, index, axis=0),
        tracker_id=np.delete(detections.tracker_id, index, axis=0),
        mask=detections.mask,
        data={key: np.delete(value, index, axis=0) for key, value in detections.data.items()},
        metadata={key: value for key, value in detections.metadata.items()}
    )

def draw_car_detections(frame: np.ndarray, detections: Detections):
    cars_in_quadrilaterals = get_cars_in_quadrilaterals(detections)
    update_car_time_in_quadrilaterals(cars_in_quadrilaterals)

    # kept_detections_ids = []
    # kept_detections = detections

    car_in_quad_color = ANN_COLORS.RED.value
    normal_car_color = ANN_COLORS.WHITE.value
    car_stopped_color = ANN_COLORS.GREEN.value
    
    annotations: list[object_annotation] = []

    for i in range(len(detections)):
        detection = detections[i]
        class_name = detection[-1]["class_name"][0]
        if class_name not in ["car", "truck"]:
            continue
        tracker_id = detections.tracker_id[i]
        label = format_label(class_name, tracker_id)
        xyxy = detection[0].xyxy[0]
        # print(f"Car {label} detected at {xyxy}")
        color = normal_car_color
        try:
            if detection in cars_in_quadrilaterals:
                color = car_in_quad_color
            if label in CARS_TIME_IN_QUADRILATERALS and CARS_TIME_IN_QUADRILATERALS[label] >= TIME_IN_QUADRILATERAL_FOR_STOP:
                color = car_stopped_color
        except Exception as e:
            print(e)
            print(detection)
            print(cars_in_quadrilaterals)
            raise e
        annotations.append(object_annotation(label=label, xyxy=xyxy, color=color))

    annotated_frame = frame.copy()

    # annotated_frame = box_annotator.annotate(
    #     scene=annotated_frame, detections=detections)
    # annotated_frame = label_annotator.annotate(
    #     scene=annotated_frame, detections=detections, labels=labels)
    # halo_annotator = sv.HaloAnnotator(opacity=1)
    # annotated_frame = halo_annotator.annotate(
    #     scene=annotated_frame,
    #     detections=detections,
    # )

    object_annotation.draw_annotations(annotated_frame, annotations)

    return annotated_frame

# time in seconds a car must be in a quadrilateral to be considered stopped
TIME_IN_QUADRILATERAL_FOR_STOP = 2

def log_cars_stopped_in_quadrilaterals():
    for car_label, quadrilateral_times in CARS_TIME_IN_QUADRILATERALS.items():
        for i, time in quadrilateral_times.items():
            if time >= TIME_IN_QUADRILATERAL_FOR_STOP:
                print(f"Car {car_label} stopped in quadrilateral {i} for {time:.2f} seconds.")

@enum.unique
class ANN_COLORS(enum.Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    PURPLE = (255, 0, 255)
    CYAN = (255, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

@dataclasses.dataclass
class object_annotation:
    xyxy: tuple
    color: tuple
    label: Optional[str] = None

    # function that takes in a frame and draws the object annotation on the frame
    def draw(self, frame):
        x1, y1, x2, y2 = map(int, self.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)
        if self.label:
            cv2.putText(
                img=frame,
                text=self.label,
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=self.color,
                thickness=2
            )

    @classmethod
    def draw_annotations(
        cls,
        frame,
        annotations: list["object_annotation"]
    ):
        for annotation in annotations:
            annotation.draw(frame)
    


def detect_and_track_cars(frame: np.ndarray) -> np.ndarray:
    detections = get_car_detections(frame)
    return draw_car_detections(frame, detections)


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    paused = False
    # update the global FPS variable
    global FPS
    FPS = cap.get(cv2.CAP_PROP_FPS)
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame = detect_and_track_cars(frame)
            draw_quadrilaterals(frame)
        cv2.imshow("Car Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord(','):
            print("Skipping back 10 seconds.")
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0, cap.get(cv2.CAP_PROP_POS_MSEC) - 10000))
        elif key == ord('.'):
            print("Skipping forward 10 seconds.")
            cap.set(cv2.CAP_PROP_POS_MSEC, min(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) * 1000, cap.get(cv2.CAP_PROP_POS_MSEC) + 10000))
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    video_path = os.environ.get("VIDEO_PATH")
    if video_path is None:
        print("Error: VIDEO_PATH environment variable not set.")
        exit(1)
    else:
        main(video_path)
