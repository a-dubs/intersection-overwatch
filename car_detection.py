import dataclasses
import enum
from pprint import pprint
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
CARS_IN_QUADRILATERALS = {}  # map label to detection

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

# def get_cars_in_quadrilaterals(detections: Detections) -> list:
#     cars = []
#     for i in range(len(detections)):
#         detection = detections[i]
#         x1, y1, x2, y2 = map(int, detection[0].xyxy[0])
#         car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
#         for quad in quadrilaterals:
#             if cv2.pointPolygonTest(np.array(quad, dtype=np.int32), car_center, False) >= 0:
#                 cars.append(detection)
#                 break
#     print("Cars in quadrilaterals:", cars)
#     return cars

def get_car_label(detection: Detections) -> str:
    class_name = detection[-1]["class_name"][0]
    tracker_id = detection.tracker_id[0]
    return format_label(class_name, tracker_id)

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
class CarDetection:
    xyxy: tuple
    class_name: str
    tracker_id: int

    @classmethod
    def from_detections(cls, detection: Detections, index: int):
        class_name = detection[index]["class_name"][0]
        if class_name not in ("car", "truck"):
            # print(f"Skipping detection of class {class_name}.")
            return None
        return cls(
            xyxy=detection.xyxy[index],
            class_name=class_name,
            tracker_id=detection.tracker_id[index],
        )
    
    @property
    def label(self):
        return format_label(self.class_name, self.tracker_id)
    
    @property
    def is_in_quadrilateral(self):
        x1, y1, x2, y2 = map(int, self.xyxy)
        car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        for quad in quadrilaterals:
            if cv2.pointPolygonTest(np.array(quad, dtype=np.int32), car_center, False) >= 0:
                return True
        return False
    
@dataclasses.dataclass
class CarAnnotation:
    xyxy: tuple
    color: tuple = ANN_COLORS.WHITE.value
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
        annotations: list["CarAnnotation"]
    ):
        for annotation in annotations:
            annotation.draw(frame)

    @classmethod
    def from_car_detection(cls, detection: CarDetection):
        return cls(
            xyxy=detection.xyxy,
            color=get_car_annotation_color(detection),
            label=detection.label
        )

def update_car_time_in_quadrilaterals(car_detections: list[CarDetection]):
    global CARS_TIME_IN_QUADRILATERALS
    try:
        for car_detection in car_detections:
            if not car_detection.is_in_quadrilateral:
                continue
            car_label = car_detection.label
            if car_label not in CARS_TIME_IN_QUADRILATERALS:
                CARS_TIME_IN_QUADRILATERALS[car_label] = 0
            CARS_TIME_IN_QUADRILATERALS[car_label] += 1 / FPS
        car_labels = [detection.label for detection in car_detections]
        for car_label in list(CARS_TIME_IN_QUADRILATERALS.keys()):
            if car_label not in car_labels:
                CARS_TIME_IN_QUADRILATERALS[car_label] = 0
        # delete any cars with time 0
        CARS_TIME_IN_QUADRILATERALS = {car_label: time for car_label, time in CARS_TIME_IN_QUADRILATERALS.items() if time > 0}
    except Exception as e:
        raise e
    pprint(CARS_TIME_IN_QUADRILATERALS)

def get_car_annotation_color(cd: CarDetection) -> tuple:
    if cd.is_in_quadrilateral:
        if (
            cd.label in CARS_TIME_IN_QUADRILATERALS 
            and CARS_TIME_IN_QUADRILATERALS[cd.label] >= TIME_IN_QUADRILATERAL_FOR_STOP
        ):
            return ANN_COLORS.GREEN.value
        return ANN_COLORS.RED.value
    return ANN_COLORS.WHITE.value


def parse_car_annotations_from_detections(car_detections: list[CarDetection]) -> list[CarAnnotation]:
    annotations = []
    for car_detection in car_detections:
        annotation = CarAnnotation.from_car_detection(car_detection)
        annotations.append(annotation)
    return annotations

def parse_car_detections(detections: Detections) -> list[CarDetection]:
    car_detections = []
    for i in range(len(detections)):
        detection = CarDetection.from_detections(detections, i)
        if detection:
            car_detections.append(detection)
    return car_detections

def draw_car_annotations(
    frame: np.ndarray,
    car_annotations: list[CarAnnotation],
) -> np.ndarray:
    """
    Updates the frame with given car annotations.
    """
    annotated_frame = frame.copy()

    CarAnnotation.draw_annotations(annotated_frame, car_annotations)

    return annotated_frame

# time in seconds a car must be in a quadrilateral to be considered stopped
TIME_IN_QUADRILATERAL_FOR_STOP = 1

# def log_cars_stopped_in_quadrilaterals():
#     for car_label, quadrilateral_times in CARS_TIME_IN_QUADRILATERALS.items():
#         for i, time in quadrilateral_times.items():
#             if time >= TIME_IN_QUADRILATERAL_FOR_STOP:
#                 print(f"Car {car_label} stopped in quadrilateral {i} for {time:.2f} seconds.")

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
            sv_detections = get_car_detections(frame)
            car_detections = parse_car_detections(sv_detections)
            update_car_time_in_quadrilaterals(car_detections)
            car_annotations = parse_car_annotations_from_detections(car_detections)
            annotated_frame = draw_car_annotations(frame, car_annotations)
            draw_quadrilaterals(annotated_frame)
        cv2.imshow("Car Detection", annotated_frame)
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
