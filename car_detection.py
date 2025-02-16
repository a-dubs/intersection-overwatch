import dataclasses
import datetime
import enum
from pprint import pprint
from typing import Optional
import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from supervision.detection.core import Detections
import supervision as sv

# Constants
BOX_COLOR = (128, 128, 255)  # Bluer grey color in BGR
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
    def is_in_quadrilateral(self) -> Optional[str]:
        x1, y1, x2, y2 = map(int, self.xyxy)
        car_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        for quad in quadrilaterals:
            if cv2.pointPolygonTest(np.array(quad.bounding_box, dtype=np.int32), car_center, False) >= 0:
                return quad.name
        return None
    
@dataclasses.dataclass
class CarAnnotation:
    xyxy: tuple
    color: tuple = ANN_COLORS.WHITE.value
    label: Optional[str] = None
    fill_color: Optional[tuple] = None

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
        if self.fill_color:
            fill_opacity = 0.4
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.fill_color, -1)
            cv2.addWeighted(overlay, fill_opacity, frame, 1 - fill_opacity, 0, frame)


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
        fill_color = None
        if (
            detection.label in CAR_DATA_DB
            and CAR_DATA_DB[detection.label].entered_through_quad
            and CAR_DATA_DB[detection.label].time_in_entered_quad >= TIME_IN_QUADRILATERAL_FOR_STOP
        ):
            fill_color = ANN_COLORS.GREEN.value
        return cls(
            xyxy=detection.xyxy,
            color=get_car_annotation_color(detection),
            label=detection.label,
            fill_color=fill_color,
        )


@dataclasses.dataclass
class CarData:
    label: str
    time_in_entered_quad: Optional[float] = None
    timestamp_first_seen: Optional[str] = None  # iso format tz aware string
    timestamp_last_seen: Optional[str] = None  # iso format tz aware string
    entered_through_quad: Optional[str] = None  # the first quadrilateral the car entered through
    exited_through_quad: Optional[str] = None  # the last quadrilateral the car exited through before leaving the frame 

    @classmethod
    def from_car_detection(cls, detection: CarDetection):
        current_quad = detection.is_in_quadrilateral
        if not current_quad:
            time_in_quad = None
        else:
            time_in_quad = CARS_TIME_IN_QUADRILATERALS[detection.label]["time"]
        return cls(
            label=detection.label,
            timestamp_first_seen=timestamp_now(),
            timestamp_last_seen=timestamp_now(),
            entered_through_quad=detection.is_in_quadrilateral,
            time_in_entered_quad=time_in_quad
        )
    
    def update(self) -> None:
        self.timestamp_last_seen = timestamp_now()
        if self.label in CARS_TIME_IN_QUADRILATERALS:
            if not self.entered_through_quad:
                self.entered_through_quad = CARS_TIME_IN_QUADRILATERALS[self.label]["quad_name"]
                self.time_in_entered_quad = CARS_TIME_IN_QUADRILATERALS[self.label]["time"]
                return
            # if the car is still in the first quadrilateral it entered through, update the time
            if self.entered_through_quad == CARS_TIME_IN_QUADRILATERALS[self.label]["quad_name"]:
                self.time_in_entered_quad += 1 / FPS
            # otherwise, use the current quadrilateral as the last quadrilateral the car exited through
            else:
                self.exited_through_quad = CARS_TIME_IN_QUADRILATERALS[self.label]["quad_name"]


CAR_DATA_DB: dict[str,CarData] = {}


@dataclasses.dataclass
class Quadrilateral:
    name: str
    bounding_box: list[list[int]]

    @classmethod
    def from_json(cls, quad: dict) -> "Quadrilateral":
        name = quad["name"]
        bounding_box = quad["bounding_box"]
        if len(bounding_box) != 4:
            raise ValueError("Bounding box must have 4 points.")
        if not all(len(point) == 2 for point in bounding_box):
            raise ValueError("Each point in the bounding box must have an x and y coordinate.")    
        return cls(
            name=name,
            bounding_box=bounding_box
        )
    
    @classmethod
    def load_all_from_json(cls, quads: list[dict]) -> list["Quadrilateral"]:
        return [cls.from_json(quad) for quad in quads]

    def to_json(self):
        return {
            "name": self.name,
            "bounding_box": self.bounding_box
        }

quadrilaterals: list[Quadrilateral] = []

# Load existing quadrilaterals or create a new structure
if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r") as f:
        quads_list = json.load(f)["quadrilaterals"]
        quadrilaterals = Quadrilateral.load_all_from_json(quads_list)


def draw_quadrilaterals(frame):
    """Draw the quadrilaterals on the frame."""
    overlay = frame.copy()
    for quad in quadrilaterals:
        pts = np.array(quad.bounding_box, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], BOX_COLOR)
        # Draw the name of the quadrilateral
        cv2.putText(
            img=overlay,
            text=quad.name,
            org=(quad.bounding_box[0][0], quad.bounding_box[0][1] - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=ANN_COLORS.WHITE.value,
            thickness=2
        )
    # Draw the quadrilaterals on the frame
    cv2.addWeighted(overlay, BOX_OPACITY, frame, 1 - BOX_OPACITY, 0, frame)
    # Draw the bounding boxes and points
    for quad in quadrilaterals:
        pts = np.array(quad.bounding_box, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=BOX_COLOR, thickness=2)
        for point in quad.bounding_box:
            cv2.circle(frame, (point[0], point[1]), POINT_RADIUS, (255, 255, 255), -1)

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

def get_car_label(detection: Detections) -> str:
    class_name = detection[-1]["class_name"][0]
    tracker_id = detection.tracker_id[0]
    return format_label(class_name, tracker_id)

def update_car_time_in_quadrilaterals(car_detections: list[CarDetection]):
    global CARS_TIME_IN_QUADRILATERALS
    try:
        for car_detection in car_detections:
            quad_name = car_detection.is_in_quadrilateral
            if not quad_name:
                continue
            car_label = car_detection.label
            # if the car is not already in the dict or is in a different quadrilateral, add it to the dict
            # with a time of 0
            if (
                car_label not in CARS_TIME_IN_QUADRILATERALS 
                or CARS_TIME_IN_QUADRILATERALS[car_label]["quad_name"] != quad_name
            ):
                CARS_TIME_IN_QUADRILATERALS[car_label] = {
                    "quad_name": quad_name,
                    "time": 0
                }
            CARS_TIME_IN_QUADRILATERALS[car_label]["time"] += 1 / FPS
        # remove any cars that are no longer actively in a quadrilateral
        car_labels = [detection.label for detection in car_detections]
        for car_label in list(CARS_TIME_IN_QUADRILATERALS.keys()):
            if car_label not in car_labels:
                CARS_TIME_IN_QUADRILATERALS[car_label]["time"] = 0
        # delete any cars with time 0
        CARS_TIME_IN_QUADRILATERALS = {
            car_label: {
                "quad_name": quad_info["quad_name"],
                "time": quad_info["time"],
            } for car_label, quad_info in CARS_TIME_IN_QUADRILATERALS.items() if quad_info["time"] > 0
        }
    except Exception as e:
        raise e
    pprint(CARS_TIME_IN_QUADRILATERALS)

def get_car_annotation_color(cd: CarDetection) -> tuple:
    if cd.is_in_quadrilateral:
        if (
            cd.label in CARS_TIME_IN_QUADRILATERALS 
            and CARS_TIME_IN_QUADRILATERALS[cd.label]["time"] >= TIME_IN_QUADRILATERAL_FOR_STOP
            # check DB to see if this is its entered quadrilateral
            and CAR_DATA_DB[cd.label].entered_through_quad == cd.is_in_quadrilateral
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

def timestamp_now() -> str:
    return datetime.datetime.now().isoformat()


def update_car_data_db(car_detections: list[CarDetection]) -> None:
    global CAR_DATA_DB
    for car_detection in car_detections:
        car_label = car_detection.label
        if car_label not in CAR_DATA_DB:
            car_data = CarData.from_car_detection(car_detection)
            CAR_DATA_DB[car_label] = car_data
        else:
            car_data = CAR_DATA_DB[car_label]
            car_data.update()

def write_car_data_db_to_json() -> None:
    car_data_list = [data.__dict__ for data in CAR_DATA_DB.values()]
    with open("car_data_db.json", "w") as f:
        json.dump(car_data_list, f, indent=4)

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
            update_car_data_db(car_detections)
            write_car_data_db_to_json()
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
