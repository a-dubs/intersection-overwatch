import cv2
import json
import os
import numpy as np

# Constants
BOX_COLOR = (0, 0, 255)  # Red color in BGR
BOX_OPACITY = 0.4
POINT_RADIUS = 5
JSON_FILE = "quadrilaterals.json"

# Load existing quadrilaterals or create a new structure
if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r") as f:
        quadrilaterals = json.load(f).get("quadrilaterals", [])
else:
    quadrilaterals = []

current_quad = []
selected_point = None


def draw_quadrilaterals(frame):
    """Draw the quadrilaterals on the frame."""
    overlay = frame.copy()
    for quad in quadrilaterals:
        pts = np.array(quad, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], BOX_COLOR)
    cv2.addWeighted(overlay, BOX_OPACITY, frame, 1 - BOX_OPACITY, 0, frame)

    # Draw border and points for each quadrilateral
    for quad in quadrilaterals:
        pts = np.array(quad, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=BOX_COLOR, thickness=2)
        for point in quad:
            cv2.circle(frame, (point[0], point[1]), POINT_RADIUS, (255, 255, 255), -1)


def mouse_callback(event, x, y, flags, param):
    global current_quad, selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(current_quad) < 4:
            # Add a new point
            current_quad.append([x, y])
        else:
            # Check if clicking near an existing point to adjust it
            for i, point in enumerate(current_quad):
                if abs(point[0] - x) < POINT_RADIUS and abs(point[1] - y) < POINT_RADIUS:
                    selected_point = i
                    break
    elif event == cv2.EVENT_MOUSEMOVE:
        # Move the selected point
        if selected_point is not None:
            current_quad[selected_point] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right-click to finalize the current quadrilateral or remove one
        if len(current_quad) == 4:
            quadrilaterals.append(current_quad)
            current_quad = []
        else:
            # Check if inside an existing quadrilateral to remove it
            for i, quad in enumerate(quadrilaterals):
                if cv2.pointPolygonTest(np.array(quad, dtype=np.int32), (x, y), False) >= 0:
                    quadrilaterals.pop(i)
                    break


def save_quadrilaterals():
    """Save the quadrilaterals to a JSON file."""
    with open(JSON_FILE, "w") as f:
        json.dump({"quadrilaterals": quadrilaterals}, f, indent=4)


def setup_mode(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cv2.namedWindow("Setup Mode")
    cv2.setMouseCallback("Setup Mode", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Draw current and saved quadrilaterals
        draw_quadrilaterals(frame)
        if len(current_quad) > 0:
            for point in current_quad:
                cv2.circle(frame, (point[0], point[1]), POINT_RADIUS, (255, 255, 255), -1)
            if len(current_quad) > 1:
                cv2.polylines(frame, [np.array(current_quad, dtype=np.int32)], isClosed=False, color=BOX_COLOR, thickness=2)

        # Display the instructions
        instructions = [
            "Left-click to place/move a point",
            "Right-click to finalize/remove a quadrilateral",
            "Press 's' to save and exit",
            "Press 'q' to exit without saving",
        ]
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Setup Mode", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_quadrilaterals()
            print("Quadrilaterals saved.")
            break
        elif key == ord("q"):
            print("Exited without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run setup mode
    video_path = "intersection_livestream.mov"  # Replace with your video file
    setup_mode(video_path)
