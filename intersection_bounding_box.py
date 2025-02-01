import cv2
import json
import os
import numpy as np

# Constants
BOX_COLOR = (0, 0, 255)  # Red color in BGR
BOX_OPACITY = 0.4
POINT_RADIUS = 5
SELECT_RADIUS = 50  # Radius around points for selection
JSON_FILE = "quadrilaterals.json"

# Load existing quadrilaterals or create a new structure
if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r") as f:
        quadrilaterals = json.load(f).get("quadrilaterals", [])
else:
    quadrilaterals = []

current_quad = []
selected_point = None
edit_quad_index = None
edit_point_index = None
mode = None  # Modes: "add", "edit", "delete"


def draw_quadrilaterals(frame):
    """Draw the quadrilaterals on the frame."""
    overlay = frame.copy()
    for quad in quadrilaterals:
        pts = np.array(quad, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], BOX_COLOR)
    cv2.addWeighted(overlay, BOX_OPACITY, frame, 1 - BOX_OPACITY, 0, frame)

    # Draw borders and points
    for quad in quadrilaterals:
        pts = np.array(quad, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=BOX_COLOR, thickness=2)
        for point in quad:
            cv2.circle(frame, (point[0], point[1]), POINT_RADIUS, (255, 255, 255), -1)


def mouse_callback(event, x, y, flags, param):
    global current_quad, selected_point, edit_quad_index, edit_point_index, mode

    if mode == "add":
        if event == cv2.EVENT_LBUTTONDOWN and len(current_quad) < 4:
            current_quad.append([x, y])
    elif mode == "edit":
        if event == cv2.EVENT_LBUTTONDOWN:
            if edit_point_index is None:
                # Find the closest point within radius
                for i, quad in enumerate(quadrilaterals):
                    for j, point in enumerate(quad):
                        if abs(point[0] - x) < SELECT_RADIUS and abs(point[1] - y) < SELECT_RADIUS:
                            edit_quad_index, edit_point_index = i, j
                            return
            else:
                # Move the selected point
                quadrilaterals[edit_quad_index][edit_point_index] = [x, y]
                edit_quad_index, edit_point_index = None  # Deselect after moving
    elif mode == "delete":
        if event == cv2.EVENT_LBUTTONDBLCLK:
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
    global mode, current_quad, edit_quad_index, edit_point_index

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

        draw_quadrilaterals(frame)

        # Draw current quadrilateral in add mode
        if mode == "add" and len(current_quad) > 0:
            for point in current_quad:
                cv2.circle(frame, (point[0], point[1]), POINT_RADIUS, (255, 255, 255), -1)
            if len(current_quad) > 1:
                cv2.polylines(frame, [np.array(current_quad, dtype=np.int32)], isClosed=False, color=BOX_COLOR, thickness=2)

        # Display instructions
        instructions = [
            "Press 'n' to start adding a quadrilateral",
            "Press 'e' to enter edit mode",
            "Press 'd' to enter delete mode",
            "Press 's' to save and exit",
            "Press 'q' to exit without saving",
        ]
        if mode == "add":
            instructions += [
                "Left-click to place points (4 needed)",
                "Press 'Enter' to save, 'Esc' to cancel"
            ]
        elif mode == "edit":
            instructions += [
                "Left-click near a point to select it",
                "Left-click elsewhere to move it",
                "Press 'Enter' to save edits, 'Esc' to cancel"
            ]
        elif mode == "delete":
            instructions += [
                "Double-click inside a quadrilateral to delete it",
                "Press 'Esc' to exit delete mode"
            ]

        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Setup Mode", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("n"):
            mode = "add"
            current_quad = []
        elif key == ord("e"):
            mode = "edit"
            edit_quad_index, edit_point_index = None, None
        elif key == ord("d"):
            mode = "delete"
        elif key == ord("s"):
            save_quadrilaterals()
            print("Quadrilaterals saved.")
            break
        elif key == ord("q"):
            print("Exited without saving.")
            break
        elif key == 27:  # Esc key
            if mode == "add":
                current_quad = []
            elif mode == "edit":
                edit_quad_index, edit_point_index = None, None
            mode = None
        elif key == 13:  # Enter key
            if mode == "add" and len(current_quad) == 4:
                quadrilaterals.append(current_quad)
                current_quad = []
                mode = None
            elif mode == "edit":
                mode = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run setup mode
    video_path = os.environ.get("VIDEO_PATH")

    if video_path is None:
        print("Error: VIDEO_PATH environment variable not set.")
        exit(1)
    else:
        setup_mode(video_path)
