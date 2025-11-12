import cv2
import dlib
import json
import os
import numpy as np

# === CONFIG ===
TRACKER_TYPE = "CSRT"  # Options: CSRT, KCF, MIL, etc.
OUTPUT_DIR = "../data/annotations/"
FACE_DETECTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # if available

# === INITIALIZE DETECTORS ===
try:
    predictor = dlib.shape_predictor(FACE_DETECTOR_PATH)
    face_detector = dlib.get_frontal_face_detector()
    face_estimation = True
except:
    print("[INFO] Dlib facial landmarks not found. Head orientation will use bbox aspect ratio only.")
    face_estimation = False


def estimate_head_orientation(frame, bbox):
    """Estimate yaw direction: 'left', 'right', 'forward', or 'unknown'."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if face_estimation:
        dets = face_detector(gray)
        if len(dets) > 0:
            shape = predictor(gray, dets[0])
            coords = np.array([[p.x, p.y] for p in shape.parts()])
            nose = np.mean(coords[27:36], axis=0)
            left_eye = np.mean(coords[36:42], axis=0)
            right_eye = np.mean(coords[42:48], axis=0)
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            yaw_ratio = dx / (dy + 1e-5)
            if yaw_ratio > 2.5:
                return "left"
            elif yaw_ratio < 1.5:
                return "right"
            else:
                return "forward"
    else:
        aspect = (x2 - x1) / (y2 - y1 + 1e-5)
        if aspect > 0.7 and aspect < 0.9:
            return "forward"
        elif aspect <= 0.7:
            return "left"
        elif aspect >= 0.9:
            return "right"

    return "unknown"


def annotate_video(video_path, output_file, clip_id):
    cap = cv2.VideoCapture(video_path)
    trackers = cv2.MultiTracker_create()
    annotations = []
    initialized = False
    frame_idx = 0

    print("[INFO] Press 's' to select ROIs, 'q' to quit, 'p' to pause.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_disp = frame.copy()

        if not initialized:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('s'):
                boxes = cv2.selectROIs("Annotate", frame_disp, fromCenter=False)
                for box in boxes:
                    tracker = cv2.TrackerCSRT_create()
                    trackers.add(tracker, frame, tuple(box))
                initialized = True
                print(f"[INFO] Tracking {len(boxes)} objects...")
            elif key == ord('q'):
                break
        else:
            success, boxes = trackers.update(frame)
            if success:
                for i, box in enumerate(boxes):
                    x, y, w, h = [int(v) for v in box]
                    x2, y2 = x + w, y + h
                    cv2.rectangle(frame_disp, (x, y), (x2, y2), (0, 255, 0), 2)

                    head_dir = estimate_head_orientation(frame, (x, y, x2, y2))

                    ann = {
                        "clip_id": clip_id,
                        "frame_index": frame_idx,
                        "object_id": i,
                        "bbox": [x, y, x2, y2],
                        "head_orientation": head_dir
                    }
                    annotations.append(ann)

                    cv2.putText(frame_disp, f"ID {i} | {head_dir}",
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                print("[WARN] Tracking lost.")

        cv2.imshow("Annotate", frame_disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(-1)  # pause

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"[INFO] Saved annotations to {output_file}")


if __name__ == "__main__":
    annotate_video("../data/clips/clip_01.mp4", "../data/annotations/clip_01_tracked.json", "clip_01")
