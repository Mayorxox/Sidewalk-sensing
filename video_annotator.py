import cv2
import json
import os

annotations = []
current_bbox = []
drawing = False

def draw_bbox(event, x, y, flags, param):
    global current_bbox, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_bbox = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        frame_copy = param.copy()
        cv2.rectangle(frame_copy, current_bbox[0], (x, y), (0, 255, 0), 2)
        cv2.imshow('Annotate', frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_bbox.append((x, y))
        cv2.rectangle(param, current_bbox[0], current_bbox[1], (0, 255, 0), 2)
        cv2.imshow('Annotate', param)

def annotate_video(video_path, output_path, clip_id):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_copy = frame.copy()
        cv2.putText(frame_copy, f"Frame {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Annotate', frame_copy)
        cv2.setMouseCallback('Annotate', draw_bbox, frame_copy)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):  # save bbox annotation
            if len(current_bbox) == 2:
                x1, y1 = current_bbox[0]
                x2, y2 = current_bbox[1]
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                print("\nEvent info:")
                gaze_shift = input("Gaze shift (1/0): ") == '1'
                hesitation = input("Hesitation (1/0): ") == '1'
                trajectory_change = input("Trajectory change (major/minor): ")
                adjustment_type = input("Adjustment type (sidestep/pause/none/retreat): ")

                annotations.append({
                    "clip_id": clip_id,
                    "frame_index": frame_idx,
                    "bbox": bbox,
                    "gaze_shift": gaze_shift,
                    "hesitation": hesitation,
                    "trajectory_change": trajectory_change,
                    "adjustment_type": adjustment_type
                })
                print("Annotation saved!\n")
            else:
                print("No valid bounding box drawn.")
        elif key == ord('n'):  # next frame
            frame_idx += 1
        elif key == ord('q'):  # quit
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"Saved annotations to {output_path}")

if __name__ == "__main__":
    annotate_video("../data/clips/clip_01.mp4", "../data/annotations/clip_01_annotations.json", "clip_01")
