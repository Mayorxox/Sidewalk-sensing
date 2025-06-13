import json
from heuristic_detector import detect_yielding_event, PARAMS

# Load sample annotation format
with open("example_annotations.json", "r") as f:
    data = json.load(f)

for clip_id, clip_data in data.items():
    print(f"Processing {clip_id}...")
    events = detect_yielding_event(
        trajectory=clip_data["trajectory"],
        speed=clip_data["speed"],
        gaze_angle=clip_data["gaze"],
        params=PARAMS
    )
    print(f"Detected {sum(events)} yielding events.")
