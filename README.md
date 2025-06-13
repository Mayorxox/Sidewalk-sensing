# Opportunistic HRI Detection

This repository implements a simple rule-based heuristic model for detecting pedestrian path-yielding behavior in video data captured by autonomous delivery robots (ADRs). The model operates on annotated pedestrian trajectories and infers yielding events based on lateral deviation, hesitation, and gaze direction.

## Components
- `heuristic_detector.py`: Core logic for detecting path-yielding events.
- `example_annotations.json`: Sample format for annotations.
- `run_detection.py`: Script to apply detection on annotated video metadata.

## Usage
1. Place your annotated JSON files in the `data/` folder.
2. Run the script: `python run_detection.py`
3. Outputs will include event timestamps and detection results per clip.

## Requirements
- Python 3.7+
- NumPy
- JSON

## License
See [LICENSE](./LICENSE) for details.
