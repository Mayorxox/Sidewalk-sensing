# Opportunistic HRI Detection

This repository implements a simple rule-based heuristic model for detecting pedestrian path-yielding behavior in video data captured by autonomous delivery robots (ADRs). The model operates on annotated pedestrian trajectories and infers yielding events based on lateral deviation, hesitation, and gaze direction.

# ADR Yield Detection

This repository demonstrates a simple, interpretable framework for detecting pedestrian path-yielding behavior using heuristic rules and lightweight ML models. It also includes a temporal window sensitivity analysis for optimizing real-time event detection.

### Contents
- `src/ml_detection.py`: Decision Tree model for yielding event classification.
- `src/temporal_window_analysis.py`: Compares precision/recall/F1 across time windows.
- `results/`: Stores generated plots and confusion matrices.



## Components
- `heuristic_detector.py`: Core logic for detecting path-yielding events.
- `example_annotations.json`: Sample format for annotations.
- `run_detection.py`: Script to apply detection on annotated video metadata.

## Usage
1. Place your annotated JSON files in the `data/` folder.
2. Run the script: `python run_detection.py`
3. Outputs will include event timestamps and detection results per clip.
4. Run python ml_detection.py
5. Run python temporal_window_analysis.py

## Requirements
- Python 3.7+
- NumPy
- JSON
- pip install pandas scikit-learn matplotlib numpy

## License
See [LICENSE](./LICENSE) for details.
