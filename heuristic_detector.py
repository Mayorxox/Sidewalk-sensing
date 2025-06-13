import math

def detect_yielding_event(trajectory, speed, gaze_angle, params):
    """
    Inputs:
        trajectory: list of (x, y) positions over time
        speed: list of speed values over time
        gaze_angle: list of gaze angles (degrees) over time
        params: dict of thresholds
    Output:
        Boolean list of event flags over time
    """
    events = []
    baseline_x = trajectory[0][0]  # x position at t0
    for i in range(len(trajectory)):
        dx = abs(trajectory[i][0] - baseline_x)
        v = speed[i]
        theta = gaze_angle[i]

        trajectory_deviation = dx > params['x_th']
        hesitation = v < params['v_th']
        gaze_alignment = theta < params['theta_th']

        event = gaze_alignment and (trajectory_deviation or hesitation)
        events.append(event)
    return events

# Default parameters
PARAMS = {
    'x_th': 0.5,           # meters
    'v_th': 0.2,           # m/s
    'theta_th': 15.0       # degrees
}
