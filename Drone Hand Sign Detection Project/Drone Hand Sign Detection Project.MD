Here's a complete `README.md` based on our conversation, which outlines the purpose of each file, installation steps, and instructions for running the project.

---

# Drone Hand Sign Detection Project

This project uses a DJI Tello drone to recognize hand signs in real-time through Python scripts. The project involves controlling the drone, capturing the video feed, and processing hand signs using a neural network model with MediaPipe and PyTorch.

## Project Files

- **`drone_control.py`**: Controls the DJI Tello drone, handling takeoff, hover, and landing.
- **`camera_processing.py`**: Captures video feed from the Tello drone, uses MediaPipe to detect hand landmarks, and predicts hand signs using a pre-trained PyTorch model.
- **`runboth.bat`**: A batch file that runs both `drone_control.py` and `camera_processing.py` in parallel on Windows.
- **`requirements.txt`**: Lists all necessary dependencies for the project.
- **`hand_sign_model.pth`**: The pre-trained PyTorch model used to recognize hand signs.

## Installation

1. **Clone or download the project files** into a directory on your machine.
2. **Install the dependencies** by running the following command:

   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

The dependencies for this project are as follows:
- `djitellopy`: Library to control DJI Tello drones.
- `mediapipe`: For real-time hand tracking and hand landmark detection.
- `torch`: PyTorch for deep learning, used to load and run the hand sign recognition model.
- `opencv-python`: OpenCV for capturing and processing frames from the drone’s video feed.
- `numpy`: For array manipulation and data processing.

The `requirements.txt` file includes these libraries:

```plaintext
djitellopy
mediapipe
torch
opencv-python
numpy
```

## Project Structure

Your project directory should look like this:

```
your_project_directory/
├── drone_control.py          # Script to control the DJI Tello drone
├── camera_processing.py      # Script to process video feed and recognize hand signs
├── runboth.bat               # Batch file to run both scripts in parallel on Windows
├── requirements.txt          # Required libraries for the project
└── hand_sign_model.pth       # Pre-trained PyTorch model for hand sign recognition
```

## Usage Instructions

### Running on Windows

To run the project on Windows, use the provided `runboth.bat` file. This batch file will start both `drone_control.py` and `camera_processing.py` in separate Command Prompt windows, allowing them to operate in parallel.

1. Open Command Prompt.
2. Navigate to the directory where `runboth.bat` is located.
3. Run the following command:

   ```cmd
   runboth.bat
   ```

This will open two Command Prompt windows:
- One titled "Drone Control," which runs `drone_control.py` and handles the drone's takeoff, hovering, and landing.
- Another titled "Camera Processing," which runs `camera_processing.py` and handles video capture and hand sign detection.

### Running the Scripts Independently

If you need to run each script independently:
1. Open a Command Prompt window for each script.
2. Navigate to the project directory.
3. Run the following commands in separate Command Prompt windows:

   ```cmd
   python drone_control.py
   ```

   ```cmd
   python camera_processing.py
   ```

## Code Explanation

### `drone_control.py`

This script controls the DJI Tello drone. It connects to the drone, initiates takeoff, and hovers in place. The `safe_land` function ensures the drone lands safely by retrying the `land` command if necessary.

```python
import time
from djitellopy import Tello

def safe_land(drone, max_retries=5):
    for attempt in range(max_retries):
        drone.send_control_command("land")
        response = drone.get_last_response()
        if response == 'ok':
            print("Successfully landed.")
            return True
        else:
            print(f"Land command failed (attempt {attempt + 1}/{max_retries}): {response}")
            time.sleep(1)
    return False

def main():
    tello = Tello()
    tello.connect()
    tello.streamon()

    try:
        tello.takeoff()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if not safe_land(tello):
            print("Failed to land the drone properly.")
        tello.streamoff()

if __name__ == '__main__':
    main()
```

### `camera_processing.py`

This script captures video frames from the DJI Tello drone, detects hand landmarks using MediaPipe, and classifies hand signs using a PyTorch model.

```python
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from djitellopy import Tello

class HandSignModel(nn.Module):
    def __init__(self):
        super(HandSignModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(21 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = HandSignModel()
model.load_state_dict(torch.load('hand_sign_model.pth'))
model.eval()

tello = Tello()
tello.connect()
tello.streamon()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

labels = ["love", "thanks", "bye", "ok", "no_sign"]

def preprocess_landmarks(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    return torch.tensor([landmarks], dtype=torch.float32)

def main():
    try:
        while True:
            frame_read = tello.get_frame_read()
            frame = frame_read.frame

            if frame is None or frame.size == 0:
                continue

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = preprocess_landmarks(hand_landmarks)
                        with torch.no_grad():
                            output = model(landmarks)
                        prediction = torch.argmax(output, dim=1).item()
                        hand_sign = labels[prediction]
                        cv2.putText(frame_bgr, hand_sign, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame_bgr, "no_sign", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Hand Sign Detection', frame_bgr)
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                continue

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        tello.streamoff()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
```

## Notes

- Ensure your `hand_sign_model.pth` file is in the project directory.
- The `runboth.bat` script uses the `start` command to open separate Command Prompt windows for each script.
- To stop the scripts, close the Command Prompt windows or use `Ctrl+C` in each.

--- 

This `README.md` provides a complete overview of the project setup, files, and usage instructions.