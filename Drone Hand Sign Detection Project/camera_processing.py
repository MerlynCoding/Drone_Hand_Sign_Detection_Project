import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from djitellopy import Tello

# Define the PyTorch model architecture (ensure it matches your trained model)
class HandSignModel(nn.Module):
    def __init__(self):
        super(HandSignModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(21 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # Five classes: love, thanks, bye, ok, no_sign
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load the trained PyTorch model
model = HandSignModel()
model.load_state_dict(torch.load('hand_sign_model.pth'))
model.eval()

# Initialize the DJI Tello drone
tello = Tello()
tello.connect()
tello.streamon()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Label mapping
labels = ["love", "thanks", "bye", "ok", "no_sign"]

# Function to preprocess the frame
def preprocess_landmarks(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    landmarks = torch.tensor([landmarks], dtype=torch.float32)
    return landmarks

def main():
    try:
        while True:
            frame_read = tello.get_frame_read()
            frame = frame_read.frame

            # Check if the frame is valid
            if frame is None or frame.size == 0:
                continue

            try:
                # Convert BGR to RGB for MediaPipe processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks on the RGB frame
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
