import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from djitellopy import Tello
import time
import logging
import threading

# Define the PyTorch model architecture (ensure it matches your trained model)
class HandSignModel(nn.Module):
    def __init__(self):
        super().__init__()
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

# Set up logging to file
handler = logging.FileHandler('tello.log')
handler.setFormatter(Tello.FORMATTER)
Tello.LOGGER.addHandler(handler)

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
    return torch.tensor([landmarks], dtype=torch.float32)

# Function to periodically send keep-alive commands
def send_keep_alive_commands(drone, interval=10):
    while True:
        time.sleep(interval)
        try:
            drone.send_control_command("command")
        except Exception as e:
            print(f"Keep-alive command failed: {e}")

# Function to move the drone based on hand signs
def perform_drone_movement(drone, hand_sign, ok_count):
    try:
        if hand_sign == "ok":
            if ok_count >= 10:
                drone.move_down(20)
                time.sleep(1)
                drone.move_up(20)
                time.sleep(1)
                ok_count = 0
        elif hand_sign == "bye":
            drone.move_left(20)
            time.sleep(1)
            drone.move_right(20)
            time.sleep(1)
        elif hand_sign == "love":
            for _ in range(3):
                drone.flip_back()
                time.sleep(3)  # Increased delay between flips
        elif hand_sign == "thanks":
            drone.flip_left()
            time.sleep(3)  # Increased delay between flips
            drone.flip_right()
            time.sleep(3)  # Increased delay between flips
    except Exception as e:
        print(f"Movement command failed: {e}")
    return ok_count

def safe_land(drone, max_retries=5):
    for attempt in range(max_retries):
        drone.send_control_command("land")
        response = drone.get_last_response()
        if response == 'ok':
            print("Successfully landed.")
            return True
        print(f"Land command failed (attempt {attempt + 1}/{max_retries}): {response}")
        time.sleep(1)  # Wait a bit before retrying
    return False

keep_alive_thread = threading.Thread(target=send_keep_alive_commands, args=(tello,), daemon=True)
keep_alive_thread.start()

ok_count = 0

try:
    tello.takeoff()  # Takeoff the drone
    tello.move_up(100)  # Move the drone up to 100 cm
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

                    # Debugging: Print the detected hand sign
                    print(f"Detected hand sign: {hand_sign}")

                    cv2.putText(frame_bgr, hand_sign, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
                    
                    if hand_sign == "ok":
                        ok_count += 1
                    else:
                        ok_count = 0

                    ok_count = perform_drone_movement(tello, hand_sign, ok_count)
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
    try:
        if not safe_land(tello):
            print("Failed to land the drone properly.")
    except Exception as e:
        print(f"Exception during landing: {e}")
    tello.streamoff()
    cv2.destroyAllWindows()
