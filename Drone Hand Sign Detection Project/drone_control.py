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
            time.sleep(1)  # Wait a bit before retrying
    return False

def main():
    # Initialize the DJI Tello drone
    tello = Tello()
    tello.connect()
    tello.streamon()

    try:
        tello.takeoff()  # Takeoff the drone
        while True:
            time.sleep(1)  # Simulate some drone activity, e.g., hovering
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if not safe_land(tello):
            print("Failed to land the drone properly.")
        tello.streamoff()

if __name__ == '__main__':
    main()
