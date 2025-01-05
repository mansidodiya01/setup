import cv2
from ultralytics import YOLO
import os
import time
import subprocess


def run_cry_detection():
    """
    This function runs the cry detection script.
    Modify the path and arguments as per your environment.
    """
    subprocess.Popen(
        ["python3", "audioinput.py", "--model", "yamnet.tflite", "--maxResults", "5", "--scoreThreshold", "0.8"]
    )


def main():
    # Path to the exported NCNN model
    model_path = "best_ncnn_model"

    # Paths to save the images
    base_dir = "images"
    original_dir = os.path.join(base_dir, "original")
    cropped_dir = os.path.join(base_dir, "cropped")
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)

    # Initialize the YOLO model
    ncnn_model = YOLO(model_path, task="detect")

    # Open the camera stream
    cap = cv2.VideoCapture(0)  # Use the default camera
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Synchronization interval in seconds
    interval_between_inference = 0.35

    # Variables to track cry detection status and timer
    cry_detection_running = False
    cry_detection_timer = 0
    CRY_DETECTION_DURATION = 5  # Cry detection runs for 5 seconds after baby detection

    while True:
        # Record the start time for interval tracking
        start_time = time.time()

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        # Run inference directly on the captured frame
        results = ncnn_model(frame)

        # Save the original frame resized to 224x224
        resized_original = cv2.resize(frame, (224, 224))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        original_image_path = os.path.join(original_dir, f"original_{timestamp}.jpg")
        cv2.imwrite(original_image_path, resized_original)

        # Variables to store the bounding box with the highest confidence score
        best_box = None
        best_score = 0
        baby_detected = False

        # Find the bounding box with the highest score
        for result in results:
            boxes = result.boxes.xyxy.numpy()  # Bounding box coordinates [x_min, y_min, x_max, y_max]
            scores = result.boxes.conf.numpy()  # Confidence scores
            classes = result.boxes.cls.numpy()  # Class indices (replace with your baby class index)

            for i, cls in enumerate(classes):
                if cls == 0:  # Replace with your baby's class index
                    baby_detected = True
                    best_box = boxes[i]
                    best_score = scores[i]

        # Print baby detection status
        if baby_detected:
            print("Baby detected.")
            cry_detection_timer = time.time()
            if not cry_detection_running:
                run_cry_detection()
                cry_detection_running = True
        else:
            print("No baby detected.")

        # Stop cry detection after the set duration
        if cry_detection_running and (time.time() - cry_detection_timer > CRY_DETECTION_DURATION):
            cry_detection_running = False

        # Crop the image to the highest confidence bounding box
        if best_box is not None:
            x_min, y_min, x_max, y_max = map(int, best_box)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
            cropped_image = frame[y_min:y_max, x_min:x_max]

            if cropped_image.shape[0] >= 224 and cropped_image.shape[1] >= 224:
                resized_cropped = cv2.resize(cropped_image, (224, 224))
                cropped_image_path = os.path.join(cropped_dir, f"cropped_{timestamp}.jpg")
                cv2.imwrite(cropped_image_path, resized_cropped)

        # Calculate the time spent in processing
        elapsed_time = time.time() - start_time

        # Sleep for the remaining time in the interval
        sleep_time = max(0, interval_between_inference - elapsed_time)
        time.sleep(sleep_time)

        # Exit the loop when 'q' is pressed
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
