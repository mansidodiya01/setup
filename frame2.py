import cv2
from ultralytics import YOLO
import os
import time
from timestamp import shared_timestamp

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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while True:
        # Update and get the shared timestamp
        shared_timestamp.update_timestamp()
        timestamp = shared_timestamp.get_timestamp()

        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from the camera.")
            break

        # Run inference
        results = ncnn_model(frame)

        # Save the original frame resized to 224x224
        resized_original = cv2.resize(frame, (224, 224))
        original_image_path = os.path.join(original_dir, f"original_{int(timestamp)}.jpg")
        cv2.imwrite(original_image_path, resized_original)

        # Find the bounding box with the highest confidence score
        best_box, best_score = None, 0
        for result in results:
            boxes = result.boxes.xyxy.numpy()
            scores = result.boxes.conf.numpy()
            for i, box in enumerate(boxes):
                if scores[i] > best_score:
                    best_box, best_score = box, scores[i]

        # Crop and save the image
        if best_box is not None:
            x_min, y_min, x_max, y_max = map(int, best_box)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)
            cropped_image = frame[y_min:y_max, x_min:x_max]
            resized_cropped = cv2.resize(cropped_image, (224, 224))
            cropped_image_path = os.path.join(cropped_dir, f"cropped_{int(timestamp)}.jpg")
            cv2.imwrite(cropped_image_path, resized_cropped)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
