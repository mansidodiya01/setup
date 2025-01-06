import threading
import time
from timestamp import shared_timestamp
from frame import main as frame_main
from audioinput import run as audio_main

def synchronized_frame_capture(sync_event):
    """Thread for synchronized frame capture."""
    print("Starting synchronized frame capture...")
    while True:
        sync_event.wait()  # Wait for the signal to capture frame
        sync_event.clear()  # Reset the event
        shared_timestamp.update_timestamp()  # Update the shared timestamp
        frame_main()  # Perform frame inference

def synchronized_audio_input(sync_event):
    """Thread for synchronized audio input."""
    print("Starting synchronized audio input...")
    while True:
        sync_event.wait()  # Wait for the signal to record audio
        sync_event.clear()  # Reset the event
        audio_main(model="yamnet.tflite", max_results=5, score_threshold=0.3)  # Perform audio inference

def main():
    # Create an event for synchronization
    sync_event = threading.Event()

    # Start the synchronized frame capture and audio input threads
    frame_thread = threading.Thread(target=synchronized_frame_capture, args=(sync_event,))
    audio_thread = threading.Thread(target=synchronized_audio_input, args=(sync_event,))

    frame_thread.start()
    audio_thread.start()

    print("Starting synchronization loop...")
    try:
        while True:
            start_time = time.time()  # Record the start time

            # Signal both threads to execute
            sync_event.set()

            # Wait for processing time of both threads
            while sync_event.is_set():
                time.sleep(0.01)  # Short sleep to avoid busy-waiting

            end_time = time.time()  # Record the end time

            # Calculate the processing duration
            processing_duration = end_time - start_time

            # Target interval (adjustable for desired frequency)
            target_interval = 0.5  # Desired interval in seconds

            # Sleep to maintain the target interval
            sleep_time = max(0, target_interval - processing_duration)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping synchronization...")
        frame_thread.join()
        audio_thread.join()

if __name__ == "__main__":
    main()
