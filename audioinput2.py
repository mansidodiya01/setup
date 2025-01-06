import time
from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio
from timestamp import shared_timestamp

def run(model: str, max_results: int, score_threshold: float) -> None:
    def save_result(result: audio.AudioClassifierResult, timestamp_ms: int):
        detected = False
        for category in result.classifications[0].categories:
            if ("baby cry" in category.category_name.lower() or
                "infant cry" in category.category_name.lower()) and category.score > score_threshold:
                print(f"[{int(shared_timestamp.get_timestamp())}] Baby Cry Detected: {category.category_name}, Confidence: {category.score:.2f}")
                detected = True
        if not detected:
            print(f"[{int(shared_timestamp.get_timestamp())}] No baby crying detected.")

    base_options = python.BaseOptions(model_asset_path=model)
    options = audio.AudioClassifierOptions(
        base_options=base_options,
        running_mode=audio.RunningMode.AUDIO_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result,
    )
    classifier = audio.AudioClassifier.create_from_options(options)

    buffer_size, sample_rate = 15600, 16000
    record = audio_record.AudioRecord(1, sample_rate, buffer_size)
    audio_data = containers.AudioData(buffer_size, containers.AudioDataFormat(1, sample_rate))
    record.start_recording()

    while True:
        try:
            shared_timestamp.update_timestamp()
            timestamp = shared_timestamp.get_timestamp()

            data = record.read(buffer_size)
            audio_data.load_from_array(data)
            classifier.classify_async(audio_data, int(timestamp * 1000))
            time.sleep(buffer_size / sample_rate)
        except KeyboardInterrupt:
            break

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yamnet.tflite')
    parser.add_argument('--maxResults', default=5)
    parser.add_argument('--scoreThreshold', default=0.3)
    args = parser.parse_args()
    run(args.model, int(args.maxResults), float(args.scoreThreshold))

if __name__ == "__main__":
    main()
