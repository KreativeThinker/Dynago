# Pseudocode for wake word detection
import subprocess


def listen_for_wake_word():
    while True:
        # Record audio chunk
        audio = record_audio_chunk()

        # Process with whisper.cpp
        result = subprocess.run(
            ["./whisper.cpp", "--model", "base.en", "--audio", audio],
            capture_output=True,
            text=True,
        )
        text = result.stdout.lower()

        if "dyno" in text:
            return text.split("dyno", 1)[1].strip()  # Return text after wake word
