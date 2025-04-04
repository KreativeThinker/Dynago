import multiprocessing
import subprocess
import json
import requests
import sounddevice as sd
from scipy.io.wavfile import write
import logging
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("voice_control.log")],
)
logger = logging.getLogger(__name__)


class VoiceControl:
    def __init__(self, voice_queue, llm_queue):
        logger.info("Initializing VoiceControl")
        self.voice_queue = voice_queue
        self.llm_queue = llm_queue
        self.sample_rate = 16000
        self.recording_duration = 4  # seconds
        self.ollama_model = "gemma2:2b"
        self.is_running = True

        # Verify audio devices
        devices = sd.query_devices()
        logger.debug(f"Available audio devices: {devices}")
        logger.info(f"Using default input device: {sd.default.device[0]}")

        # System prompt for the LLM
        self.system_prompt = """You are a voice assistant that responds with JSON function calls.
        Available functions:
        - type(text: str): Types the specified text
        - search(query: str): Searches the web for the query
        - calculate(expression: str): Evaluates a mathematical expression
        - play: Plays media
        - pause: Pauses media
        - next: Skips to next track
        - previous: Goes to previous track
        
        Respond ONLY in this JSON format:
        {"function": "function_name", "parameters": {"param1": "value1"}}
        """

    def record_audio(self):
        """Record audio from microphone"""
        try:
            logger.debug("Starting audio recording...")
            audio = sd.rec(
                int(self.recording_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            sd.wait()
            logger.debug("Audio recording completed")
            return audio
        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            raise

    def save_audio(self, audio, filename="temp_audio.wav"):
        """Save audio to temporary file for whisper.cpp"""
        try:
            logger.debug(f"Saving audio to {filename}")
            write(filename, self.sample_rate, audio)
            logger.debug("Audio file saved successfully")
            return filename
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            raise

    def detect_wake_word(self):
        """Continuously listen for the wake word using whisper.cpp"""
        logger.info("Starting wake word detection loop")
        while self.is_running:
            try:
                # Record and save audio
                audio = self.record_audio()
                audio_file = self.save_audio(audio)

                # Verify model file exists
                model_path = Path("dynago/models/base_en.bin")
                if not model_path.exists():
                    logger.error(f"Model file not found at {model_path.absolute()}")
                    time.sleep(1)
                    continue

                # Build whisper command
                cmd = [
                    "whisper-cli",
                    audio_file,
                    "--model",
                    str(model_path),
                    "--no-prints",
                ]
                logger.debug(f"Executing command: {' '.join(cmd)}")

                # Execute whisper
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                # Log full results
                logger.debug(f"whisper-cli return code: {result.returncode}")
                logger.debug(f"whisper-cli stdout: {result.stdout.strip()}")
                if result.stderr:
                    logger.debug(f"whisper-cli stderr: {result.stderr.strip()}")

                transcription = result.stdout.lower().strip()
                logger.debug(f"Raw transcription: {transcription}")

                if "agent" in transcription.lower():
                    command = transcription.split("agent", 1)[1].strip()
                    logger.info(f"Detected command: {command}")
                    self.voice_queue.put(command)
                else:
                    logger.debug("Wake word not detected in transcription")

            except subprocess.TimeoutExpired:
                logger.warning("whisper-cli timed out after 10 seconds")
            except Exception as e:
                logger.error(f"Error in wake word detection: {e}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on errors

    def process_voice_command(self):
        """Process commands from the voice queue using Ollama"""
        logger.info("Starting voice command processing loop")
        while True:
            try:
                if not self.voice_queue.empty():
                    command = self.voice_queue.get()
                    logger.debug(f"Processing command from queue: {command}")

                    if command is None:  # Termination signal
                        logger.info("Received termination signal")
                        break

                    response = self.query_llm(command)
                    if self.validate_response(response):
                        logger.info(f"Valid LLM response received: {response}")
                        self.llm_queue.put(response)
                    else:
                        logger.warning(f"Invalid LLM response: {response}")
                else:
                    time.sleep(0.1)  # Reduce CPU usage
            except Exception as e:
                logger.error(f"Error in command processing: {e}", exc_info=True)

    def query_llm(self, prompt):
        """Query local Ollama instance"""
        try:
            logger.debug(f"Sending to LLM: {prompt}")
            start_time = time.time()

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "format": "json",
                    "stream": False,
                },
                timeout=10,
            )

            try:
                json_response = response.json()
                parsed = json.loads(json_response.get("response", "{}"))
                return parsed
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response: {response.text}")
                return {}

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM connection error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected LLM error: {e}")
            return {}

    def validate_response(self, response):
        """Validate the LLM response format"""
        try:
            if not isinstance(response, dict):
                logger.debug("Response is not a dictionary")
                return False
            if "function" not in response:
                logger.debug("Missing 'function' key in response")
                return False
            if "parameters" not in response:
                logger.debug("Missing 'parameters' key in response")
                return False
            return True
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def run(self):
        """Run both voice detection and command processing"""
        logger.info("Starting voice control system")

        wake_word_detector = multiprocessing.Process(
            target=self.detect_wake_word, daemon=True
        )
        command_processor = multiprocessing.Process(
            target=self.process_voice_command, daemon=True
        )

        logger.info("Starting processes...")
        wake_word_detector.start()
        command_processor.start()

        try:
            wake_word_detector.join()
            command_processor.join()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
        finally:
            logger.info("Voice control system stopped")
