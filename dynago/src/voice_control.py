import json
import logging
import multiprocessing
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import which
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover - library availability varies by system
    sd = None  # type: ignore
    logger.warning(
        "sounddevice is not installed. Voice capture will be disabled until it is available."
    )

try:
    from scipy.io.wavfile import write  # type: ignore
except ImportError:  # pragma: no cover
    write = None  # type: ignore
    logger.warning(
        "scipy is not installed. Audio recording cannot be saved without scipy."
    )


@dataclass
class VoiceControlConfig:
    """Configuration options for the voice control agent."""

    sample_rate: int = 16_000
    wake_window_seconds: float = 2.0
    command_window_seconds: float = 5.0
    wake_word: str = "agent"
    whisper_binary: Optional[str] = None
    whisper_candidates: tuple[str, ...] = ("whisper-cli", "main")
    whisper_model_path: Path = Path("dynago/models/base_en.bin")
    transcription_timeout: float = 20.0
    audio_tmp_dir: Path = Path("dynago/tmp/voice")
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    ollama_model: str = "gemma2:2b"
    llm_timeout: float = 15.0
    channels: int = 1
    keep_recordings: bool = False
    log_level: int = logging.INFO
    system_prompt: str = (
        "You are a voice assistant that responds with JSON function calls.\n"
        "Available functions:\n"
        "- type(text: str): Types the specified text\n"
        "- search(query: str): Searches the web for the query\n"
        "- calculate(expression: str): Evaluates a mathematical expression\n"
        "- play: Plays media\n"
        "- pause: Pauses media\n"
        "- next: Skips to next track\n"
        "- previous: Goes to previous track\n\n"
        "Respond ONLY in this JSON format:\n"
        '{"function": "function_name", "parameters": {"param1": "value1"}}'
    )

    def to_payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data["whisper_model_path"] = str(self.whisper_model_path)
        data["audio_tmp_dir"] = str(self.audio_tmp_dir)
        data["whisper_candidates"] = list(self.whisper_candidates)
        return data

    @classmethod
    def from_payload(cls, payload: Optional[Dict[str, Any]] = None) -> "VoiceControlConfig":
        if not payload:
            return cls()
        data = dict(payload)
        if "whisper_model_path" in data:
            data["whisper_model_path"] = Path(data["whisper_model_path"])
        if "audio_tmp_dir" in data:
            data["audio_tmp_dir"] = Path(data["audio_tmp_dir"])
        if "whisper_candidates" in data:
            data["whisper_candidates"] = tuple(data["whisper_candidates"])
        return cls(**data)


class WhisperRunner:
    """Wrapper around whisper.cpp or compatible CLI binaries."""

    def __init__(self, config: VoiceControlConfig):
        self.config = config
        self.binary = self._resolve_binary()
        self.model_path = config.whisper_model_path.expanduser().resolve()
        self.uses_whisper_cpp = Path(self.binary).name == "main"
        if not self.model_path.exists():
            logger.warning("Whisper model not found at %s", self.model_path)
        logger.debug("Using whisper backend '%s' with model '%s'", self.binary, self.model_path)

    def _resolve_binary(self) -> str:
        candidates = []
        if self.config.whisper_binary:
            candidates.append(self.config.whisper_binary)
        candidates.extend(self.config.whisper_candidates)

        for candidate in candidates:
            candidate_path = Path(candidate)
            if candidate_path.exists() and candidate_path.is_file():
                return str(candidate_path.resolve())
            discovered = which(candidate)
            if discovered:
                return discovered
            local_candidate = Path.cwd() / candidate
            if local_candidate.exists() and local_candidate.is_file():
                return str(local_candidate.resolve())

        raise FileNotFoundError(
            "Unable to locate a whisper binary. Tried: " + ", ".join(map(str, candidates))
        )

    def transcribe(self, audio_path: Path, prompt: Optional[str] = None) -> str:
        if self.uses_whisper_cpp:
            return self._transcribe_with_whisper_cpp(audio_path, prompt)
        return self._transcribe_with_cli(audio_path, prompt)

    def _transcribe_with_cli(self, audio_path: Path, prompt: Optional[str]) -> str:
        cmd = [self.binary, str(audio_path), "--model", str(self.model_path), "--no-prints"]
        if prompt:
            cmd.extend(["--prompt", prompt])
        logger.debug("Running whisper CLI: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.transcription_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.error("Whisper CLI timed out after %.1fs", self.config.transcription_timeout)
            return ""
        except Exception as exc:
            logger.error("Whisper CLI failed: %s", exc)
            return ""

        if result.returncode != 0:
            logger.error("Whisper CLI error (code %s): %s", result.returncode, result.stderr.strip())
            return ""

        transcript = (result.stdout or "").strip()
        logger.debug("Whisper CLI transcript: %s", transcript)
        return transcript

    def _transcribe_with_whisper_cpp(self, audio_path: Path, prompt: Optional[str]) -> str:
        output_prefix = audio_path.with_suffix("")
        output_txt = output_prefix.with_suffix(".txt")
        cmd = [
            self.binary,
            "-m",
            str(self.model_path),
            "-f",
            str(audio_path),
            "-otxt",
            "-of",
            str(output_prefix),
            "-l",
            "en",
        ]
        if prompt:
            cmd.extend(["-p", prompt])
        logger.debug("Running whisper.cpp: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.transcription_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "whisper.cpp binary timed out after %.1fs", self.config.transcription_timeout
            )
            return ""
        except Exception as exc:
            logger.error("whisper.cpp invocation failed: %s", exc)
            return ""

        if result.returncode != 0:
            logger.error("whisper.cpp error (code %s): %s", result.returncode, result.stderr.strip())
            return ""

        if not output_txt.exists():
            logger.error("whisper.cpp completed but produced no output at %s", output_txt)
            return ""

        transcript = output_txt.read_text(encoding="utf-8").strip()
        logger.debug("whisper.cpp transcript: %s", transcript)
        if not self.config.keep_recordings:
            output_txt.unlink(missing_ok=True)
        return transcript


class LLMAgent:
    """Handles interaction with the local LLM via Ollama."""

    def __init__(self, config: VoiceControlConfig):
        self.endpoint = config.ollama_endpoint
        self.model = config.ollama_model
        self.timeout = config.llm_timeout
        self.system_prompt = config.system_prompt
        self.allowed_functions = {
            "type",
            "search",
            "calculate",
            "play",
            "pause",
            "next",
            "previous",
        }

    def interpret(self, command_text: str) -> Optional[Dict[str, Any]]:
        payload = {
            "model": self.model,
            "prompt": command_text,
            "system": self.system_prompt,
            "stream": False,
            "format": "json",
        }
        logger.debug("Querying LLM with payload: %s", payload)
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to reach LLM endpoint: %s", exc)
            return None

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            logger.error("LLM returned non-JSON payload: %s", exc)
            return None

        raw_response = data.get("response", "").strip()
        if not raw_response:
            logger.error("LLM returned empty response payload")
            return None

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            logger.error("Unable to parse LLM JSON response '%s': %s", raw_response, exc)
            return None

        if not self._validate(parsed):
            logger.error("LLM response failed validation: %s", parsed)
            return None

        logger.debug("Validated LLM response: %s", parsed)
        return parsed

    def _validate(self, response: Dict[str, Any]) -> bool:
        if not isinstance(response, dict):
            return False
        function_name = response.get("function")
        if function_name not in self.allowed_functions:
            logger.error("Function '%s' is not in the allowed function list", function_name)
            return False
        parameters = response.get("parameters")
        if not isinstance(parameters, dict):
            logger.error("Parameters must be a dictionary")
            return False
        return True


class VoiceControl:
    """Voice control agent orchestrating wake-word detection and command execution."""

    def __init__(
        self,
        voice_queue,
        llm_queue,
        stop_event,
        config: Optional[VoiceControlConfig] = None,
    ) -> None:
        self.config = config or VoiceControlConfig()
        logger.setLevel(self.config.log_level)
        self.voice_queue = voice_queue
        self.llm_queue = llm_queue
        self.stop_event = stop_event
        self.audio_dir = self.config.audio_tmp_dir.expanduser().resolve()
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initialising VoiceControl v2 with wake word '%s'", self.config.wake_word)
        self._initialise_audio_device()
        self.whisper = WhisperRunner(self.config)
        self.llm_agent = LLMAgent(self.config)

    def _initialise_audio_device(self) -> None:
        if sd is None:
            logger.error(
                "sounddevice library is unavailable. Install 'sounddevice' to enable voice control."
            )
            return
        try:
            devices = sd.query_devices()
            logger.debug("Detected audio devices: %s", devices)
            default_input = sd.default.device[0]
            logger.info("Using input device index: %s", default_input)
        except Exception as exc:
            logger.warning("Unable to query audio devices: %s", exc)

    def run(self) -> None:
        logger.info("Voice control loop started. Waiting for wake word '%s'.", self.config.wake_word)
        try:
            while not self.stop_event.is_set():
                wake_phrase = self._capture_window(self.config.wake_window_seconds, stage="wake")
                if not wake_phrase:
                    continue
                logger.debug("Wake window transcript: '%s'", wake_phrase)
                if self.config.wake_word.lower() not in wake_phrase.lower():
                    continue
                logger.info("Wake word detected: '%s'", wake_phrase)
                command_text = self._capture_window(
                    self.config.command_window_seconds,
                    stage="command",
                    prompt="You are capturing a short command following the wake word.",
                )
                if not command_text:
                    logger.warning("Command window produced no transcription")
                    continue
                self._dispatch_command(command_text)
        except KeyboardInterrupt:
            logger.info("Voice control interrupted by user")
        finally:
            logger.info("Voice control loop exiting")

    def _capture_window(self, duration: float, stage: str, prompt: Optional[str] = None) -> str:
        if self.stop_event.is_set():
            return ""
        audio = self._record_audio(duration)
        if audio is None:
            return ""
        filename = self._save_audio(audio, stage)
        try:
            transcript = self.whisper.transcribe(filename, prompt)
            return transcript.strip()
        finally:
            if not self.config.keep_recordings:
                filename.unlink(missing_ok=True)

    def _record_audio(self, duration: float):
        if sd is None:
            logger.error("sounddevice is not available; cannot record audio")
            return None
        try:
            frames = int(duration * self.config.sample_rate)
            logger.debug("Recording %.2fs of audio (%s frames)", duration, frames)
            recording = sd.rec(
                frames,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="int16",
            )
            sd.wait()
            return recording
        except Exception as exc:
            logger.error("Audio recording failed: %s", exc)
            time.sleep(0.5)
            return None

    def _save_audio(self, audio, stage: str) -> Path:
        if write is None:
            raise RuntimeError(
                "scipy is required to save audio recordings. Install 'scipy' to continue."
            )
        timestamp = int(time.time() * 1000)
        filename = self.audio_dir / f"{stage}_{timestamp}.wav"
        try:
            write(filename, self.config.sample_rate, audio)
            logger.debug("Saved audio chunk to %s", filename)
        except Exception as exc:
            logger.error("Failed to write audio file %s: %s", filename, exc)
            raise
        return filename

    def _dispatch_command(self, command_text: str) -> None:
        clean_text = command_text.strip()
        if not clean_text:
            return
        logger.info("Captured voice command: '%s'", clean_text)
        if self.voice_queue is not None:
            self.voice_queue.put(clean_text)
        llm_response = self.llm_agent.interpret(clean_text)
        if llm_response:
            logger.info("Dispatching LLM command: %s", llm_response)
            self.llm_queue.put(llm_response)
        else:
            logger.warning("LLM did not return a valid command for: '%s'", clean_text)


def voice_control_process_entry(
    voice_queue,
    llm_queue,
    stop_event,
    config_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Entry point for the multiprocessing voice control worker."""

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
    )
    config = VoiceControlConfig.from_payload(config_payload)
    controller = VoiceControl(voice_queue, llm_queue, stop_event, config)
    controller.run()
