import json
import logging
import multiprocessing
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from shutil import which
from typing import Any, Dict, Optional

import requests

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

logger = logging.getLogger(__name__)

if np is None:
    logger.warning(
        "numpy is not installed. Falling back to a slower RMS calculation for voice capture."
    )

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
    wake_window_seconds: float = 1.5
    command_window_seconds: float = 5.0
    wake_word: str = "hello"
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
    min_command_length: int = 4
    noise_phrases: tuple[str, ...] = ("applause", "laughter", "music", "noise")
    wake_similarity_threshold: float = 0.50
    wake_energy_threshold: float = 80.0
    wake_retries: int = 1
    wake_aliases: tuple[str, ...] = ()
    command_retries: int = 0
    wake_prompt_template: str = (
        "You are only listening for the wake word '{wake_word}'. Respond with that exact word if heard."  # noqa: E501
    )
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

    def __post_init__(self) -> None:
        if not self.wake_aliases:
            base = self.wake_word.lower().strip()
            variants = {base}
            # Remove spaces
            variants.add(base.replace(" ", ""))
            
            # Specific phonetic variants for "hello"
            if base == "hello":
                variants.update([
                    "ello",      # Dropped H
                    "helo",      # Single L
                    "hullo",     # British variant
                    "hallo",     # Germanic variant
                ])
            
            # Generic transformations for any wake word
            if base.startswith("h") and len(base) > 1:
                variants.add(base[1:])  # Drop leading H
            
            self.wake_aliases = tuple(sorted({alias for alias in variants if alias and len(alias) >= 3}))
        if self.wake_retries < 0:
            self.wake_retries = 0
        if self.wake_energy_threshold < 0:
            self.wake_energy_threshold = 0.0

    def to_payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data["whisper_model_path"] = str(self.whisper_model_path)
        data["audio_tmp_dir"] = str(self.audio_tmp_dir)
        data["whisper_candidates"] = list(self.whisper_candidates)
        data["noise_phrases"] = list(self.noise_phrases)
        data["wake_aliases"] = list(self.wake_aliases)
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
        if "noise_phrases" in data:
            data["noise_phrases"] = tuple(data["noise_phrases"])
        if "wake_aliases" in data:
            data["wake_aliases"] = tuple(data["wake_aliases"])
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

        logger.info("Initialising VoiceControl v3 with wake word '%s'", self.config.wake_word)
        self._initialise_audio_device()
        self.whisper = WhisperRunner(self.config)
        self.llm_agent = LLMAgent(self.config)
        self._last_command_text = ""

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
                wake_result = self._capture_window(
                    self.config.wake_window_seconds, stage="wake"
                )
                raw_wake, clean_wake = wake_result["raw"], wake_result["clean"]
                if not raw_wake:
                    continue
                logger.debug("Wake window transcript: '%s'", raw_wake)
                if not self._contains_wake_word(clean_wake) and not self._contains_wake_word(raw_wake):
                    continue
                
                logger.info("✓ Wake word detected! Listening for command...")
                # Brief pause to let user finish saying wake word and start command
                time.sleep(0.3)
                
                command_result = self._capture_window(
                    self.config.command_window_seconds,
                    stage="command",
                    prompt=None,  # No prompt bias for commands
                )
                raw_command, clean_command = (
                    command_result["raw"],
                    command_result["clean"],
                )
                if not raw_command:
                    logger.warning("Command window produced no transcription")
                    continue
                if not clean_command:
                    logger.warning("Discarded noisy command transcript: '%s'", raw_command)
                    continue
                if clean_command == self._last_command_text:
                    logger.debug("Skipping duplicate command: '%s'", clean_command)
                    continue
                self._last_command_text = clean_command
                self._dispatch_command(clean_command, raw_command)
        except KeyboardInterrupt:
            logger.info("Voice control interrupted by user")
        finally:
            logger.info("Voice control loop exiting")

    def _capture_window(
        self, duration: float, stage: str, prompt: Optional[str] = None
    ) -> Dict[str, str]:
        if self.stop_event.is_set():
            return {"raw": "", "clean": ""}
        audio_payload = self._record_audio(duration)
        if audio_payload is None:
            return {"raw": "", "clean": ""}
        audio, rms = audio_payload
        if stage == "wake":
            logger.debug(
                "Wake window RMS=%.2f threshold=%.2f (%.1f%%)",
                rms,
                self.config.wake_energy_threshold,
                (rms / self.config.wake_energy_threshold * 100) if self.config.wake_energy_threshold > 0 else 0
            )
            if rms < self.config.wake_energy_threshold:
                logger.debug("Wake audio below energy threshold; skipping transcription")
                return {"raw": "", "clean": ""}
        filename = self._save_audio(audio, stage)
        stage_prompt = prompt
        if stage == "wake" and not stage_prompt:
            stage_prompt = self.config.wake_prompt_template.format(
                wake_word=self.config.wake_word
            )
        try:
            transcript, cleaned = self._transcribe_with_retries(filename, stage, stage_prompt)
            return {"raw": transcript.strip(), "clean": cleaned}
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
            rms = self._calculate_rms(recording)
            return recording, rms
        except Exception as exc:
            logger.error("Audio recording failed: %s", exc)
            time.sleep(0.5)
            return None

    def _calculate_rms(self, audio) -> float:
        if audio is None:
            return 0.0
        if np is not None:
            arr = np.asarray(audio, dtype=np.float32)
            if arr.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(arr**2)))
        # Fallback for environments without numpy
        flat = audio.flatten() if hasattr(audio, "flatten") else audio
        total = 0.0
        count = 0
        for sample in flat:
            value = float(sample)
            total += value * value
            count += 1
        if count == 0:
            return 0.0
        return float(total / count) ** 0.5

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

    def _dispatch_command(self, clean_text: str, raw_text: str) -> None:
        logger.info("Captured voice command: '%s'", raw_text)
        if self.voice_queue is not None:
            self.voice_queue.put(clean_text)
        llm_response = self.llm_agent.interpret(clean_text)
        if llm_response:
            logger.info("Dispatching LLM command: %s", llm_response)
            self.llm_queue.put(llm_response)
        else:
            logger.warning("LLM did not return a valid command for: '%s'", clean_text)

    def _clean_transcript(self, transcript: str, stage: str) -> str:
        if not transcript:
            return ""
        text = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]", "", transcript)
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not any(ch.isalpha() for ch in text):
            return ""
        if stage == "wake":
            return text
        if len(text) < self.config.min_command_length:
            return ""
        normalized = text.lower().strip()
        if normalized in self.config.noise_phrases:
            return ""
        if all(token in self.config.noise_phrases for token in normalized.split()):
            return ""
        return text

    def _contains_wake_word(self, transcript: str) -> bool:
        wake_word = self.config.wake_word.lower().strip()
        if not transcript:
            return False
        candidate = transcript.lower().strip()
        
        # Direct exact match (full word boundary)
        words = re.findall(r'\b[a-z]+\b', candidate)
        if wake_word in words:
            logger.debug("Wake word exact match (word boundary) in transcript: '%s'", transcript)
            return True
        
        # Check all aliases for exact word matches
        for alias in self.config.wake_aliases:
            alias_norm = alias.lower().strip()
            if alias_norm and alias_norm in words:
                logger.debug("Wake word alias detected: '%s' in '%s'", alias_norm, transcript)
                return True
        
        # Fuzzy matching as fallback
        best_ratio = 0.0
        best_match = ""
        
        for word in words:
            # Check wake word
            ratio = SequenceMatcher(None, wake_word, word).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = word
            if ratio >= self.config.wake_similarity_threshold:
                logger.debug("Wake word fuzzy match: word='%s' ratio=%.2f", word, ratio)
                return True
            
            # Check aliases
            for alias in self.config.wake_aliases:
                alias_norm = alias.lower().strip()
                if not alias_norm:
                    continue
                alias_ratio = SequenceMatcher(None, alias_norm, word).ratio()
                if alias_ratio > best_ratio:
                    best_ratio = alias_ratio
                    best_match = f"{word}~{alias_norm}"
                if alias_ratio >= self.config.wake_similarity_threshold:
                    logger.debug(
                        "Wake alias fuzzy match: alias='%s' word='%s' ratio=%.2f",
                        alias,
                        word,
                        alias_ratio,
                    )
                    return True
        
        # Log the best we found even if it didn't match
        logger.debug(
            "Wake word not detected; best_ratio=%.2f best_match='%s' transcript='%s'",
            best_ratio,
            best_match,
            transcript
        )
        return False

    def _transcribe_with_retries(
        self, audio_path: Path, stage: str, prompt: Optional[str]
    ) -> tuple[str, str]:
        attempts = 1
        if stage == "wake":
            attempts = max(1, self.config.wake_retries + 1)
        elif stage == "command":
            attempts = max(1, self.config.command_retries + 1)
        
        best_raw = ""
        best_clean = ""
        best_score = 0.0
        last_raw = ""
        last_clean = ""
        
        for attempt in range(attempts):
            attempt_prompt = prompt
            
            if stage == "wake" and not attempt_prompt:
                # For wake detection, use minimal prompting
                if attempt == 0:
                    attempt_prompt = None  # No prompt on first try
                else:
                    attempt_prompt = f"Wake word: {self.config.wake_word}"
            elif stage == "command":
                # For commands, never bias with wake word
                attempt_prompt = "Transcribe the voice command accurately."
            
            raw = self.whisper.transcribe(audio_path, attempt_prompt)
            clean = self._clean_transcript(raw, stage)
            last_raw, last_clean = raw, clean
            
            if raw:
                logger.debug(
                    "Transcription attempt %d/%d (stage=%s): '%s'",
                    attempt + 1,
                    attempts,
                    stage,
                    raw,
                )
            
            if stage == "wake":
                score = self._wake_similarity_score(clean or raw)
                logger.debug("Wake similarity score: %.3f", score)
                if score > best_score:
                    best_score = score
                    best_raw, best_clean = raw, clean
                    logger.debug("New best wake transcription (score=%.3f): '%s'", score, raw)
                
                # Check if we have a match
                if self._contains_wake_word(clean) or self._contains_wake_word(raw):
                    logger.info("✓ Wake word confirmed on attempt %d/%d", attempt + 1, attempts)
                    return raw, clean
            else:
                # For commands, return first valid transcription
                if clean:
                    logger.info("Command transcribed: '%s'", clean)
                    return raw, clean
                if not best_raw:
                    best_raw, best_clean = raw, clean
        
        # Return best match found
        if not best_raw:
            best_raw, best_clean = last_raw, last_clean
        
        if stage == "wake" and best_raw:
            logger.debug(
                "Best wake transcription after %d attempts (score=%.3f): '%s'",
                attempts,
                best_score,
                best_raw
            )
        
        return best_raw or "", best_clean or ""

    def _wake_similarity_score(self, transcript: str) -> float:
        if not transcript:
            return 0.0
        candidate = transcript.lower().strip()
        wake_word = self.config.wake_word.lower().strip()
        tokens = re.findall(r"[a-z]+", candidate)
        scores = [SequenceMatcher(None, wake_word, candidate).ratio()]
        for alias in self.config.wake_aliases:
            alias_norm = alias.lower().strip()
            if alias_norm:
                scores.append(SequenceMatcher(None, alias_norm, candidate).ratio())
        for token in tokens:
            scores.append(SequenceMatcher(None, wake_word, token).ratio())
            for alias in self.config.wake_aliases:
                alias_norm = alias.lower().strip()
                if alias_norm:
                    scores.append(SequenceMatcher(None, alias_norm, token).ratio())
        return max(scores) if scores else 0.0


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
