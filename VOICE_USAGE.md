# Voice Control Usage Guide

## How to Use Voice Control

### 1. Start the Application
```bash
poetry run dev
```

### 2. Wait for "Voice control loop started" message
The system will start listening for the wake word "hello"

### 3. Say the Wake Word
**Say clearly:** "Hello"

Wait for the message: **"✓ Wake word detected! Listening for command..."**

### 4. Give Your Command (WAIT 0.3 seconds after wake word)
After you see the wake word detected message, the system automatically waits 0.3 seconds, then starts recording for **5 seconds** to capture your command.

**During this 5 second window, say your command:**

Examples:
- "calculate 10 plus 5"
- "search for Python tutorials"
- "type hello world"
- "play"
- "pause"
- "next track"

### 5. Wait for Command Processing
The system will:
1. Transcribe your command (you'll see: "Command transcribed: '...'")
2. Send it to the LLM for interpretation
3. Execute the function

---

## Timing Breakdown

```
You say: "Hello"          ← Wake word (captured in 1.5 second window)
          ↓
System: "✓ Wake word detected! Listening for command..."
          ↓
     [0.3 sec pause]       ← Automatic pause
          ↓
You say: "calculate 10 + 5"   ← Command (5 second window starts NOW)
          ↓
System transcribes and executes
```

---

## Important Notes

### DO:
✓ Speak clearly and at normal pace
✓ Wait for the "✓ Wake word detected!" message before giving your command
✓ Give your full command within the 5 second window
✓ Keep commands concise (e.g., "calculate 10 plus 5")

### DON'T:
✗ Don't say the wake word again after it's detected
✗ Don't speak immediately after saying "hello" - let the system acknowledge it first
✗ Don't give complex multi-sentence commands
✗ Don't expect instant response - transcription takes 1-3 seconds

---

## Common Issues

### "System keeps hearing 'hello' in everything"
- Fixed! The system now requires exact word boundaries
- Wake detection is now more strict (threshold: 0.50)

### "Command gets transcribed as 'hello'"
- Fixed! Commands now use neutral prompts, not biased toward wake word
- System now waits 0.3 seconds after wake detection before listening for command

### "Wake word not detected"
- Speak louder (RMS energy must be > 80.0)
- Check logs for energy levels: `Wake window RMS=X.XX threshold=80.00`
- If you see RMS consistently below 80, lower `wake_energy_threshold` in config

---

## Configuration Tweaks (in voice_control.py)

If detection is still problematic, adjust these values:

```python
wake_similarity_threshold: float = 0.50   # Lower = more lenient (try 0.45)
wake_energy_threshold: float = 80.0       # Lower = more sensitive (try 60.0)
wake_retries: int = 1                     # Increase if wake detection fails
command_window_seconds: float = 5.0       # Increase if commands get cut off
```

---

## Logs to Watch

Look for these in the terminal:

1. **Wake Detection:**
   ```
   Wake window RMS=150.23 threshold=80.00 (187.8%)
   Wake window transcript: 'hello'
   ✓ Wake word exact match (word boundary) in transcript: 'hello'
   ✓ Wake word confirmed on attempt 1/2
   ✓ Wake word detected! Listening for command...
   ```

2. **Command Capture:**
   ```
   Command transcribed: 'calculate 10 plus 5'
   Captured voice command: 'calculate 10 plus 5'
   Dispatching LLM command: {'function': 'calculate', 'parameters': {'expression': '10+5'}}
   ```

3. **Issues:**
   ```
   Wake audio below energy threshold; skipping transcription  ← Speak louder
   Wake word not detected; best_ratio=0.35 ← Increase sensitivity or speak clearer
   Command window produced no transcription ← Speak during 5-sec window
   ```
