import multiprocessing

try:
    import pyautogui  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None  # type: ignore

from dynago.src.capture import capture_landmarks, command_worker, cleanup
from dynago.src.voice_control import VoiceControlConfig, voice_control_process_entry


def main():
    ctx = multiprocessing.get_context("spawn")

    # Create queues for inter-process communication
    cmd_queue = ctx.Queue(maxsize=10)  # For gesture commands
    voice_queue = ctx.Queue(maxsize=10)  # For raw voice transcripts (debug/logging)
    llm_queue = ctx.Queue(maxsize=10)  # For LLM function calls
    voice_stop_event = ctx.Event()
    voice_config = VoiceControlConfig(
    whisper_binary="/home/bufferfis/code/Dynago/whisper.cpp/build/bin/whisper-cli",
)

    gesture_worker = None
    voice_process = None
    llm_processor = None

    try:
        # Start gesture command worker
        gesture_worker = ctx.Process(
            target=command_worker,
            args=(cmd_queue,),
            daemon=True,
        )
        gesture_worker.start()

        # Start voice control process
        voice_process = ctx.Process(
            target=voice_control_process_entry,
            args=(voice_queue, llm_queue, voice_stop_event, voice_config.to_payload()),
            daemon=True,
        )
        voice_process.start()

        # Start LLM command processor
        llm_processor = ctx.Process(
            target=process_llm_commands, args=(llm_queue,), daemon=True
        )
        llm_processor.start()

        # Main capture process
        capture_landmarks(cmd_queue)

    finally:
        # Cleanup all processes
        cleanup()

        # Signal workers to stop
        cmd_queue.put(None)
        llm_queue.put(None)
        voice_stop_event.set()

        # Wait for workers to finish
        if gesture_worker is not None:
            gesture_worker.join(timeout=1)
        if voice_process is not None:
            voice_process.join(timeout=3)
        if llm_processor is not None:
            llm_processor.join(timeout=1)

        # Force terminate if needed
        if gesture_worker is not None and gesture_worker.is_alive():
            gesture_worker.terminate()
        if voice_process is not None and voice_process.is_alive():
            voice_process.terminate()
        if llm_processor is not None and llm_processor.is_alive():
            llm_processor.terminate()

        # Clean up queues
        cmd_queue.close()
        voice_queue.close()
        llm_queue.close()


def process_llm_commands(llm_queue):
    """Process function calls from the LLM"""
    while True:
        command = llm_queue.get()
        if command is None:  # Termination signal
            break
        execute_function(command)


def execute_function(command):
    """Execute the function specified by the LLM"""
    try:
        if command["function"] == "type":
            text_to_type = command["parameters"]["text"]

            if pyautogui is None:
                print("pyautogui is not installed; cannot type text.")
                return

            try:
                pyautogui.write(text_to_type, interval=0.05)
                return
            except Exception as e:
                print(f"pyautogui typing failed: {e}")

        elif command["function"] == "search":
            import webbrowser

            query = command["parameters"]["query"]
            webbrowser.open(f"https://www.google.com/search?q={query}")

        elif command["function"] == "calculate":
            result = eval(command["parameters"]["expression"])
            import subprocess
            result_text = str(result)

            try:
                subprocess.run(["notify-send", result_text])
            except Exception as notify_error:
                print(f"notify-send failed: {notify_error}")
            print(f"Calculation result: {result_text}")

        elif command["function"] in ["play", "pause", "next", "previous"]:
            import subprocess

            subprocess.run(["playerctl", command["function"]])

    except Exception as e:
        print(f"Error executing command: {e}")


if __name__ == "__main__":
    main()
