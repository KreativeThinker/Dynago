import multiprocessing
from dynago.src.capture import capture_landmarks, command_worker, cleanup
from dynago.src.voice_control import VoiceControl
import pyautogui


def main():
    ctx = multiprocessing.get_context("spawn")

    # Create queues for inter-process communication
    cmd_queue = ctx.Queue(maxsize=10)  # For gesture commands
    voice_queue = ctx.Queue(maxsize=10)  # For voice commands
    llm_queue = ctx.Queue(maxsize=10)  # For LLM function calls

    try:
        # Start gesture command worker
        gesture_worker = ctx.Process(
            target=command_worker,
            args=(cmd_queue,),
            daemon=True,
        )
        gesture_worker.start()

        # Start voice control process
        voice_control = VoiceControl(voice_queue, llm_queue)
        voice_process = ctx.Process(target=voice_control.run)
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
        voice_queue.put(None)
        llm_queue.put(None)

        # Wait for workers to finish
        gesture_worker.join(timeout=1)
        voice_control.is_running = False
        voice_process.kill()
        voice_process.join(timeout=1)
        llm_processor.join(timeout=1)

        # Force terminate if needed
        if gesture_worker.is_alive():
            gesture_worker.terminate()
        if voice_process.is_alive():
            voice_process.terminate()
        if llm_processor.is_alive():
            llm_processor.terminate()

        # Clean up queues
        cmd_queue.close()
        voice_queue.close()
        llm_queue.close()


def process_llm_commands(llm_queue):
    """Process function calls from the LLM"""
    while True:
        if not llm_queue.empty():
            command = llm_queue.get()
            if command is None:  # Termination signal
                break
            execute_function(command)


def execute_function(command):
    """Execute the function specified by the LLM"""
    try:
        if command["function"] == "type":
            text_to_type = command["parameters"]["text"]

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

            subprocess.run(["notify-send", result])
            print(f"Calculation result: {result}")

        elif command["function"] in ["play", "pause", "next", "previous"]:
            import subprocess

            subprocess.run(["playerctl", command["function"]])

    except Exception as e:
        print(f"Error executing command: {e}")


if __name__ == "__main__":
    main()
