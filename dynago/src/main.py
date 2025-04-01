import multiprocessing

from dynago.src.capture import capture_landmarks, command_worker


def main():
    multiprocesssing = multiprocessing.get_context("spawn")

    # Create command queue
    cmd_queue = multiprocesssing.Queue(maxsize=10)

    try:
        # Start command worker
        worker = multiprocesssing.Process(
            target=command_worker,
            args=(cmd_queue,),
            daemon=True,
        )
        worker.start()
        capture_landmarks(cmd_queue)

    finally:
        # Cleanup
        cmd_queue.put(None)  # Signal worker to stop
        worker.join(timeout=1)  # Wait for worker to finish
        if worker.is_alive():
            worker.terminate()  # Force terminate if needed
        cmd_queue.close()  # Clean up queue
