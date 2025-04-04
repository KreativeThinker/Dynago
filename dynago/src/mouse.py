import pyautogui
import numpy as np
from pynput.mouse import Controller, Button
import threading
from queue import Queue


class GestureMouse:
    def __init__(
        self,
        smoothing_factor=0.2,
        mouse_threshold=0.07,
        min_click_threshold=0.01,  # Minimum movement to consider as potential click
        max_click_threshold=0.06,  # Maximum movement to still count as click
        click_cooldown=10,
    ):
        """
        Args:
            smoothing_factor: float (0-1) for cursor movement smoothing (lower = smoother)
            min_click_threshold: minimum movement distance to consider as potential click
            max_click_threshold: maximum movement distance to still count as click
            click_cooldown: frames to wait between click detections
        """
        self.mouse = Controller()
        self.screen_width, self.screen_height = pyautogui.size()
        self.smoothing_factor = smoothing_factor
        self.min_click_threshold = min_click_threshold
        self.max_click_threshold = max_click_threshold
        self.click_cooldown = click_cooldown
        self.mouse_threshold = mouse_threshold
        self.cooldown_counter = 0
        self.last_position = None
        self.smoothed_pos = None
        self.click_velocity_buffer = []

        # Threading-related attributes
        self.task_queue = Queue()
        self.worker_thread = threading.Thread(target=self._process_updates, daemon=True)
        self.worker_thread.start()
        self.lock = threading.Lock()

    def _process_updates(self):
        """Worker thread function to process updates from the queue."""
        while True:
            finger_tip_pos = self.task_queue.get()
            self._update_mouse(finger_tip_pos)
            self.task_queue.task_done()

    def _update_mouse(self, finger_tip_pos):
        """Actual mouse update logic (runs in worker thread)."""
        # Convert normalized coordinates to screen coordinates
        current_x = int(finger_tip_pos[0] * self.screen_width)
        current_y = int(finger_tip_pos[1] * self.screen_height)

        # Apply smoothing
        with self.lock:
            if self.smoothed_pos is None:
                self.smoothed_pos = (current_x, current_y)
            else:
                self.smoothed_pos = (
                    int(
                        self.smoothed_pos[0]
                        + self.smoothing_factor * (current_x - self.smoothed_pos[0])
                    ),
                    int(
                        self.smoothed_pos[1]
                        + self.smoothing_factor * (current_y - self.smoothed_pos[1])
                    ),
                )

            # Detect click (sudden movement within thresholds)
            if self.last_position is not None and self.cooldown_counter <= 0:
                dx = finger_tip_pos[0] - self.last_position[0]
                dy = finger_tip_pos[1] - self.last_position[1]
                movement_magnitude = np.sqrt(dx**2 + dy**2)

                # Add to velocity buffer (for more robust click detection)
                self.click_velocity_buffer.append(movement_magnitude)
                if (
                    len(self.click_velocity_buffer) > 3
                    and movement_magnitude > self.mouse_threshold
                ):
                    self.click_velocity_buffer.pop(0)

                # Check for click (sudden movement between thresholds)
                if len(self.click_velocity_buffer) == 3 and all(
                    self.min_click_threshold < v < self.max_click_threshold
                    for v in self.click_velocity_buffer
                ):
                    # self.mouse.click(Button.left)
                    self.cooldown_counter = self.click_cooldown
                    self.click_velocity_buffer.clear()

            # Update mouse position
            self.mouse.position = self.smoothed_pos
            self.last_position = finger_tip_pos

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

    def update(self, finger_tip_pos):
        """
        Process new finger tip position and update mouse state (threaded)
        Args:
            finger_tip_pos: (x,y) normalized coordinates (0-1) of index finger tip
        """
        # Put the update task in the queue
        self.task_queue.put(finger_tip_pos)

    def __del__(self):
        """Clean up when the object is destroyed."""
        self.task_queue.join()  # Wait for all tasks to complete
