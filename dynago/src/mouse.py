import pyautogui
import numpy as np
from pynput.mouse import Controller, Button


class GestureMouse:
    def __init__(
        self,
        smoothing_factor=0.2,
        min_click_threshold=0.01,  # Minimum movement to consider as potential click
        max_click_threshold=0.03,  # Maximum movement to still count as click
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
        self.cooldown_counter = 0
        self.last_position = None
        self.smoothed_pos = None
        self.click_velocity_buffer = []

    def update(self, finger_tip_pos):
        """
        Process new finger tip position and update mouse state
        Args:
            finger_tip_pos: (x,y) normalized coordinates (0-1) of index finger tip
        """
        # Convert normalized coordinates to screen coordinates
        current_x = int(finger_tip_pos[0] * self.screen_width)
        current_y = int(finger_tip_pos[1] * self.screen_height)

        # Apply smoothing
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
            if len(self.click_velocity_buffer) > 3:
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
