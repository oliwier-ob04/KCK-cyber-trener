"""Motion analysis using MediaPipe pose detection."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class MovementState:
    """Motion snapshot produced for every UI tick."""

    phase: float
    phase_value: float
    movement_state: str
    hip_angle: float
    repetition_detected: bool
    detected_letter: str = "Brak"


class MovementAnalyzer:
    """Real-time movement analysis using MediaPipe pose detection."""

    def __init__(
        self,
        phase_speed: float = 2.4,
        repetition_threshold: float = 0.92,
        base_hip_angle: float = 170.0,
        hip_angle_amplitude: float = 8.0,
        history_buffer_size: int = 7,
    ) -> None:
        """Initialize MediaPipe pose detector and movement parameters."""

        self.phase_speed = phase_speed
        self.repetition_threshold = repetition_threshold
        self.base_hip_angle = base_hip_angle
        self.hip_angle_amplitude = hip_angle_amplitude
        self.history_buffer_size = history_buffer_size
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(model_complexity=0)
        
        self.reset()

    def reset(self) -> None:
        """Reset the internal state before a new session."""

        self.current_phase = 0.0
        self._previous_state = "OPUSZCZANIE"
        self.history = []
        self.last_detected_letter = "Brak"

    def _detect_letter(self, lm) -> str:
        """Detect gesture letter from pose landmarks (I, T, Y, L)."""
        
        if not lm:
            return "Brak"
            
        try:
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            tol = 0.15
            i_height = 0.2

            # Letter "I" - both arms up
            if l_wr.y < l_sh.y - i_height and r_wr.y < r_sh.y - i_height:
                if abs(l_wr.x - l_sh.x) < tol and abs(r_wr.x - r_sh.x) < tol:
                    return "I"

            # Letter "T" - arms horizontal
            if abs(l_wr.y - l_sh.y) < tol and abs(r_wr.y - r_sh.y) < tol:
                if l_wr.x > l_sh.x and r_wr.x < r_sh.x:
                    return "T"

            # Letter "Y" - both arms up and spread
            if l_wr.y < l_sh.y - tol and r_wr.y < r_sh.y - tol:
                if l_wr.x > l_sh.x + 0.1 and r_wr.x < r_sh.x - 0.1:
                    return "Y"

            # Letter "L" - right up, left horizontal
            r_up = r_wr.y < r_sh.y - tol and abs(r_wr.x - r_sh.x) < tol
            l_side = abs(l_wr.y - l_sh.y) < tol

            if r_up and l_side:
                return "L"

            return "Brak"
        except Exception:
            return "Brak"

    def analyze_frame(self, frame) -> tuple[str, any]:
        """Process a video frame, draw landmarks and return detected letter and modified frame."""
        
        if frame is None:
            return "Brak", frame
            
        try:
            frame_copy = frame.copy()
            img_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)

            current_letter = "Brak"
            if results.pose_landmarks:
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame_copy, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                current_letter = self._detect_letter(results.pose_landmarks.landmark)

            self.history.append(current_letter)
            if len(self.history) > self.history_buffer_size:
                self.history.pop(0)

            if self.history:
                self.last_detected_letter = Counter(self.history).most_common(1)[0][0]
            else:
                self.last_detected_letter = "Brak"
            
            # Draw detected letter on frame
            cv2.putText(
                frame_copy,
                f"Litera: {self.last_detected_letter}",
                (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )
                
            return self.last_detected_letter, frame_copy
        except Exception as e:
            print(f"[POSE] BŁĄD: {e}")
            import traceback
            traceback.print_exc()
            return "Brak", frame

    def step(self, delta_seconds: float) -> MovementState:
        """Advance the motion model and return the current state."""

        self.current_phase += delta_seconds * self.phase_speed
        phase_value = math.sin(self.current_phase)
        movement_state = "PODNOSZENIE" if phase_value > 0 else "OPUSZCZANIE"
        repetition_detected = (
            self._previous_state == "OPUSZCZANIE"
            and movement_state == "PODNOSZENIE"
            and phase_value > self.repetition_threshold
        )
        self._previous_state = movement_state
        hip_angle = self.base_hip_angle + math.sin(self.current_phase * 1.3) * self.hip_angle_amplitude
        return MovementState(
            phase=self.current_phase,
            phase_value=phase_value,
            movement_state=movement_state,
            hip_angle=hip_angle,
            repetition_detected=repetition_detected,
            detected_letter=self.last_detected_letter,
        )
