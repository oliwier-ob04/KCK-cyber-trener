"""Motion analysis using MediaPipe pose detection."""

from __future__ import annotations

import math
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
    knee_error: float = float('nan')


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
        self.last_knee_error = float('nan')

    def detect_knee_error(self, lm) -> float:
        """
        Detect knee alignment error based on whether the legs are parallel.
        Measures the angle between leg vectors (from knee to ankle).
        Returns percentage: 100% = perfectly parallel (0°), decreases with angle.
        Returns NaN if legs not visible.
        """
        
        if not lm:
            return float('nan')
            
        try:
            # Get knee and ankle landmarks
            l_knee = lm[self.mp_pose.PoseLandmark.LEFT_KNEE]
            l_ankle = lm[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            r_knee = lm[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            r_ankle = lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Check visibility threshold - if landmarks not visible enough, return NaN
            visibility_threshold = 0.5
            if (l_knee.visibility < visibility_threshold or 
                l_ankle.visibility < visibility_threshold or
                r_knee.visibility < visibility_threshold or
                r_ankle.visibility < visibility_threshold):
                return float('nan')
            
            # Create vectors for each leg (from knee to ankle)
            left_vector = (l_ankle.x - l_knee.x, l_ankle.y - l_knee.y)
            right_vector = (r_ankle.x - r_knee.x, r_ankle.y - r_knee.y)
            
            # Calculate vector lengths
            left_length = math.sqrt(left_vector[0]**2 + left_vector[1]**2)
            right_length = math.sqrt(right_vector[0]**2 + right_vector[1]**2)
            
            if left_length == 0 or right_length == 0:
                return float('nan')
            
            # Calculate dot product
            dot_product = left_vector[0] * right_vector[0] + left_vector[1] * right_vector[1]
            
            # Calculate cosine of angle between vectors
            cos_angle = dot_product / (left_length * right_length)
            
            # Clamp to [-1, 1] to avoid numerical errors
            cos_angle = max(-1.0, min(1.0, cos_angle))
            
            # Calculate angle in degrees
            angle_rad = math.acos(abs(cos_angle))
            angle_deg = math.degrees(angle_rad)
            
            # Convert to percentage: 100% at 0°, decreases with angle
            # Each degree of difference reduces the score by 1%
            score = max(0, 100 - angle_deg)
            
            return score / 100.0  # Return as 0-1 range
            
        except Exception:
            return float('nan')

    def analyze_frame(self, frame) -> tuple[float, any]:
        """Process a video frame, draw landmarks and return knee error percentage and modified frame."""
        
        if frame is None:
            return float('nan'), frame
            
        try:
            frame_copy = frame.copy()
            img_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)

            current_knee_error = float('nan')
            if results.pose_landmarks:
                # Draw landmarks on frame
                self.mp_drawing.draw_landmarks(
                    frame_copy, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                current_knee_error = self.detect_knee_error(results.pose_landmarks.landmark)

            # Update last_knee_error with current value (including NaN)
            self.last_knee_error = current_knee_error
                
            return current_knee_error, frame_copy
        except Exception as e:
            print(f"[POSE] BŁĄD: {e}")
            import traceback
            traceback.print_exc()
            return float('nan'), frame

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
            knee_error=self.last_knee_error,
        )
