from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class PoseMetrics:
    """Zestaw czystych metryk pobranych bezpośrednio z klatki obrazu."""

    pose_detected: bool
    side: str = "unknown"
    side_name: str = "unknown"
    visibility: float = 0.0
    knee_angle_front: float = float("nan")
    knee_angle_side: float = float("nan")
    upper_body_angle: float = float("nan")
    hand_raised: bool = False
    message: str = "Brak wykrytej sylwetki"


@dataclass(frozen=True)
class MovementState:
    """Migawka stanu modelu ruchu dla interfejsu."""

    phase: float
    phase_value: float
    movement_state: str
    hip_angle: float
    repetition_detected: bool


class MovementAnalyzer:
    """Analiza ruchu użytkownika w czasie rzeczywistym przy użyciu MediaPipe Pose."""

    def __init__(
        self,
        phase_speed: float = 2.4,
        repetition_threshold: float = 0.90,
        base_hip_angle: float = 170.0,
        hip_angle_amplitude: float = 8.0,
        history_buffer_size: int = 7,
    ) -> None:
        self.phase_speed = phase_speed
        self.repetition_threshold = repetition_threshold
        self.base_hip_angle = base_hip_angle
        self.hip_angle_amplitude = hip_angle_amplitude
        self.history_buffer_size = history_buffer_size
        
        # Inicjalizacja MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(model_complexity=0)
        
        self.reset()

    def reset(self) -> None:
        """Resetowanie historii pomiarów."""
        self.current_phase = 0.0
        self._previous_state = "OPUSZCZANIE"
        self.history = []
        self._knee_angle_history: list[float] = []
        self._upper_angle_history: list[float] = []
        self.last_pose_metrics = PoseMetrics(False)

    @staticmethod
    def _angle(a, b, c) -> float:
        """Oblicza płaski kąt ABC w stopniach (b to wierzchołek) z pominięciem głębi Z."""
        ab = np.array([a.x - b.x, a.y - b.y], dtype=float)
        cb = np.array([c.x - b.x, c.y - b.y], dtype=float)
        
        ab_len = np.linalg.norm(ab)
        cb_len = np.linalg.norm(cb)
        if ab_len == 0 or cb_len == 0:
            return float("nan")

        cosine = float(np.dot(ab, cb) / (ab_len * cb_len))
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    def _smooth_angle(self, history: list[float], angle: float) -> float:
        """Wygładzanie wskazań za pomocą krótkiego filtru medianowego."""
        if math.isnan(angle):
            return angle
        history.append(angle)
        if len(history) > self.history_buffer_size:
            del history[0]
        return float(statistics.median(history))
        
    def _analyze_front_view(self, frame, landmarks, visibility) -> PoseMetrics:
        """ANALIZA WIDOKU Z PRZODU: Wylicza różnicę kątową nachylenia łydek."""
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

        if any(lm.visibility < 0.5 for lm in [left_knee, left_ankle, right_knee, right_ankle]):
            return PoseMetrics(
                pose_detected=True, side="front", side_name="front", visibility=visibility,
                message="FRONT: Cofnij się, aby było widać całe nogi"
            )

        # Wyznaczenie kątów łydek w układzie ekranu
        left_calf_angle = math.degrees(math.atan2(left_ankle.x - left_knee.x, left_ankle.y - left_knee.y))
        right_calf_angle = math.degrees(math.atan2(right_ankle.x - right_knee.x, right_ankle.y - right_knee.y))
        angle_diff = abs(left_calf_angle - right_calf_angle)

        # Detekcja uniesienia dłoni nad linię barków
        hand_raised = (
            (landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y - 0.05) or
            (landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y < landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y - 0.05)
        )

        # Rysowanie tekstu na obrazie
        line_color = (34, 197, 94) if angle_diff < 10.0 else (59, 68, 239)
        cv2.putText(frame, f"Roznica: {angle_diff:.1f}*", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

        return PoseMetrics(
            pose_detected=True,
            side="front",
            side_name="front",
            visibility=visibility,
            knee_angle_front=angle_diff,
            hand_raised=hand_raised,
            message="FRONT: Analiza poprawna"
        )

    def _analyze_side_view(self, side_name, landmarks, visibility) -> PoseMetrics:
        """ANALIZA WIDOKU Z BOKU: Liczy kąt tułowia oraz kąt łydki względem podłoża."""
        if side_name == "left":
            shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        else:
            shoulder = landmarks[getattr(self.mp_pose.PoseLandmark, "RIGHT_SHOULDER")]
            hip = landmarks[getattr(self.mp_pose.PoseLandmark, "RIGHT_HIP")]
            knee = landmarks[getattr(self.mp_pose.PoseLandmark, "RIGHT_KNEE")]
            ankle = landmarks[getattr(self.mp_pose.PoseLandmark, "RIGHT_ANKLE")]
            wrist = landmarks[getattr(self.mp_pose.PoseLandmark, "RIGHT_WRIST")]

        # upper_body_angle: kąt kolano -> biodro -> bark
        raw_upper_angle = self._angle(knee, hip, shoulder)
        upper_body_angle = self._smooth_angle(self._upper_angle_history, raw_upper_angle)
        
        # knee_angle_side: kąt linii łydki względem poziomego podłoża (na płaskim ekranie)
        dx = ankle.x - knee.x
        dy = ankle.y - knee.y
        shin_length = math.sqrt(dx**2 + dy**2)
        
        if shin_length > 0:
            # Używamy asin z rzutu pionowego Y, co zwraca 90 stopni dla idealnego pionu łydki do ziemi
            raw_knee_side = math.degrees(math.asin(max(-1.0, min(1.0, abs(dy) / shin_length))))
        else:
            raw_knee_side = float("nan")
            
        knee_angle_side = self._smooth_angle(self._knee_angle_history, raw_knee_side)
        hand_raised = shoulder.visibility > 0.4 and wrist.visibility > 0.4 and wrist.y < shoulder.y - 0.05

        return PoseMetrics(
            pose_detected=True,
            side=side_name,
            side_name=side_name,
            visibility=visibility,
            knee_angle_side=knee_angle_side,
            upper_body_angle=upper_body_angle,
            hand_raised=hand_raised,
            message=f"BOK ({side_name.upper()}): Analiza poprawna"
        )

    def analyze_frame(self, frame) -> tuple[PoseMetrics, any]:
        """Główna pętla przetwarzania klatki wideo."""
        if frame is None:
            self.last_pose_metrics = PoseMetrics(False)
            return self.last_pose_metrics, frame
            
        try:
            frame_copy = frame.copy()
            img_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)

            metrics = PoseMetrics(False)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                sides = {}
                for prefix in ("LEFT", "RIGHT"):
                    vis = sum(landmarks[getattr(self.mp_pose.PoseLandmark, f"{prefix}_{joint}")].visibility 
                              for joint in ("SHOULDER", "HIP", "KNEE", "ANKLE")) / 4.0
                    sides[prefix.lower()] = vis

                dominant_side = max(sides, key=sides.get)
                visibility = sides[dominant_side]
                
                if abs(sides["left"] - sides["right"]) < 0.1:
                    side_name = "front"
                else:
                    side_name = dominant_side

                if side_name == "front":
                    metrics = self._analyze_front_view(frame_copy, landmarks, visibility)
                else:
                    metrics = self._analyze_side_view(side_name, landmarks, visibility)

                self._draw_key_overlay(frame_copy, metrics, landmarks)
                
            self.last_pose_metrics = metrics
            return metrics, frame_copy
            
        except Exception as e:
            print(f"[POSE] BŁĄD: {e}")
            self.last_pose_metrics = PoseMetrics(False)
            return self.last_pose_metrics, frame

    def _draw_key_overlay(self, frame, metrics: PoseMetrics, landmarks) -> None:
        """Rysowanie linii na ekranie."""
        try:
            h, w = frame.shape[:2]
            pt = lambda lm: (int(lm.x * w), int(lm.y * h))

            color_top = (0, 200, 255)
            color_joint = (0, 0, 255)
            
            if metrics.side == "front":
                calf_color = (34, 197, 94) if metrics.knee_angle_front < 10.0 else (59, 68, 239)
                for prefix in ("LEFT", "RIGHT"):
                    kn = landmarks[getattr(self.mp_pose.PoseLandmark, f"{prefix}_KNEE")]
                    an = landmarks[getattr(self.mp_pose.PoseLandmark, f"{prefix}_ANKLE")]
                    cv2.line(frame, pt(kn), pt(an), calf_color, 4)
                    cv2.circle(frame, pt(kn), 6, (0, 0, 255), -1)
                    cv2.circle(frame, pt(an), 6, (0, 0, 255), -1)
            else:          
                prefix = metrics.side.upper()
                shoulder = landmarks[getattr(self.mp_pose.PoseLandmark, f"{prefix}_SHOULDER")]
                hip = landmarks[getattr(self.mp_pose.PoseLandmark, f"{prefix}_HIP")]
                knee = landmarks[getattr(self.mp_pose.PoseLandmark, f"{prefix}_KNEE")]
                ankle = landmarks[getattr(self.mp_pose.PoseLandmark, f"{prefix}_ANKLE")]
                wrist = landmarks[getattr(self.mp_pose.PoseLandmark, f"{prefix}_WRIST")]

                hip_angle = metrics.upper_body_angle
                if not math.isnan(hip_angle):
                    if 175.0 <= hip_angle <= 185.0:
                        color_hip_line = (34, 197, 94)    # Zielony (180 +/- 5)
                    elif 85.0 <= hip_angle < 175.0:
                        color_hip_line = (0, 165, 255)    # Pomarańczowy
                    else:
                        color_hip_line = (59, 68, 239)    # Czerwony
                else:
                    color_hip_line = (255, 255, 255)     # Biały, jeśli brak danych

                knee_angle = metrics.knee_angle_side
                if not math.isnan(knee_angle) and (85.0 <= knee_angle <= 95.0):
                    color_calf_line = (34, 197, 94)   # Zielony
                else:
                    color_calf_line = (0, 255, 255)
                    
                cv2.line(frame, pt(shoulder), pt(hip), color_hip_line, 2)
                cv2.line(frame, pt(hip), pt(knee), color_hip_line, 4)
                cv2.line(frame, pt(knee), pt(ankle), color_calf_line, 4)
                
                for joint in (hip, knee, ankle):
                    cv2.circle(frame, pt(joint), 5 if joint != ankle else 4, color_joint, -1)

                if metrics.hand_raised:
                    cv2.line(frame, pt(shoulder), pt(wrist), color_top, 2)
                    cv2.circle(frame, pt(wrist), 4, color_top, -1)
        except Exception:
            pass

    def step(self, delta_seconds: float) -> MovementState:
        """Aktualizuje stan modelu ruchu."""
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
        )