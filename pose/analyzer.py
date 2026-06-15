"""Motion analysis using MediaPipe pose detection."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class PoseMetrics:
    """Measured pose state extracted from the current frame."""

    pose_detected: bool
    side: str = "unknown"
    knee_angle: float = float("nan")
    upper_body_angle: float = float("nan")
    hand_raised: bool = False
    start_ready: bool = False
    top_ready: bool = False
    bottom_ready: bool = False
    pose_visibility: float = 0.0
    knee_error: float = float("nan")
    message: str = "Brak wykrytej sylwetki"


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
        repetition_threshold: float = 0.90,
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

    @staticmethod
    def _angle(a, b, c, use_z: bool = False) -> float:
        """Return the angle ABC in degrees.
        
        Args:
            a, b, c: Landmarks (b is the vertex)
            use_z: If False, uses 2D (X,Y) only (for side-view). 
                   If True, uses 3D (X,Y,Z) for depth-aware calculation.
        """

        if use_z:
            ab = np.array([a.x - b.x, a.y - b.y, getattr(a, "z", 0.0) - getattr(b, "z", 0.0)], dtype=float)
            cb = np.array([c.x - b.x, c.y - b.y, getattr(c, "z", 0.0) - getattr(b, "z", 0.0)], dtype=float)
        else:
            # For side-view, use only 2D coordinates to avoid Z distortion
            ab = np.array([a.x - b.x, a.y - b.y], dtype=float)
            cb = np.array([c.x - b.x, c.y - b.y], dtype=float)
        
        ab_len = np.linalg.norm(ab)
        cb_len = np.linalg.norm(cb)
        if ab_len == 0 or cb_len == 0:
            return float("nan")

        cosine = float(np.dot(ab, cb) / (ab_len * cb_len))
        cosine = max(-1.0, min(1.0, cosine))
        raw_angle = math.degrees(math.acos(cosine))
        return raw_angle

    @staticmethod
    def _normalize_view_angle(angle: float, target: float, tolerance: float = 35.0) -> float:
        """Apply minimal smoothing/correction to measured angle.
        
        With proper 2D calculation, angles should be nearly accurate.
        Only apply small corrections for edge cases.
        """

        if math.isnan(angle):
            return angle
        
        if abs(target - 90.0) < 1e-6:
            # For 90° target, apply very minimal correction
            if angle < 50.0:
                return angle + 5.0
            elif angle > 130.0:
                return angle - 5.0
            return angle
        
        if abs(target - 180.0) < 1e-6:
            # For 180° target, apply very minimal correction
            if angle < 160.0:
                return angle + 3.0
            return angle
        
        return angle

    def reset(self) -> None:
        """Reset the internal state before a new session."""

        self.current_phase = 0.0
        self._previous_state = "OPUSZCZANIE"
        self.history = []
        self._knee_angle_history: list[float] = []
        self._upper_angle_history: list[float] = []
        self.last_knee_error = float('nan')
        self.last_pose_metrics = PoseMetrics(False, "unknown")

    def _smooth_angle(self, history: list[float], angle: float) -> float:
        """Smooth angle readings with a short median filter."""

        if math.isnan(angle):
            return angle
        history.append(angle)
        if len(history) > self.history_buffer_size:
            del history[0]
        if len(history) == 1:
            return angle
        return float(statistics.median(history))

    def detect_knee_error(self, lm) -> float:
        """
        Detect knee stability on the most visible leg.

        In a side view, the hidden leg is often hallucinated by MediaPipe, which makes
        comparing both calves noisy. We therefore use the better visible leg and measure
        how close the shin vector is to vertical.

        Returns percentage: 100% = shin aligned with vertical, decreases with tilt.
        Returns NaN if no leg is visible enough.
        """
        
        if not lm:
            return float('nan')
            
        try:
            candidates = (
                (
                    lm[self.mp_pose.PoseLandmark.LEFT_KNEE],
                    lm[self.mp_pose.PoseLandmark.LEFT_ANKLE],
                ),
                (
                    lm[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                    lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE],
                ),
            )

            visibility_threshold = 0.5
            best_candidate = None
            best_visibility = visibility_threshold
            for knee, ankle in candidates:
                leg_visibility = min(knee.visibility, ankle.visibility)
                if leg_visibility > best_visibility:
                    best_visibility = leg_visibility
                    best_candidate = (knee, ankle)

            if best_candidate is None:
                return float('nan')

            knee, ankle = best_candidate
            shin_vector = (ankle.x - knee.x, ankle.y - knee.y)
            shin_length = math.sqrt(shin_vector[0] ** 2 + shin_vector[1] ** 2)
            if shin_length == 0:
                return float('nan')

            vertical_cos = abs(shin_vector[1] / shin_length)
            vertical_cos = max(-1.0, min(1.0, vertical_cos))
            angle_deg = math.degrees(math.acos(vertical_cos))

            score = max(0.0, 100.0 - angle_deg)
            return score / 100.0  # Return as 0-1 range
            
        except Exception:
            return float('nan')
        
    def _analyze_front_view(self, frame, landmarks) -> PoseMetrics:
        """Analizuje widok od frontu pod kątem równoległości łydek (kolana-kostki)."""
        h, w = frame.shape[:2]

        # Pobieramy punkty kluczowe dla obu łydek
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Sprawdzamy widoczność punktów dolnej partii ciała
        min_vis = 0.5
        if (left_knee.visibility < min_vis or left_ankle.visibility < min_vis or
            right_knee.visibility < min_vis or right_ankle.visibility < min_vis):
            return PoseMetrics(
                pose_detected=True,
                side="front",
                knee_error=float('nan'),
                message="FRONT: Cofnij się, aby było widać całe nogi"
            )

        # 1. Obliczamy kąt nachylenia lewej łydki (w stopniach) względem osi Y (pionu)
        # math.atan2(dx, dy) zwraca kąt w radianach
        left_calf_angle = math.degrees(math.atan2(left_ankle.x - left_knee.x, left_ankle.y - left_knee.y))
        
        # 2. Obliczamy kąt nachylenia prawej łydki względem osi Y
        right_calf_angle = math.degrees(math.atan2(right_ankle.x - right_knee.x, right_ankle.y - right_knee.y))

        # 3. Różnica między kątami mówi nam, jak bardzo łydki odchylają się od równoległości
        # Stosujemy abs(), ponieważ nie interesuje nas, w którą stronę jest odchylenie
        angle_diff = abs(left_calf_angle - right_calf_angle)

        # 4. Mapujemy to na wskaźnik błędu kolan (knee_error) w skali 0.0 do 1.0
        # Załóżmy, że idealnie równoległe to 0 stopni różnicy (knee_error = 1.0 - wszystko super)
        # A limit tolerancji to np. 12 stopni różnicy (powyżej tego kolana bardzo schodzą się lub rozchodzą)
        max_allowable_diff = 12.0
        
        # Obliczamy stabilność (1.0 = idealnie równolegle, 0.0 = maksymalny błąd)
        stability_score = max(0.0, min(1.0, 1.0 - (angle_diff / max_allowable_diff)))

        # Definiujemy próg błędu (np. jeśli stabilność spada poniżej 75%, to mamy błąd)
        has_error = stability_score < 0.75
        
        # Dobieramy kolor linii AR w zależności od wyniku
        line_color = (34, 197, 94) if not has_error else (59, 68, 239) # Zielony (RGB/BGR) vs Czerwony
        message = "FRONT: Kolana stabilne (lydki rownolegle)" if not has_error else "FRONT: UWAGA! Kolana uciekaja!"

        # --- RYSOWANIE LINII ŁYDEK DLA WIDOKU Z PRZODU ---
        def pt(lm):
            return (int(lm.x * w), int(lm.y * h))

        # Rysujemy grubszą linię dla lewej i prawej łydki
        cv2.line(frame, pt(left_knee), pt(left_ankle), line_color, 4)
        cv2.line(frame, pt(right_knee), pt(right_ankle), line_color, 4)
        
        # Zaznaczamy stawy kropkami
        cv2.circle(frame, pt(left_knee), 6, (0, 0, 255), -1)
        cv2.circle(frame, pt(left_ankle), 6, (0, 0, 255), -1)
        cv2.circle(frame, pt(right_knee), 6, (0, 0, 255), -1)
        cv2.circle(frame, pt(right_ankle), 6, (0, 0, 255), -1)

        # Wypisujemy kąt różnicy na ekranie w celach debugowania
        cv2.putText(frame, f"Roznica: {angle_diff:.1f}*", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

        return PoseMetrics(
            pose_detected=True,
            side="front",
            knee_error=stability_score, # Zwracamy wartość stabilności do głównej pętli
            message=message
        )

    def analyze_frame(self, frame) -> tuple[PoseMetrics, any]:
        """Process a video frame, draw landmarks and return pose metrics plus modified frame."""
        
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
                side_candidates = []
                for side_name, shoulder_idx, hip_idx, knee_idx, ankle_idx, wrist_idx in (
                    (
                        "left",
                        self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                        self.mp_pose.PoseLandmark.LEFT_HIP,
                        self.mp_pose.PoseLandmark.LEFT_KNEE,
                        self.mp_pose.PoseLandmark.LEFT_ANKLE,
                        self.mp_pose.PoseLandmark.LEFT_WRIST,
                    ),
                    (
                        "right",
                        self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        self.mp_pose.PoseLandmark.RIGHT_HIP,
                        self.mp_pose.PoseLandmark.RIGHT_KNEE,
                        self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                        self.mp_pose.PoseLandmark.RIGHT_WRIST,
                    ),
                ):
                    shoulder = landmarks[shoulder_idx]
                    hip = landmarks[hip_idx]
                    knee = landmarks[knee_idx]
                    ankle = landmarks[ankle_idx]
                    wrist = landmarks[wrist_idx]
                    visibility = (shoulder.visibility + hip.visibility + knee.visibility + ankle.visibility) / 4.0
                    side_candidates.append((visibility, side_name, shoulder, hip, knee, ankle, wrist))

                # Wybór dominującej strony i obliczenie widoczności
                visibility, side_name, shoulder, hip, knee, ankle, wrist = max(side_candidates, key=lambda item: item[0])
                left_vis = side_candidates[0][0]
                right_vis = side_candidates[1][0]

                # Detekcja czy to jest FRONT
                if abs(left_vis - right_vis) < 0.15:
                    side_name = "front"

                if side_name == "front":
                    # Wywołanie nowej, pustej funkcji dla widoku z przodu
                    metrics = self._analyze_front_view(frame_copy, landmarks)
                    current_knee_error = metrics.knee_error
                else:
                    # Dotychczasowa logika dla widoku z boku (left / right)
                    raw_knee_angle = self._normalize_view_angle(self._angle(hip, knee, ankle, use_z=False), 90.0)
                    raw_upper_angle = self._normalize_view_angle(self._angle(shoulder, hip, knee, use_z=False), 180.0)
                    knee_angle = self._smooth_angle(self._knee_angle_history, raw_knee_angle)
                    upper_body_angle = self._smooth_angle(self._upper_angle_history, raw_upper_angle)
                    
                    hand_raised = False
                    if shoulder.visibility > 0.4 and wrist.visibility > 0.4:
                        hand_raised = wrist.y < shoulder.y - 0.05

                    current_knee_error = self.detect_knee_error(landmarks)
                    bottom_ready = not math.isnan(knee_angle) and 87.0 <= knee_angle <= 95.0 and not math.isnan(upper_body_angle) and upper_body_angle < 160.0
                    top_ready = not math.isnan(upper_body_angle) and 170.0 <= upper_body_angle <= 183.0
                    
                    if top_ready:
                        message = "Pozycja górna: biodra na wysokości pleców"
                    elif bottom_ready:
                        message = "Pozycja startowa: ustaw biodro-kolano-stopa ~90°"
                    else:
                        message = "Ustaw pozycję startową lub unieś biodra wyżej"
                    
                    metrics = PoseMetrics(
                        pose_detected=True,
                        side=side_name,
                        knee_angle=knee_angle,
                        upper_body_angle=upper_body_angle,
                        hand_raised=hand_raised,
                        start_ready=not math.isnan(knee_angle) and 87.0 <= knee_angle <= 95.0,
                        top_ready=top_ready,
                        bottom_ready=bottom_ready,
                        pose_visibility=visibility,
                        knee_error=current_knee_error,
                        message=message,
                    )

            # Przypisanie ostatnich wartości globalnych dla analizatora
            self.last_knee_error = current_knee_error
            self.last_pose_metrics = metrics

            # Rysowanie nakładki na obraz (przekazujemy metrics, które wie czy jest front czy side)
            if results.pose_landmarks:
                self._draw_key_overlay(frame_copy, metrics, landmarks)
                
            return metrics, frame_copy
            
        except Exception as e:
            print(f"[POSE] BŁĄD: {e}")
            import traceback
            traceback.print_exc()
            self.last_pose_metrics = PoseMetrics(False)
            return self.last_pose_metrics, frame

    def _draw_key_overlay(self, frame, metrics: PoseMetrics, landmarks) -> None:
        """Draw only the important pose lines and angle markers."""

        try:
            h, w = frame.shape[:2]

            def pt(lm):
                return (int(lm.x * w), int(lm.y * h))

            color_body = (255, 255, 255)
            color_knee = (0, 255, 255)
            color_top = (0, 200, 255)
            color_joint = (0, 0, 255)
            
            if metrics.side == "front":
                shoulder_left = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                hip_left = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                knee_left = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                ankle_left = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                wrist_left = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]

                shoulder_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                hip_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                knee_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                ankle_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                wrist_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                # Draw both sides for front view
                cv2.line(frame, pt(shoulder_left), pt(hip_left), color_body, 2)
                cv2.line(frame, pt(hip_left), pt(knee_left), color_knee, 4)
                cv2.line(frame, pt(knee_left), pt(ankle_left), color_knee, 4)
                cv2.circle(frame, pt(hip_left), 5, color_joint, -1)
                cv2.circle(frame, pt(knee_left), 5, color_joint, -1)
                cv2.circle(frame, pt(ankle_left), 4, color_joint, -1)

                cv2.line(frame, pt(shoulder_right), pt(hip_right), color_body, 2)
                cv2.line(frame, pt(hip_right), pt(knee_right), color_knee, 4)
                cv2.line(frame, pt(knee_right), pt(ankle_right), color_knee, 4)
                cv2.circle(frame, pt(hip_right), 5, color_joint, -1)
                cv2.circle(frame, pt(knee_right), 5, color_joint, -1)
                cv2.circle(frame, pt(ankle_right), 4, color_joint, -1)

                if metrics.hand_raised:
                    if wrist_left.y < shoulder_left.y - 0.05:
                        cv2.line(frame, pt(shoulder_left), pt(wrist_left), color_top, 2)
                        cv2.circle(frame, pt(wrist_left), 4, color_top, -1)
                    if wrist_right.y < shoulder_right.y - 0.05:
                        cv2.line(frame, pt(shoulder_right), pt(wrist_right), color_top, 2)
                        cv2.circle(frame, pt(wrist_right), 4, color_top, -1)
            else:          
                if metrics.side == "left":
                    shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                    hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                    knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
                    ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
                    wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                else:
                    shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                    knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                    ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                    wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

                # key skeleton: shoulder-hip-knee-ankle and a single arm reference
                cv2.line(frame, pt(shoulder), pt(hip), color_body, 2)
                cv2.line(frame, pt(hip), pt(knee), color_knee, 4)
                cv2.line(frame, pt(knee), pt(ankle), color_knee, 4)
                cv2.circle(frame, pt(hip), 5, color_joint, -1)
                cv2.circle(frame, pt(knee), 5, color_joint, -1)
                cv2.circle(frame, pt(ankle), 4, color_joint, -1)

                if metrics.hand_raised:
                    cv2.line(frame, pt(shoulder), pt(wrist), color_top, 2)
                    cv2.circle(frame, pt(wrist), 4, color_top, -1)
                
        except Exception:
            pass

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
