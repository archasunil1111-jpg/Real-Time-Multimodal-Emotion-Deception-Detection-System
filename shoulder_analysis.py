import cv2
import numpy as np
import mediapipe as mp
import time

class ShoulderAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Lie cues: shrug = raised shoulders, asymmetry = uneven
        self.left_shoulder_base = None
        self.right_shoulder_base = None
        self.shapes_history = []  # Track variance
        self.fidget_counter = 0
        self.fidget_events = 0
        self.shrug_threshold = 50  # px raise from baseline
        self.asymmetry_threshold = 0.15  # Increased to reduce rapid fluctuation (0.15 = 15% difference)
        self.frame_count = 0
        self.shoulder_flag = 0
    
    def reset(self):
        """Reset analyzer state for new analysis"""
        self.left_shoulder_base = None
        self.right_shoulder_base = None
        self.shapes_history = []
        self.fidget_counter = 0
        self.fidget_events = 0
        self.frame_count = 0
        self.shoulder_flag = 0

    def process_frame(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        shoulder_risk = "NORMAL"
        left_y, right_y = 0, 0
        shrug_left, shrug_right = False, False
        asymmetry = 0
        if results.pose_landmarks:
            h, w, _ = image.shape
            landmarks = results.pose_landmarks.landmark
            # MediaPipe Pose indices: 11=Left shoulder, 12=Right shoulder
            left_shoulder = np.array([int(landmarks[11].x * w), int(landmarks[11].y * h)])
            right_shoulder = np.array([int(landmarks[12].x * w), int(landmarks[12].y * h)])
            left_y, right_y = left_shoulder[1], right_shoulder[1]
            # Establish baseline (first 30 frames)
            if self.left_shoulder_base is None and self.frame_count > 30:
                self.left_shoulder_base = left_y
                self.right_shoulder_base = right_y
            # Shrug detection: shoulders raised > threshold
            if self.left_shoulder_base:
                shrug_left = (self.left_shoulder_base - left_y) > self.shrug_threshold
                shrug_right = (self.right_shoulder_base - right_y) > self.shrug_threshold
            # Asymmetry: |left_y - right_y| / avg_width
            avg_y = (left_y + right_y) / 2
            shoulder_width = np.abs(left_shoulder[0] - right_shoulder[0])
            asymmetry = np.abs(left_y - right_y) / shoulder_width if shoulder_width > 0 else 0
            
            # Simple logic: LIE DETECTED if asymmetry exceeds threshold, else SAFE
            if asymmetry > self.asymmetry_threshold:
                shoulder_risk = "LIE DETECTED"
                self.shoulder_flag = 1
                self.fidget_events += 1
            else:
                shoulder_risk = "SAFE"
            
            # Draw ONLY a single line between shoulders (no skeleton)
            color = (0, 0, 255) if shoulder_risk == "LIE DETECTED" else (0, 255, 0)
            cv2.line(image, tuple(left_shoulder), tuple(right_shoulder), color, 3)
            
            # Optional: small circles at shoulder points
            cv2.circle(image, tuple(left_shoulder), 5, color, -1)
            cv2.circle(image, tuple(right_shoulder), 5, color, -1)
        
        # Shoulder overlay - positioned below eye metrics (y=120)
        cv2.putText(image, f"Shoulder: {shoulder_risk} | Flag: {self.shoulder_flag}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if shoulder_risk == "LIE DETECTED" else (0, 255, 0), 2)
        
        self.frame_count += 1
        
        # Add delay to slow down processing (50ms per frame)
        time.sleep(0.05)
        
        return {
            'left_y': left_y,
            'right_y': right_y,
            'asymmetry': asymmetry,
            'shoulder_risk': shoulder_risk,
            'fidget_events': self.fidget_events,
            'shoulder_flag': self.shoulder_flag,
            'image': image
        }
