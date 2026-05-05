import cv2
import numpy as np
import mediapipe as mp
import time

class HeadPoseDetector:
    def __init__(self, calibration_file=None):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            max_num_faces=1,
            refine_landmarks=True
        )
        self.aversion_events = 0
        self.lie_chance_events = 0
        self.lie_chance_flag = 0
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.is_calibrated = False
        
        # Smoothing for more stable detection
        self.yaw_history = []
        self.history_size = 5  # Average over 5 frames
        self.aversion_frame_threshold = 15  # Must exceed threshold for 15 consecutive frames
        self.aversion_counter = 0
    
    def reset(self):
        """Reset detector state for new analysis"""
        self.aversion_events = 0
        self.lie_chance_events = 0
        self.lie_chance_flag = 0
        self.yaw_history = []
        self.aversion_counter = 0
        
        # Threshold range: -45° to +45° (outside this range = lie risk)
        self.yaw_min_threshold = -30.0
        self.yaw_max_threshold = 30.0
        
        # 3D Model Points (generic human face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def process_frame(self, image):
        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        pitch, yaw, roll = 0, 0, 0
        head_risk = "TRUTHFUL"
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract 2D coordinates for the 6 key points
            # Landmark indices: 1=Nose tip, 199=Chin, 33=Left eye, 263=Right eye, 61=Left mouth, 291=Right mouth
            image_points = []
            for landmark_idx in [1, 199, 33, 263, 61, 291]:
                landmark = face_landmarks.landmark[landmark_idx]
                image_points.append((landmark.x * w, landmark.y * h))
            image_points = np.array(image_points, dtype="double")
            
            # Camera internals
            focal_length = w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ], dtype="double")
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            
            # Solve PnP to get rotation and translation vectors
            success, rot_vec, trans_vec = cv2.solvePnP(
                self.model_points, image_points, cam_matrix, dist_matrix
            )
            
            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            
            # Get Euler angles using RQDecomp3x3
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            pitch = angles[0]
            yaw = angles[1]
            roll = angles[2]
            
            # Apply calibration offset if available
            if self.is_calibrated:
                calibrated_yaw = yaw - self.neutral_yaw
            else:
                # No calibration - use raw yaw
                calibrated_yaw = yaw
            
            # Add to history for smoothing
            self.yaw_history.append(calibrated_yaw)
            if len(self.yaw_history) > self.history_size:
                self.yaw_history.pop(0)
            
            # Use smoothed value (average of recent frames)
            smoothed_yaw = np.mean(self.yaw_history) if self.yaw_history else calibrated_yaw

            # Simple logic: SAFE if -30 < yaw < 30, else LIE DETECTED
            if self.yaw_min_threshold < smoothed_yaw < self.yaw_max_threshold:
                head_risk = "SAFE"
                color = (0, 255, 0)
            else:
                head_risk = "LIE DETECTED"
                color = (0, 0, 255)
                self.lie_chance_events += 1
                if self.lie_chance_events >= 2:
                    self.lie_chance_flag = 1
                else:
                    self.lie_chance_flag = 0
                # Print all relevant flags for analysis
                # You must pass these flags from main.py when calling process_frame for each module
                try:
                    print(f"[LIE DETECTION ANALYSIS] Flags: Head={self.lie_chance_flag}, Eye={{eye_flag}}, EyeBall={{eye_ball_flag}}, Shoulder={{shoulder_flag}}")
                except Exception:
                    pass

            # Visual indicator line from nose to chin
            nose_2d = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))
            chin_2d = (int(face_landmarks.landmark[199].x * w), int(face_landmarks.landmark[199].y * h))
            cv2.line(image, nose_2d, chin_2d, color, 3)
        else:
            # No face detected
            return {
                'yaw': 0, 'pitch': 0, 'motion_variance': 0,
                'lie_risk': 'NO_FACE', 'aversion_events': self.aversion_events,
                'image': image
            }
        
        # Head pose metrics - Right side overlay with dynamic positioning
        frame_width = image.shape[1]
        right_x = max(10, frame_width - 400)  # Keep 400 pixels from right edge
        cv2.putText(image, f"Yaw: {yaw:.1f} | Pitch: {pitch:.1f}", (right_x, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Lie Risk: {head_risk}", (right_x, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if 'LIE' in head_risk else (0, 165, 255), 2)
        cv2.putText(image, f"Head Flag: {self.lie_chance_flag} (Aversions: {self.aversion_events})", (right_x, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return {
            'yaw': yaw,
            'pitch': pitch,
            'motion_variance': abs(yaw),
            'lie_risk': head_risk,
            'aversion_events': self.aversion_events,
            'lie_chance_events': self.lie_chance_events,
            'lie_chance_flag': self.lie_chance_flag,
            'image': image
        }
        
        # Add delay to slow down processing (50ms per frame)
        time.sleep(0.05)
