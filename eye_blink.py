import cv2
import numpy as np
import mediapipe as mp
import time

class EyeBlinkDetector:
    def __init__(self, face_mesh=None):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        # Accept external FaceMesh instance, or create if not provided
        if face_mesh is not None:
            self.face_mesh = face_mesh
            self._owns_mesh = False
        else:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.3, min_tracking_confidence=0.3
            )
            self._owns_mesh = True
        # Use proven 6-point eye landmarks for EAR
        self.l_eyelids = [362, 385, 387, 263, 373, 380]
        self.r_eyelids = [33, 160, 158, 133, 153, 144]
        self.ear_threshold = 0.23  # Lower threshold = less sensitive, avoids noise
        self.counter = 0
        self.total_blinks = 0
        self.blink_cooldown = 0  # Frames to wait before detecting next blink
        self.cooldown_frames = 10  # Wait 10 frames (~0.3 seconds) between blinks
        self.eye_flag = 0
        self.blink_threshold = 5  # If total_blinks > 5, then flag = 1
    
    def reset(self):
        """Reset detector state for new analysis"""
        self.counter = 0
        self.total_blinks = 0
        self.blink_cooldown = 0
        self.eye_flag = 0

    def eye_aspect_ratio(self, eye):
        # Classic EAR formula - proven to work
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    def process_frame(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        ear = 0.0
        frame_blinked = False
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                landmarks = np.array([[int(lm.x * w), int(lm.y * h)] 
                                    for lm in face_landmarks.landmark])
                
                left_eye = landmarks[self.l_eyelids]
                right_eye = landmarks[self.r_eyelids]
                
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                # Debug output every 10 frames to avoid spam
                if self.counter % 10 == 0 and ear > 0:
                    print(f"EAR: {ear:.3f}, Threshold: {self.ear_threshold}, Blinks: {self.total_blinks}")
                
                # Blink detection with cooldown to prevent multiple counts per blink
                if self.blink_cooldown > 0:
                    # In cooldown period, just decrement and skip detection
                    self.blink_cooldown -= 1
                    self.counter = 0
                elif ear < self.ear_threshold:
                    self.counter += 1
                else:
                    if self.counter >= 1:  # Just 1 frame is enough
                        self.total_blinks += 1
                        self.blink_cooldown = self.cooldown_frames  # Start cooldown
                    self.counter = 0
                
                # Clean eye contours
                self.mp_drawing.draw_landmarks(
                    image, face_landmarks, self.mp_face_mesh.FACEMESH_LEFT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                self.mp_drawing.draw_landmarks(
                    image, face_landmarks, self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        
        # Eye metrics - Top left corner
        cv2.putText(image, "EAR: {:.2f}".format(ear), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(image, "Blinks: {} | Eye Flag: {}".format(self.total_blinks, self.eye_flag), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Simple logic: if total_blinks > threshold, set flag = 1
        if self.total_blinks > self.blink_threshold:
            self.eye_flag = 1
        else:
            self.eye_flag = 0
        
        # Add delay to slow down processing (50ms per frame)
        time.sleep(0.05)
        
        return {
            'ear': ear,
            'blinks': self.total_blinks,
            'eye_flag': self.eye_flag,
            'image': image
        }
