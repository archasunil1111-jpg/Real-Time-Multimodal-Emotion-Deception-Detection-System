"""
Standalone eyeball tracker using MediaPipe Face Mesh.
Usage:
    python eye_ball.py           # webcam
    python eye_ball.py video.mp4 # analyze video file

Shows overlay with iris center and inferred gaze direction (LEFT/RIGHT/UP/DOWN/CENTER)
Prints gaze estimates to the terminal per frame.
"""

import cv2
import numpy as np
import mediapipe as mp
import sys
import time

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices used here (MediaPipe FaceMesh):
# left iris: 468-472, right iris: 473-477
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
# eye corner candidates
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263


def get_iris_center(landmarks, indices, w, h):
    pts = [landmarks[i] for i in indices]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    cx = np.mean(xs) * w
    cy = np.mean(ys) * h
    return (int(cx), int(cy))


def get_eye_box(landmarks, outer_idx, inner_idx, w, h, pad=0.02):
    p_outer = landmarks[outer_idx]
    p_inner = landmarks[inner_idx]
    x1 = min(p_outer.x, p_inner.x)
    x2 = max(p_outer.x, p_inner.x)
    y1 = min(p_outer.y, p_inner.y) - pad
    y2 = max(p_outer.y, p_inner.y) + pad
    return (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))


def infer_gaze(iris_center, eye_box):
    x1, y1, x2, y2 = eye_box
    ex = (x1 + x2) / 2
    ey = (y1 + y2) / 2
    ew = max(1, x2 - x1)
    eh = max(1, y2 - y1)
    nx = (iris_center[0] - ex) / ew  # normalized -0.5..0.5 roughly
    ny = (iris_center[1] - ey) / eh
    # thresholds tuned empirically
    if nx < -0.18:
        horiz = 'LEFT'
    elif nx > 0.18:
        horiz = 'RIGHT'
    else:
        horiz = 'CENTER'
    if ny < -0.18:
        vert = 'UP'
    elif ny > 0.18:
        vert = 'DOWN'
    else:
        vert = 'CENTER'
    if horiz == 'CENTER' and vert == 'CENTER':
        return 'CENTER', nx, ny
    if horiz == 'CENTER':
        return vert, nx, ny
    if vert == 'CENTER':
        return horiz, nx, ny
    return horiz + '-' + vert, nx, ny


class EyeBallTracker:
    """Reusable eyeball tracker for integration with main.py

    Methods:
        process_frame(frame) -> gaze_text
        process_frame(frame, annotate=True) -> (annotated_frame, gaze_text)
        close()
    """
    def __init__(self, face_mesh=None, horiz_threshold=0.12, fps=30.0):
        # Accept an external face_mesh to avoid multiple MediaPipe instances
        if face_mesh is not None:
            self.face_mesh = face_mesh
            self._owns_mesh = False
        else:
            self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
            self._owns_mesh = True
        self.horiz_threshold = float(horiz_threshold)
        self.eye_ball_flag = 0
        self.gaze_history = []  # Track gaze direction
        
        # Calculate window size based on actual FPS (10 seconds)
        self.fps = fps
        self.window_size = int(fps * 10)  # 10 seconds worth of frames
        self.threshold_frames = int(fps * 1)  # 1 second of sustained gaze
        print(f"EyeBall Tracker: FPS={fps:.1f}, Window={self.window_size} frames (10s), Threshold={self.threshold_frames} frames (1s)")
    
    def reset(self):
        """Reset tracker state for new analysis"""
        self.eye_ball_flag = 0
        self.gaze_history = []

    def process_frame(self, frame, annotate=False):
        # compute gaze on the provided frame; by default do not modify the frame
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        gaze_text = ''
        annotated = frame.copy() if annotate else None
        if results and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # left
            l_center = get_iris_center(lm, LEFT_IRIS, w, h)
            l_box = get_eye_box(lm, LEFT_EYE_OUTER, LEFT_EYE_INNER, w, h)
            l_gaze, l_nx, l_ny = infer_gaze(l_center, l_box)
            # right
            r_center = get_iris_center(lm, RIGHT_IRIS, w, h)
            r_box = get_eye_box(lm, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, w, h)
            r_gaze, r_nx, r_ny = infer_gaze(r_center, r_box)

            # Simple fusion -> horizontal-only: use average normalized x
            # compute avg normalized x from left and right if available
            avg_nx = None
            try:
                l_n = float(l_nx)
            except Exception:
                l_n = None
            try:
                r_n = float(r_nx)
            except Exception:
                r_n = None
            if l_n is not None and r_n is not None:
                avg_nx = (l_n + r_n) / 2.0
            elif l_n is not None:
                avg_nx = l_n
            elif r_n is not None:
                avg_nx = r_n
            else:
                avg_nx = 0.0

            if avg_nx < -self.horiz_threshold:
                gaze_text = 'LEFT'
            elif avg_nx > self.horiz_threshold:
                gaze_text = 'RIGHT'
            else:
                gaze_text = ''
            # Track gaze history for flag
            if gaze_text in ['LEFT', 'RIGHT']:
                self.gaze_history.append(1)
            else:
                self.gaze_history.append(0)
            if len(self.gaze_history) > self.window_size:
                self.gaze_history.pop(0)
            if sum(self.gaze_history) > self.threshold_frames:
                self.eye_ball_flag = 1
            else:
                self.eye_ball_flag = 0

            if annotate:
                # Draw a large, thick, bright yellow plus sign at each iris center
                for center in [l_center, r_center]:
                    pass  # '+' drawing commented out
                cv2.rectangle(annotated, (l_box[0], l_box[1]), (l_box[2], l_box[3]), (255, 0, 0), 1)
                cv2.rectangle(annotated, (r_box[0], r_box[1]), (r_box[2], r_box[3]), (255, 0, 0), 1)
                cv2.putText(annotated, f'L:{gaze_text if gaze_text=="LEFT" else l_gaze}', (l_box[0], l_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
                cv2.putText(annotated, f'R:{gaze_text if gaze_text=="RIGHT" else r_gaze}', (r_box[0], r_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)

        if annotate:
            # Gaze info - dynamic positioning (right side, but adjusted for frame width)
            frame_width = annotated.shape[1]
            right_x = max(10, frame_width - 250)  # Keep 250 pixels from right edge
            cv2.putText(annotated, f'Gaze: {gaze_text}', (right_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated, f'EyeBall Flag: {self.eye_ball_flag}', (right_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add delay to slow down processing (50ms per frame)
            time.sleep(0.05)
            
            return annotated, gaze_text, self.eye_ball_flag
        
        # Add delay to slow down processing (50ms per frame)
        time.sleep(0.05)
        
        return gaze_text, self.eye_ball_flag

    def close(self):
        # release resources only if this instance owns the mesh
        if getattr(self, '_owns_mesh', False):
            self.face_mesh.close()
