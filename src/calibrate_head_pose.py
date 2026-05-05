import cv2
import mediapipe as mp
import numpy as np
import json
import os

def calibrate_head_pose(video_path=None, calib_file='head_calibration.json'):
    # AUTO-DETECT: video file OR webcam
    source = video_path if video_path and os.path.exists(video_path) else 0
    source_name = video_path or "WEBCAM"
    
    print(f"🔄 Calibrating from {source_name} (100 frames)... LOOK STRAIGHT!")
    
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Cannot open {source_name}")
        return
    
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    yaw_list = []
    TARGET_FRAMES = 100
    
    for frame_num in range(TARGET_FRAMES):
        ret, frame = cap.read()
        if not ret:
            print("❌ End of video")
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # PERFECT YAW LANDMARKS
            left_eye = landmarks[33]   # Left eye inner
            right_eye = landmarks[362] # Right eye inner
            nose = landmarks[1]        # Nose tip
            
            eye_center_x = (left_eye.x + right_eye.x) / 2
            delta_x = nose.x - eye_center_x
            yaw = np.degrees(np.arctan2(delta_x, 0.18))
            
            yaw_list.append(yaw)
            
            # GREEN LINE FEEDBACK
            h, w = frame.shape[:2]
            cv2.line(frame, 
                    (int(nose.x*w), int(nose.y*h)), 
                    (int(eye_center_x*w), int((left_eye.y+right_eye.y)/2 * h)), 
                    (0, 255, 0), 3)
        
        # PROGRESS
        cv2.putText(frame, f"Frame {frame_num+1}/100", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("CALIBRATE STRAIGHT", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(yaw_list) > 70:  # Need good data
        neutral_yaw = float(np.median(yaw_list))
        json.dump({"neutral_yaw": neutral_yaw}, open(calib_file, 'w'), indent=2)
        print(f"✅ SAVED: neutral_yaw = {neutral_yaw:.1f}° ({len(yaw_list)} frames)")
    else:
        print("❌ FAILED: Not enough face detections")

if __name__ == "__main__":
    import sys
    
    # Check for command-line argument
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        print(f"\n📹 Using video file: {video_file}\n")
        calibrate_head_pose(video_file)
    else:
        # Default: use webcam
        print("\n📷 Using WEBCAM for calibration\n")
        print("Usage: python calibrate_head_pose.py <video_file.avi>\n")
        calibrate_head_pose()
