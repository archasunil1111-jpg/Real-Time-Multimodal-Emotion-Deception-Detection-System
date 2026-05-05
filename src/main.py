import cv2
from eye_blink import EyeBlinkDetector
from head_pose import HeadPoseDetector
from shoulder_analysis import ShoulderAnalyzer
import time
import sys
import os
import mediapipe as mp
from eye_ball import EyeBallTracker

def main(video_source=0, calibration_file='head_calibration.json'):
    """
    Main lie detection function
    
    Args:
        video_source: 0 for webcam, or path to video file (e.g., 'temp_video.avi')
    """
    # Determine if using webcam or video file
    is_webcam = isinstance(video_source, int)
    
    if not is_webcam:
        if not os.path.exists(video_source):
            print(f"Error: Video file '{video_source}' not found.")
            return
        print(f"\n{'='*60}")
        print(f"Analyzing video file: {video_source}")
        print(f"{'='*60}\n")
    else:
        print("Using webcam for live analysis")
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        if is_webcam:
            print("Error: Could not open webcam.")
        else:
            print(f"Error: Could not open video file '{video_source}'.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    
    # If FPS is invalid or zero, default to 30
    if fps == 0 or fps is None:
        fps = 30.0
    
    print(f"Camera/Video FPS: {fps:.1f}")
    
    # Shared MediaPipe FaceMesh for all detectors to reduce interference
    shared_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    eye_detector = EyeBlinkDetector(face_mesh=shared_face_mesh)
    head_detector = HeadPoseDetector(calibration_file=calibration_file)
    shoulder_analyzer = ShoulderAnalyzer()
    eye_ball_tracker = EyeBallTracker(face_mesh=shared_face_mesh, fps=fps)

    if is_webcam:
        print("Lie Detection System Active (Press 'q' to quit)")
    else:
        print(f"Analyzing video: {total_frames} frames at {fps:.1f} FPS")
        print("Press 'q' to quit, SPACE to pause/resume")
    
    frame_count = 0
    paused = False
    start_time = time.time()
    frame_width = None
    frame_height = None
    
    while True:
        if not paused:
            success, frame = cap.read()
            if not success:
                if is_webcam:
                    break
                else:
                    print("\nEnd of video reached.")
                    break
            
            frame_count += 1
            
            # Get frame dimensions on first frame
            if frame_width is None or frame_height is None:
                frame_height, frame_width = frame.shape[:2]
                print(f"Frame dimensions: {frame_width}x{frame_height}")
            
            # Only flip for webcam, not for recorded videos
            if is_webcam:
                frame = cv2.flip(frame, 1)

            # --- Integrated eyeball gaze with annotation ---
            # Get annotated frame with plus signs on iris centers
            annotated_frame, gaze_text, eye_ball_flag = eye_ball_tracker.process_frame(frame, annotate=True)
            frame = annotated_frame  # Use the annotated frame
            
            # Only report horizontal gaze changes to terminal
            if gaze_text and ('LEFT' in gaze_text or 'RIGHT' in gaze_text):
                print(f'Gaze: {gaze_text}')
            
            # 2) Eye blink processing
            eye_data = eye_detector.process_frame(frame.copy())
            frame = eye_data['image']
            
            # 3) Head pose
            head_data = head_detector.process_frame(frame)
            frame = head_data['image']
            
            # 4) Shoulder analysis
            shoulder_data = shoulder_analyzer.process_frame(frame)
            frame = shoulder_data['image']
            
            # Composite lie score (simple threshold)
            blink_rate = eye_data['blinks'] / (time.time() - start_time + 1)
            lie_score = (blink_rate * 10) + (head_data['aversion_events'] * 20) + (shoulder_data['fidget_events'] * 15)
            
            # Video progress indicator for recorded videos
            if not is_webcam:
                progress_text = f"Frame: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
                cv2.putText(frame, progress_text, (10, frame.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Bottom left - Lie Score (composite metric)
            cv2.putText(frame, f"Lie Score: {lie_score:.1f}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            # Add small delay between frame processing for smoother visualization
            time.sleep(0.025)
        
        # Display frame
        window_title = 'Lie Detection - Webcam' if is_webcam else f'Lie Detection - {os.path.basename(video_source)}'
        cv2.imshow(window_title, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not is_webcam:
            paused = not paused
            if paused:
                print("Video PAUSED. Press SPACE to resume.")
            else:
                print("Video RESUMED.")
    
    # cleanup
    # Close shared face mesh explicitly
    try:
        shared_face_mesh.close()
    except Exception:
        pass
    eye_ball_tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    print("\n" + "="*60)
    print("LIE DETECTION ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analysis Duration: {time.time()-start_time:.1f} seconds")
    print(f"Frames Processed: {frame_count}")
    print(f"Total Blinks: {eye_data['blinks']}")
    print(f"Head Aversion Events: {head_data['aversion_events']}")
    print(f"Shoulder Fidget Events: {shoulder_data['fidget_events']}")
    print(f"Final Lie Score: {lie_score:.1f}")
    print("-"*60)
    print(f"VIDEO FLAGS: Head={head_data.get('lie_chance_flag',0)}, Eye={eye_data.get('eye_flag',0)}, EyeBall={eye_ball_flag}, Shoulder={shoulder_data.get('shoulder_flag',0)}")
    print("="*60)

if __name__ == "__main__":
    # Check command line arguments for video file
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        
        print(f"\nStarting analysis of recorded video: {video_path}\n")
        
        start_time = time.time()
        main(video_source=video_path)
    else:
        # Default: use webcam
        print("\nNo video file specified. Using webcam.")
        print("Usage: python main.py <video_file.avi>\n")
        start_time = time.time()
        main(video_source=0)
