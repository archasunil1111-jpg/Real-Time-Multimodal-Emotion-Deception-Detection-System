import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
from eye_blink import EyeBlinkDetector
from head_pose import HeadPoseDetector
from shoulder_analysis import ShoulderAnalyzer
from eye_ball import EyeBallTracker
from bpm_serial_reader import SerialBPMReader
import mediapipe as mp
import os


class LoginWindow:
    """Login interface for the lie detection system"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Lie Detection System - Login")
        
        # Get screen dimensions and calculate adaptive window size (90% of screen)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(False, False)
        self.root.configure(bg="#2c3e50")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup login UI elements"""
        # Title
        title_frame = tk.Frame(self.root, bg="#34495e")
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            title_frame,
            text="LIE DETECTION SYSTEM",
            font=("Arial", 18, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        )
        title_label.pack(pady=15)
        
        # Center container for login form (480p width: 854x480)
        center_frame = tk.Frame(self.root, bg="#2c3e50")
        center_frame.pack(expand=True, fill=tk.BOTH)
        
        # Login form frame (480p: 854x480)
        login_frame = tk.Frame(
            center_frame,
            bg="#34495e",
            width=854,
            height=480,
            relief=tk.RAISED,
            bd=3
        )
        login_frame.pack(expand=True, pady=20)
        login_frame.pack_propagate(False)
        
        # Username
        tk.Label(
            login_frame,
            text="Username:",
            font=("Arial", 13, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        ).pack(pady=(40, 10), anchor=tk.W, padx=40)
        
        self.username_entry = tk.Entry(
            login_frame,
            font=("Arial", 12),
            width=35,
            bg="#ecf0f1",
            fg="#2c3e50"
        )
        self.username_entry.pack(pady=(0, 20), padx=40, fill=tk.X)
        self.username_entry.focus()
        
        # Password
        tk.Label(
            login_frame,
            text="Password:",
            font=("Arial", 13, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        ).pack(pady=(10, 10), anchor=tk.W, padx=40)
        
        self.password_entry = tk.Entry(
            login_frame,
            font=("Arial", 12),
            width=35,
            bg="#ecf0f1",
            fg="#2c3e50",
            show="●"
        )
        self.password_entry.pack(pady=(0, 30), padx=40, fill=tk.X)
        self.password_entry.bind("<Return>", lambda e: self.login())
        
        # Login button
        login_btn = tk.Button(
            login_frame,
            text="LOGIN",
            font=("Arial", 13, "bold"),
            bg="#3498db",
            fg="white",
            command=self.login,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2,
            padx=30,
            pady=10
        )
        login_btn.pack(pady=10, padx=40, fill=tk.X)
        
        # Exit button
        exit_btn = tk.Button(
            login_frame,
            text="EXIT",
            font=("Arial", 12),
            bg="#e74c3c",
            fg="white",
            command=self.root.quit,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2,
            padx=30,
            pady=10
        )
        exit_btn.pack(pady=10, padx=40, fill=tk.X)
    
    def login(self):
        """Verify login credentials"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if username == "liedet" and password == "liedet":
            self.root.destroy()
            # Launch main application
            root = tk.Tk()
            app = MainApplication(root)
            root.mainloop()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password!\n\nUsername: liedet\nPassword: liedet")
            self.password_entry.delete(0, tk.END)
            self.username_entry.delete(0, tk.END)
            self.username_entry.focus()


class MainApplication:
    """Main application window for lie detection analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Lie Detection System - Analysis")
        
        # Get screen dimensions and calculate adaptive window size (90% of screen)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Center window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(False, False)
        self.root.configure(bg="#2c3e50")
        
        # Store window dimensions for adaptive UI
        self.window_width = window_width
        self.window_height = window_height
        
        # Application state
        self.is_analyzing = False
        self.analysis_thread = None
        self.cap = None
        self.frame_count = 0
        self.analysis_duration = 40  # 40 seconds
        self.start_time = None
        self.last_update_time = 0  # For throttling UI updates
        
        # Detectors
        self.eye_detector = None
        self.head_detector = None
        self.shoulder_analyzer = None
        self.eye_ball_tracker = None
        self.shared_face_mesh = None
        
        # Results storage
        self.analysis_results = {
            'eye_blinks': 0,
            'eye_flag': 0,
            'head_aversion_events': 0,
            'head_flag': 0,
            'shoulder_fidget_events': 0,
            'shoulder_flag': 0,
            'eyeball_flag': 0,
            'bpm': 0,
            'bpm_flag': 0,
            'lie_score': 0.0,
            'deviations': []
        }
        self.bpm_reader = None
        
        self.setup_ui()
        
        # Pre-initialize detectors in background to speed up first analysis
        threading.Thread(target=self.pre_initialize_detectors, daemon=True).start()
    
    def setup_ui(self):
        """Setup main application UI"""
        # Header
        header_frame = tk.Frame(self.root, bg="#34495e", height=60)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text="LIE DETECTION SYSTEM - REAL-TIME ANALYSIS",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        )
        header_label.pack(pady=15)
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg="#2c3e50")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Video display
        left_frame = tk.Frame(content_frame, bg="#34495e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(
            left_frame,
            text="VIDEO FEED",
            font=("Arial", 11, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        ).pack(pady=5)
        
        # Calculate adaptive canvas size (60% of window width, maintain 16:9 aspect ratio)
        self.canvas_width = int(self.window_width * 0.6)
        self.canvas_height = int(self.canvas_width * 9 / 16)  # 16:9 aspect ratio
        
        # Ensure canvas fits in available height
        max_canvas_height = self.window_height - 250  # Leave space for header, buttons, status
        if self.canvas_height > max_canvas_height:
            self.canvas_height = max_canvas_height
            self.canvas_width = int(self.canvas_height * 16 / 9)
        
        # Canvas for video display
        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#000000",
            highlightthickness=2,
            highlightbackground="#3498db"
        )
        self.canvas.pack(pady=5)
        
        # Control buttons
        button_frame = tk.Frame(left_frame, bg="#34495e")
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = tk.Button(
            button_frame,
            text="START ANALYSIS (40s)",
            font=("Arial", 11, "bold"),
            bg="#27ae60",
            fg="white",
            command=self.start_analysis,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        self.start_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="STOP",
            font=("Arial", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            command=self.stop_analysis,
            cursor="hand2",
            state=tk.DISABLED,
            relief=tk.RAISED,
            bd=2
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Calibration button
        self.calib_btn = tk.Button(
            button_frame,
            text="CALIBRATION",
            font=("Arial", 11, "bold"),
            bg="#2980b9",
            fg="white",
            command=self.run_calibration,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        self.calib_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # BPM button frame (below main buttons)
        bpm_button_frame = tk.Frame(left_frame, bg="#34495e")
        bpm_button_frame.pack(fill=tk.X, pady=5)
        
        self.bpm_btn = tk.Button(
            bpm_button_frame,
            text="READ BPM",
            font=("Arial", 10, "bold"),
            bg="#9b59b6",
            fg="white",
            command=self.start_bpm_reading,
            cursor="hand2",
            relief=tk.RAISED,
            bd=2
        )
        self.bpm_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # BPM Simulator frame - COMPACT VERSION
        simulator_frame = tk.LabelFrame(
            left_frame, 
            text="BPM SIMULATOR",
            bg="#34495e",
            fg="#ecf0f1",
            font=("Arial", 9, "bold"),
            relief=tk.RAISED,
            bd=2
        )
        simulator_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Top row: Checkbox and Slider side by side
        top_row = tk.Frame(simulator_frame, bg="#34495e")
        top_row.pack(fill=tk.X, pady=5, padx=5)
        
        # Left: Checkbox
        self.use_simulator = tk.BooleanVar(value=False)
        simulator_checkbox = tk.Checkbutton(
            top_row,
            text="Enable",
            font=("Arial", 8, "bold"),
            bg="#34495e",
            fg="#ecf0f1",
            activebackground="#34495e",
            activeforeground="#ecf0f1",
            selectcolor="#2c3e50",
            variable=self.use_simulator,
            cursor="hand2"
        )
        simulator_checkbox.pack(side=tk.LEFT, padx=5)
        
        # Right: Slider
        tk.Label(
            top_row,
            text="BPM:",
            font=("Arial", 8, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        ).pack(side=tk.LEFT, padx=2)
        
        self.bpm_slider = tk.Scale(
            top_row,
            from_=0,
            to=200,
            orient=tk.HORIZONTAL,
            length=250,
            width=12,
            command=self.update_bpm_from_slider,
            bg="#34495e",
            fg="white",
            troughcolor="#3498db",
            highlightthickness=0
        )
        self.bpm_slider.set(80)
        self.bpm_slider.pack(side=tk.LEFT, padx=5)
        
        # BPM value display
        self.bpm_value_label = tk.Label(
            top_row,
            text="80",
            font=("Arial", 10, "bold"),
            bg="#34495e",
            fg="#2ecc71",
            width=4
        )
        self.bpm_value_label.pack(side=tk.LEFT, padx=2)
        
        # Bottom row: Tick marks
        tick_row = tk.Frame(simulator_frame, bg="#34495e")
        tick_row.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Add space for checkbox width
        tk.Label(tick_row, text="", bg="#34495e", width=8).pack(side=tk.LEFT)
        
        # Tick marks frame aligned with slider
        tick_frame = tk.Frame(tick_row, bg="#34495e")
        tick_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        for tick_value in [0, 50, 100, 150, 200]:
            tk.Label(
                tick_frame,
                text=str(tick_value),
                font=("Arial", 7),
                bg="#34495e",
                fg="#95a5a6"
            ).pack(side=tk.LEFT, expand=True)
        
        # Right side - Results panel (adaptive width: 35% of window)
        right_panel_width = int(self.window_width * 0.35)
        right_frame = tk.Frame(content_frame, bg="#34495e", width=right_panel_width)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        tk.Label(
            right_frame,
            text="ANALYSIS RESULTS",
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        ).pack(pady=10)
        
        # Flags Status Panel (always visible)
        flags_frame = tk.Frame(right_frame, bg="#2c3e50", relief=tk.SUNKEN, bd=2)
        flags_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        tk.Label(
            flags_frame,
            text="BEHAVIORAL FLAGS STATUS",
            font=("Arial", 10, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1"
        ).pack(pady=(5, 10))
        
        # Create flag labels
        self.flag_labels = {}
        
        flag_names = [
            ("eye_flag", "Eye Blink Flag"),
            ("head_flag", "Head Aversion Flag"),
            ("shoulder_flag", "Shoulder Flag"),
            ("eyeball_flag", "EyeBall Flag"),
            ("bpm_flag", "BPM Flag")
        ]
        
        for flag_key, flag_text in flag_names:
            flag_container = tk.Frame(flags_frame, bg="#2c3e50")
            flag_container.pack(fill=tk.X, padx=10, pady=3)
            
            tk.Label(
                flag_container,
                text=flag_text + ":",
                font=("Arial", 9),
                bg="#2c3e50",
                fg="#ecf0f1",
                anchor=tk.W,
                width=20
            ).pack(side=tk.LEFT)
            
            flag_label = tk.Label(
                flag_container,
                text="WAITING",
                font=("Arial", 9, "bold"),
                bg="#000000",
                fg="#FFFFFF",
                anchor=tk.W,
                width=12
            )
            flag_label.pack(side=tk.LEFT, padx=5)
            self.flag_labels[flag_key] = flag_label
        
        tk.Label(flags_frame, text="", bg="#2c3e50").pack(pady=2)  # Spacer
        
        # Deception Warning Label (hidden by default)
        self.warning_label = tk.Label(
            flags_frame,
            text="",
            font=("Arial", 11, "bold"),
            bg="#2c3e50",
            fg="#FF0000",
            wraplength=550
        )
        self.warning_label.pack(pady=5)
        
        # Scrollable results area
        results_scroll = tk.Frame(right_frame, bg="#34495e")
        results_scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(results_scroll)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(
            results_scroll,
            font=("Courier", 9),
            bg="#2c3e50",
            fg="#ecf0f1",
            height=30,
            width=70,
            yscrollcommand=scrollbar.set,
            relief=tk.SUNKEN,
            bd=2
        )
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Status bar
        status_frame = tk.Frame(self.root, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready for analysis",
            font=("Arial", 10),
            bg="#34495e",
            fg="#2ecc71"
        )
        self.status_label.pack(pady=5, anchor=tk.W)
    
    def update_status(self, message, color="#2ecc71"):
        """Update status bar message"""
        self.status_label.config(text=message, fg=color)
        self.root.update()
    
    def update_flag_display(self):
        """Update flag status indicators with colors, including BPM"""
        flag_configs = {
            'eye_flag': self.analysis_results.get('eye_flag', 0),
            'head_flag': self.analysis_results.get('head_flag', 0),
            'shoulder_flag': self.analysis_results.get('shoulder_flag', 0),
            'eyeball_flag': self.analysis_results.get('eyeball_flag', 0),
            'bpm_flag': self.analysis_results.get('bpm_flag', 0)
        }
        detected_count = 0
        for flag_key, flag_value in flag_configs.items():
            if flag_key == 'bpm_flag':
                bpm = self.analysis_results.get('bpm', 0)
                if bpm > 120:
                    self.flag_labels[flag_key].config(text=f"{bpm} (HIGH)", bg="#FF0000", fg="#FFFFFF")
                    detected_count += 1
                else:
                    self.flag_labels[flag_key].config(text=f"{bpm} (NORMAL)", bg="#00FF00", fg="#000000")
            else:
                if flag_value == 1:
                    detected_count += 1
                    self.flag_labels[flag_key].config(text="DETECTED", bg="#FF0000", fg="#FFFFFF")
                else:
                    self.flag_labels[flag_key].config(text="NORMAL", bg="#00FF00", fg="#000000")
        if detected_count >= 2:
            self.warning_label.config(
                text=f"*** DECEPTION DETECTED ***\n{detected_count} WARNING FLAGS ACTIVE",
                bg="#FF0000",
                fg="#FFFFFF"
            )
        else:
            self.warning_label.config(text="", bg="#2c3e50")
    
    def update_results_display(self, is_final=False):
        """Update the results text area, including BPM"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        results_text = "=" * 45 + "\n"
        if is_final:
            results_text += "FINAL ANALYSIS RESULTS\n"
        else:
            results_text += "ANALYSIS IN PROGRESS\n"
        results_text += "=" * 45 + "\n\n"
        detected_count = sum([
            self.analysis_results.get('eye_flag', 0),
            self.analysis_results.get('head_flag', 0),
            self.analysis_results.get('shoulder_flag', 0),
            self.analysis_results.get('eyeball_flag', 0),
            1 if self.analysis_results.get('bpm', 0) > 120 else 0
        ])
        if detected_count >= 2:
            results_text += "*** DECEPTION DETECTED ***\n"
            results_text += f"*** {detected_count} WARNING FLAGS ACTIVE ***\n\n"
        results_text += f"BEHAVIORAL ANALYSIS:\n"
        results_text += f"  - Eye Blinks: {self.analysis_results['eye_blinks']}\n"
        results_text += f"  - Head Movements: {self.analysis_results['head_aversion_events']}\n"
        results_text += f"  - Shoulder Movements: {self.analysis_results['shoulder_fidget_events']}\n"
        results_text += f"  - BPM: {self.analysis_results['bpm']}\n\n"
        results_text += f"STATUS FLAGS:\n"
        def get_flag_status(flag_value):
            if isinstance(flag_value, bool):
                return "DETECTED" if flag_value else "NORMAL"
            else:
                return "DETECTED" if int(flag_value) == 1 else "NORMAL"
        eye_status = get_flag_status(self.analysis_results.get('eye_flag', 0))
        results_text += f"  - Excessive Blinking: {eye_status}\n"
        head_status = get_flag_status(self.analysis_results.get('head_flag', 0))
        results_text += f"  - Looking Away: {head_status}\n"
        shoulder_status = get_flag_status(self.analysis_results.get('shoulder_flag', 0))
        results_text += f"  - Body Movement: {shoulder_status}\n"
        eyeball_status = get_flag_status(self.analysis_results.get('eyeball_flag', 0))
        results_text += f"  - Eye Gaze Shift: {eyeball_status}\n"
        bpm = self.analysis_results.get('bpm', 0)
        bpm_flag = "HIGH" if bpm > 120 else "NORMAL"
        results_text += f"  - BPM: {bpm} ({bpm_flag})\n\n"
        if detected_count >= 2:
            results_text += f"RESULT: DECEPTION LIKELY\n"
            results_text += f"({detected_count} suspicious behaviors detected)\n\n"
        elif detected_count == 1:
            results_text += f"RESULT: MINOR CONCERN\n"
            results_text += f"(1 suspicious behavior detected)\n\n"
        else:
            results_text += f"RESULT: NO DECEPTION DETECTED\n"
            results_text += f"(All behaviors normal)\n\n"
        results_text += f"CONFIDENCE SCORE:\n"
        results_text += f"  - Score: {self.analysis_results['lie_score']:.2f}\n"
        results_text += f"  - Higher score = More suspicious\n\n"
        results_text += f"BEHAVIORS DETECTED:\n"
        if self.analysis_results['deviations']:
            for i, deviation in enumerate(self.analysis_results['deviations'], 1):
                results_text += f"  {i}. {deviation}\n"
        else:
            results_text += "  - None (Behavior appears normal)\n"
        results_text += "\n" + "=" * 45 + "\n"
        if is_final:
            results_text += "ANALYSIS COMPLETE!\n"
        results_text += "Time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n";
        self.results_text.insert(1.0, results_text)
        self.results_text.config(state=tk.DISABLED)
    
    def initialize_detectors(self):
        """Initialize all detection modules (only if not already initialized)"""
        try:
            # Skip if already initialized
            if self.shared_face_mesh is not None:
                return True
                
            self.shared_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True
            )
            
            self.eye_detector = EyeBlinkDetector(face_mesh=self.shared_face_mesh)
            self.head_detector = HeadPoseDetector(calibration_file='head_calibration.json')
            self.shoulder_analyzer = ShoulderAnalyzer()
            self.eye_ball_tracker = EyeBallTracker(face_mesh=self.shared_face_mesh, fps=30)
            
            return True
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize detectors:\n{str(e)}")
            return False
    
    def pre_initialize_detectors(self):
        """Pre-initialize detectors in background to speed up first analysis"""
        try:
            print("Pre-initializing detectors in background...")
            self.initialize_detectors()
            print("Detectors pre-initialized successfully!")
        except Exception as e:
            print(f"Pre-initialization warning: {e}")
    
    def start_analysis(self):
        """Start the 40-second analysis"""
        if self.is_analyzing:
            messagebox.showwarning("Analysis Running", "Analysis is already in progress!")
            return
        
        # Show loading status
        self.update_status("Initializing camera...", "#f39c12")
        self.root.update()
        
        # Try to open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Webcam Error", "Could not open webcam. Please check your camera.")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Show detector initialization status
        self.update_status("Loading AI models...", "#f39c12")
        self.root.update()
        
        if not self.initialize_detectors():
            self.cap.release()
            return
        
        # Double-check all detectors are ready
        if not all([self.eye_ball_tracker, self.eye_detector, self.head_detector, self.shoulder_analyzer]):
            messagebox.showerror("Initialization Error", "Detectors not fully initialized. Please try again.")
            self.cap.release()
            return
        
        # Reset all detector states
        if self.eye_ball_tracker:
            self.eye_ball_tracker.reset()
        if self.eye_detector:
            self.eye_detector.reset()
        if self.head_detector:
            self.head_detector.reset()
        if self.shoulder_analyzer:
            self.shoulder_analyzer.reset()
        
        self.analysis_results = {
            'eye_blinks': 0,
            'eye_flag': 0,
            'head_aversion_events': 0,
            'head_flag': 0,
            'shoulder_fidget_events': 0,
            'shoulder_flag': 0,
            'eyeball_flag': 0,
            'bpm': 0,
            'bpm_flag': 0,
            'lie_score': 0.0,
            'deviations': []
        }
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update_time = time.time()
        for flag_label in self.flag_labels.values():
            flag_label.config(text="WAITING", bg="#000000", fg="#FFFFFF")
        self.warning_label.config(text="", bg="#2c3e50")
        self.is_analyzing = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_status("ANALYZING (40 seconds)...", "#e74c3c")
        self.analysis_thread = threading.Thread(target=self.analyze_video, daemon=True)
        self.analysis_thread.start()
    
    def analyze_video(self):
        """Main analysis loop (runs for 40 seconds)"""
        try:
            while self.is_analyzing:
                elapsed_time = time.time() - self.start_time
                
                # Stop after 40 seconds
                if elapsed_time >= self.analysis_duration:
                    self.is_analyzing = False
                    break
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip for selfie view
                frame = cv2.flip(frame, 1)
                self.frame_count += 1
                
                # Process frame with all detectors
                try:
                    # Check if detectors are initialized
                    if not all([self.eye_ball_tracker, self.eye_detector, self.head_detector, self.shoulder_analyzer]):
                        continue
                    
                    # Eye ball tracking
                    annotated_frame, gaze_text, eyeball_flag = self.eye_ball_tracker.process_frame(frame, annotate=True)
                    frame = annotated_frame
                    
                    # Eye blink detection
                    eye_data = self.eye_detector.process_frame(frame.copy())
                    if eye_data is None or 'image' not in eye_data:
                        continue
                    frame = eye_data['image']
                    
                    # Head pose detection
                    head_data = self.head_detector.process_frame(frame)
                    if head_data is None or 'image' not in head_data:
                        continue
                    frame = head_data['image']
                    
                    # Shoulder analysis
                    shoulder_data = self.shoulder_analyzer.process_frame(frame)
                    if shoulder_data is None or 'image' not in shoulder_data:
                        continue
                    frame = shoulder_data['image']
                    
                    # Update results
                    self.analysis_results['eye_blinks'] = eye_data['blinks']
                    self.analysis_results['eye_flag'] = int(eye_data.get('eye_flag', 0))
                    self.analysis_results['head_aversion_events'] = head_data['aversion_events']
                    self.analysis_results['head_flag'] = int(head_data.get('lie_chance_flag', 0))
                    self.analysis_results['shoulder_fidget_events'] = shoulder_data['fidget_events']
                    self.analysis_results['shoulder_flag'] = int(shoulder_data.get('shoulder_flag', 0))
                    self.analysis_results['eyeball_flag'] = int(eyeball_flag) if eyeball_flag is not None else 0
                    
                    # Calculate composite lie score
                    blink_rate = eye_data['blinks'] / (elapsed_time + 1)
                    self.analysis_results['lie_score'] = (blink_rate * 10) + \
                                                        (head_data['aversion_events'] * 20) + \
                                                        (shoulder_data['fidget_events'] * 15)
                    
                    # Generate deviations
                    deviations = []
                    if eye_data['eye_flag'] == 1:
                        deviations.append("Excessive blinking detected")
                    if head_data.get('lie_chance_flag', 0) == 1:
                        deviations.append("Head aversion detected")
                    if shoulder_data['shoulder_flag'] == 1:
                        deviations.append("Shoulder asymmetry detected")
                    if eyeball_flag == 1:
                        deviations.append("Sustained gaze shift detected")
                    
                    self.analysis_results['deviations'] = deviations
                    
                    # Print BPM along with Eye metrics every 30 frames to avoid spam
                    if self.frame_count % 30 == 0:
                        ear_threshold = 0.23  # From EyeBlinkDetector
                        bpm = self.analysis_results['bpm']
                        blinks = eye_data['blinks']
                        print(f"Frame {self.frame_count} | BPM: {bpm} | EAR: {eye_data.get('ear', 0):.3f} | Threshold: {ear_threshold} | Blinks: {blinks}")
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
                
                # Add timer and frame count to display
                remaining_time = max(0, self.analysis_duration - elapsed_time)
                cv2.putText(
                    frame,
                    f"Time: {remaining_time:.1f}s | Frames: {self.frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Add countdown timer at bottom left corner
                countdown = int(self.analysis_duration - elapsed_time)
                cv2.putText(
                    frame,
                    str(countdown),
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    4
                )
                
                # Display frame on canvas
                self.display_frame(frame)
                
                # Update results display only every 0.5 seconds (not every frame)
                current_time = time.time()
                if current_time - self.last_update_time >= 0.5:
                    self.update_results_display()
                    self.update_flag_display()  # Update flag colors
                    self.last_update_time = current_time
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during analysis:\n{str(e)}")
        finally:
            self.cleanup_analysis()
    
    def display_frame(self, frame):
        """Display video frame on canvas"""
        try:
            # Resize frame to fit canvas (adaptive size)
            frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo  # Keep a reference
            
        except Exception as e:
            print(f"Display error: {e}")
    
    def display_black_frame(self):
        """Display a black frame on the video canvas (1280x720)"""
        import numpy as np
        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(black_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo
    
    def stop_analysis(self):
        """Stop the analysis"""
        self.is_analyzing = False
        self.update_status("Analysis stopped by user", "#f39c12")
        self.cleanup_analysis()
    
    def start_bpm_reading(self):
        """Start BPM reading independently or use simulator"""
        # If simulator is enabled, use slider value directly
        if self.use_simulator.get():
            bpm_val = int(self.bpm_slider.get())
            print(f"DEBUG: Using BPM Simulator value: {bpm_val}")
            self.analysis_results['bpm'] = bpm_val
            self.analysis_results['bpm_flag'] = 1 if bpm_val > 120 else 0
            self.update_flag_display()
            self.update_status(f"BPM (Simulated): {bpm_val}", "#2ecc71")
            return
        
        # Otherwise, read from Arduino
        if self.bpm_reader is None:
            self.bpm_reader = SerialBPMReader()
        
        def bpm_callback(bpm_val):
            print(f"DEBUG: Received BPM value: {bpm_val}")  # Debug output
            # Set minimum BPM threshold for Arduino readings only
            if bpm_val < 30:
                bpm_val = 72
                print(f"DEBUG: BPM too low, adjusted to {bpm_val}")
            self.analysis_results['bpm'] = bpm_val
            self.analysis_results['bpm_flag'] = 1 if bpm_val > 120 else 0
            # Update the display
            self.update_flag_display()
            self.update_status(f"BPM: {bpm_val}", "#3498db")
        
        def bpm_completion():
            print("DEBUG: BPM reading completed")
            self.bpm_btn.config(state=tk.NORMAL)
            self.update_status("BPM Reading Complete", "#2ecc71")
        
        self.bpm_reader.start(callback=bpm_callback, completion_callback=bpm_completion)
        self.bpm_btn.config(state=tk.DISABLED)
        self.update_status("BPM Reading Started", "#3498db")
    
    def update_bpm_from_slider(self, value):
        """Update BPM value display from slider"""
        self.bpm_value_label.config(text=str(int(float(value))))
    
    def cleanup_analysis(self):
        """Cleanup resources after analysis"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception as e:
            print(f"Cleanup camera error: {e}")
        
        # DO NOT close shared_face_mesh - reuse it for next analysis
        # DO NOT close detectors - they will be reused
        
        try:
            if self.bpm_reader is not None:
                self.bpm_reader.stop()
        except Exception as e:
            print(f"Cleanup BPM reader error: {e}")
        self.is_analyzing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Analysis Complete - Review Results", "#2ecc71")
        if self.frame_count > 0:
            self.update_results_display(is_final=True)
            self.update_flag_display()
            self.display_black_frame()
    
    def run_calibration(self):
        """Run head pose calibration for 40 seconds (100 frames) and save to head_calibration.json, showing video in canvas."""
        import numpy as np
        import mediapipe as mp
        import json
        import time
        import cv2
        def calibration_task():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Calibration Error", "Could not open webcam for calibration.")
                return
            face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
            yaw_list = []
            TARGET_FRAMES = 100
            frame_num = 0
            start_time = time.time()
            while frame_num < TARGET_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    left_eye = landmarks[33]
                    right_eye = landmarks[362]
                    nose = landmarks[1]
                    eye_center_x = (left_eye.x + right_eye.x) / 2
                    delta_x = nose.x - eye_center_x
                    yaw = np.degrees(np.arctan2(delta_x, 0.18))
                    yaw_list.append(yaw)
                # Show frame in Tkinter canvas (adaptive size)
                if frame.shape[1] == self.canvas_width and frame.shape[0] == self.canvas_height:
                    frame_disp = frame
                else:
                    frame_disp = cv2.resize(frame, (self.canvas_width, self.canvas_height))
                frame_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.image = photo
                # Show progress in status bar
                elapsed = time.time() - start_time
                self.update_status(f"Calibrating... {frame_num+1}/100 frames ({int(elapsed)}s)", "#2980b9")
                self.root.update()
                frame_num += 1
            cap.release()
            face_mesh.close()
            self.display_black_frame()  # Black out video after calibration
            if len(yaw_list) > 70:
                neutral_yaw = float(np.median(yaw_list))
                json.dump({"neutral_yaw": neutral_yaw}, open('head_calibration.json', 'w'), indent=2)
                messagebox.showinfo("Calibration Complete", f"Calibration successful!\nNeutral yaw: {neutral_yaw:.1f}° (saved)")
                self.update_status("Calibration complete. Ready for analysis.", "#2ecc71")
            else:
                messagebox.showerror("Calibration Failed", "Not enough face detections. Please try again.")
                self.update_status("Calibration failed. Try again.", "#e74c3c")
        # Run calibration in a thread to keep UI responsive
        threading.Thread(target=calibration_task, daemon=True).start()


def main():
    """Launch the application"""
    root = tk.Tk()
    app = LoginWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
