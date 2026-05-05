import serial
import threading

class SerialBPMReader:
    def __init__(self, port='COM3', baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.arduino = None
        self.running = False
        self.bpm = 0
        self.thread = None
        self.callback = None  # Function to call with new BPM
        self.completion_callback = None  # Function to call when reading completes

    def start(self, callback=None, completion_callback=None):
        self.arduino = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self.running = True
        self.callback = callback
        self.completion_callback = completion_callback
        self.thread = threading.Thread(target=self.read_bpm, daemon=True)
        self.thread.start()

    def read_bpm(self):
        # Wait for Arduino ready
        print("BPM Reader: Waiting for Arduino to be ready...")
        while self.running:
            line = self.arduino.readline().decode(errors='ignore').strip()
            if line:
                print(f"Arduino: {line}")
                if "Press 'q' to start" in line:
                    print("BPM Reader: Arduino ready, sending 'q'...")
                    self.arduino.write(b'q')
                    break
        # Read BPM values
        print("BPM Reader: Starting to read BPM values...")
        while self.running:
            line = self.arduino.readline().decode(errors='ignore').strip()
            if line:
                print(f"Arduino: {line}")
                try:
                    if 'BPM' in line.upper():
                        # Split by colon and get the part after it
                        if ':' in line:
                            bpm_part = line.split(':')[1].strip()
                        else:
                            bpm_part = line
                        
                        # Extract only consecutive digits from the start
                        bpm_str = ''
                        for char in bpm_part:
                            if char.isdigit():
                                bpm_str += char
                            else:
                                break  # Stop at first non-digit
                        
                        if bpm_str:
                            bpm_val = int(bpm_str)
                            self.bpm = bpm_val
                            print(f"Parsed BPM: {bpm_val}")
                            if self.callback:
                                self.callback(bpm_val)
                except (ValueError, IndexError):
                    pass
                if "Press 'q' to start" in line:
                    print("BPM Reader: Measurement complete.")
                    break
        self.arduino.close()
        # Notify completion
        if self.completion_callback:
            self.completion_callback()

    def stop(self):
        self.running = False
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
