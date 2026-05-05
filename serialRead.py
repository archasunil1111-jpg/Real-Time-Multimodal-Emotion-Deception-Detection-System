import serial
import time

arduino = serial.Serial('COM4', 9600, timeout=1)

print("Waiting for Arduino...")

# Wait until Arduino is ready
while True:
    line = arduino.readline().decode(errors='ignore').strip()
    if line:
        print(line)
        if "Press 'q' to start" in line:
            break

time.sleep(0.5)

print("\nSending 'q' to Arduino...\n")
arduino.write(b'q')

# Read one full measurement cycle, then exit
while True:
    line = arduino.readline().decode(errors='ignore').strip()
    if line:
        print(line)
        if "Press 'q' to start" in line:
            print("\nMeasurement complete. Closing program.")
            break

arduino.close()

