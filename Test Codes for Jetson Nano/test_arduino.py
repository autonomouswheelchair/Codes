import serial
import time

# Replace 'COM4' with the port your Arduino is connected to
ser = serial.Serial('COM4', 115200, timeout=1)
time.sleep(2) # Wait for the serial connection to be established

while True:
    line = ser.readline() # Read a line from the serial port
    if line:
        try:
            string = line.strip().decode('utf-8') # Decode the byte string into a Unicode string
            print(string) # Print the received string
        except UnicodeDecodeError  as e:
           print(".")
           pass
