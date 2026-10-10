import serial
import time
import sys

PORT = "COM14"
BAUD = 115200

def run_test(cmd, wait_time=15):
    print(f"\n========================================================")
    print(f"  SENDING COMMAND TO PHYSICAL ESP32: '{cmd.strip()}'")
    print(f"========================================================\n")
    
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except Exception as e:
        print(f"Error opening {PORT}: {e}")
        return

    # Clear buffers
    time.sleep(1.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Send command
    ser.write(cmd.encode("utf-8"))
    ser.flush()

    start = time.time()
    while time.time() - start < wait_time:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="replace")
            print(line, end="")
            sys.stdout.flush()
        else:
            time.sleep(0.05)

    ser.close()

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "BABY"
    tokens = sys.argv[2] if len(sys.argv) > 2 else "48"
    
    if target.upper() == "BABY":
        run_test(f"BABY {tokens}\n", wait_time=10)
    elif target.upper() == "STORIES":
        run_test(f"STORIES {tokens}\n", wait_time=15)
    elif target.upper() == "C":
        run_test("C\n", wait_time=15)
    else:
        run_test(f"{target} {tokens}\n", wait_time=10)
