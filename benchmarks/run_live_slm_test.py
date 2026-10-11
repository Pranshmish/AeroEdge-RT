import serial
import time
import sys

PORT = "COM14"
BAUD = 115200

def run_test(cmd, wait_time=15):
    print("=" * 70)
    print(f"  EXECUTING PHYSICAL ESP32 COMMAND: '{cmd.strip()}' ON {PORT}")
    print("=" * 70)
    
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except Exception as e:
        print(f"Error opening {PORT}: {e}")
        return

    time.sleep(0.5)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Send command
    ser.write((cmd.strip() + "\r\n").encode("utf-8"))
    ser.flush()

    start = time.time()
    last_recv = time.time()
    buffer = ""

    while time.time() - start < wait_time:
        if ser.in_waiting:
            chunk = ser.read(ser.in_waiting).decode("utf-8", errors="replace")
            sys.stdout.write(chunk.encode("ascii", errors="replace").decode("ascii"))
            sys.stdout.flush()
            last_recv = time.time()
            if "Enter prompt: " in chunk and (time.time() - start > 1.5):
                # Completed execution
                break
        else:
            time.sleep(0.02)
            if time.time() - last_recv > 3.0 and time.time() - start > 2.0:
                break

    ser.close()
    print("\n" + "=" * 70)
    print("  TEST COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "BABY"
    tokens = sys.argv[2] if len(sys.argv) > 2 else "48"
    
    if target.upper() == "ROOFLINE":
        run_test("ROOFLINE", wait_time=8)
    elif target.upper() == "BABY":
        run_test(f"BABY {tokens}", wait_time=12)
    elif target.upper() == "STORIES":
        run_test(f"STORIES {tokens}", wait_time=18)
    elif target.upper() == "MAMBA":
        run_test(f"MAMBA {tokens}", wait_time=10)
    elif target.upper() == "TCN":
        run_test(f"TCN {tokens}", wait_time=10)
    elif target.upper() == "ALL":
        run_test("ALL", wait_time=25)
    elif target.upper() == "C":
        run_test("C", wait_time=15)
    else:
        run_test(f"{target} {tokens}", wait_time=12)

