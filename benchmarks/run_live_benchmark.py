import serial
import time
import sys

def run_benchmark():
    port = "COM14"
    baud = 115200

    # User command from arguments or default
    if len(sys.argv) > 1:
        test_cmd = " ".join(sys.argv[1:]).strip()
    else:
        test_cmd = "COMPARE 1000: once upon a time"

    if not test_cmd.endswith("\n"):
        test_cmd += "\n"

    # Determine dynamic timeout
    timeout_sec = 60.0
    if "10000" in test_cmd:
        timeout_sec = 1200.0  # 20 minutes for 10k comparison
    elif "5000" in test_cmd:
        timeout_sec = 600.0
    elif "1000" in test_cmd:
        timeout_sec = 180.0
    elif "500" in test_cmd:
        timeout_sec = 100.0

    print(f"================================================================================")
    print(f"      AeroEdge-RT Live Silicon Benchmark Harness (Port: {port} @ {baud})        ")
    print(f"================================================================================")
    print(f"Target Command : {test_cmd.strip()}")
    print(f"Session Timeout: {timeout_sec:.0f} seconds")
    print(f"--------------------------------------------------------------------------------")

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print(f"Error opening serial port {port}: {e}")
        print("Please check that no other serial monitor (like PuTTY or Arduino IDE) is using COM14.")
        return

    # Trigger software reset on ESP32
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    time.sleep(0.8)

    print("Listening for ESP32 boot banner and CLI prompt...")
    start_time = time.time()
    boot_output = ""
    prompt_found = False

    while time.time() - start_time < 6.0:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='replace')
            sys.stdout.write(line)
            sys.stdout.flush()
            boot_output += line
            if "Enter prompt:" in line:
                prompt_found = True
                break
        else:
            time.sleep(0.02)

    if not prompt_found:
        print("\nSending newline trigger to wake prompt...")
        ser.write(b"\n")
        ser.flush()
        time.sleep(0.5)

    print(f"\n---> [SENDING TO ESP32]: {test_cmd.strip()}\n")
    ser.write(test_cmd.encode('utf-8'))
    ser.flush()

    capture_start = time.time()
    total_log = ""
    last_recv_time = time.time()

    # Stream real-time output from ESP32
    while time.time() - capture_start < timeout_sec:
        if ser.in_waiting:
            raw_bytes = ser.read(ser.in_waiting)
            data = raw_bytes.decode('utf-8', errors='replace')
            sys.stdout.buffer.write(raw_bytes)
            sys.stdout.buffer.flush()
            total_log += data
            last_recv_time = time.time()

            # Check if both benchmarks completed and returned to prompt
            if "Enter prompt:" in total_log and ("EXPERIMENT B" in total_log or "EXPERIMENT A" in total_log):
                # Ensure final telemetry line flushed
                if "TELEMETRY_JSON" in total_log and ("AeroEdge-RT" in total_log.split("TELEMETRY_JSON")[-1] or "Baseline" in total_log.split("TELEMETRY_JSON")[-1]):
                    time.sleep(0.5)
                    if ser.in_waiting:
                        more_data = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
                        sys.stdout.write(more_data)
                        total_log += more_data
                    break
        else:
            time.sleep(0.01)

    ser.close()

    log_path = "benchmarks/live_run_output.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(boot_output + "\n" + total_log)

    print(f"\n\n================================================================================")
    print(f"Benchmark finished in {time.time() - capture_start:.2f} seconds. Log saved to {log_path}")
    print(f"================================================================================")

if __name__ == "__main__":
    run_benchmark()
