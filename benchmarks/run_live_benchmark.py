import serial
import time
import sys

def run_benchmark():
    port = "COM14"
    baud = 115200
    print(f"Connecting to {port} at {baud} baud...")
    
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        print(f"Error opening port: {e}")
        return

    # Reset ESP32 via DTR/RTS
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    time.sleep(1.0)
    
    print("Waiting for boot banner and prompt...")
    start_time = time.time()
    boot_output = ""
    prompt_found = False
    
    while time.time() - start_time < 8.0:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='replace')
            sys.stdout.write(line)
            sys.stdout.flush()
            boot_output += line
            if "Enter prompt:" in line:
                prompt_found = True
                break
        else:
            time.sleep(0.05)
            
    if not prompt_found:
        print("\nDid not see 'Enter prompt:' immediately, sending trigger anyway...")
        
    time.sleep(0.5)
    test_cmd = "COMPARE: once upon a time\n"
    print(f"\n---> Sending: {test_cmd.strip()}")
    ser.write(test_cmd.encode('utf-8'))
    ser.flush()
    
    capture_start = time.time()
    total_log = ""
    
    # Read output until benchmark finishes or 40s timeout
    while time.time() - capture_start < 45.0:
        if ser.in_waiting:
            data = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
            sys.stdout.write(data)
            sys.stdout.flush()
            total_log += data
            if "Enter prompt:" in total_log and ("EXPERIMENT B" in total_log or "EXPERIMENT A" in total_log):
                # Check if both reports completed
                if "EXPERIMENT B: AEROEDGE-RT" in total_log and "Tokens Generated" in total_log.split("EXPERIMENT B")[-1]:
                    time.sleep(1.0)
                    if ser.in_waiting:
                        data = ser.read(ser.in_waiting).decode('utf-8', errors='replace')
                        sys.stdout.write(data)
                        total_log += data
                    break
        else:
            time.sleep(0.05)
            
    ser.close()
    
    with open("benchmarks/live_run_output.txt", "w", encoding="utf-8") as f:
        f.write(boot_output + "\n" + total_log)
    print("\n\nLive benchmark execution finished. Saved to benchmarks/live_run_output.txt")

if __name__ == "__main__":
    run_benchmark()
