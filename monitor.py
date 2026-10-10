#!/usr/bin/env python3
"""
AeroEdge-RT Interactive Serial CLI & Benchmark Monitor
Connects to ESP32 on COM port, displays A/B testing output,
and provides an interactive terminal for sending prompts.
"""

import sys
import time
import argparse
import threading
import signal

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial is not installed. Install with: pip install pyserial")
    sys.exit(1)

running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("\n[Monitor] Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def auto_detect_port():
    ports = serial.tools.list_ports.comports()
    esp_keywords = ["CP210", "CH340", "CH9102", "FTDI", "USB-SERIAL", "Silicon Labs", "ESP"]
    for port in ports:
        desc = (port.description or "").upper()
        mfg  = (port.manufacturer or "").upper()
        for kw in esp_keywords:
            if kw.upper() in desc or kw.upper() in mfg:
                return port.device
    if ports:
        return ports[0].device
    return None

def reader_thread(ser):
    global running
    while running:
        try:
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            else:
                time.sleep(0.01)
        except Exception:
            break

def main():
    parser = argparse.ArgumentParser(description="AeroEdge-RT A/B Testing Interactive Monitor")
    parser.add_argument("--port", type=str, default="COM14", help="Serial COM port (default: COM14)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    if not port:
        print("ERROR: No serial port found. Specify with --port COMx")
        sys.exit(1)

    print(f"Connecting to {port} at {args.baud} baud...")
    try:
        ser = serial.Serial(port, args.baud, timeout=0.1)
    except Exception as e:
        print(f"Failed to open port {port}: {e}")
        sys.exit(1)

    print(f"[AeroEdge-RT] Connected to {port}. Resetting board...")
    # DTR/RTS toggle to reset ESP32
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)

    t = threading.Thread(target=reader_thread, args=(ser,), daemon=True)
    t.start()

    print("[Ready] Type prompt and press Enter. (Type 'exit' to quit)\n")
    try:
        while running:
            line = input()
            if line.strip().lower() == "exit":
                break
            ser.write((line + "\n").encode("utf-8"))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        ser.close()

if __name__ == "__main__":
    main()
