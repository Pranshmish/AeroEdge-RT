import serial
import threading
import sys
import time

PORT = "COM14"
BAUD = 115200

def reader_thread(ser, stop_event):
    while not stop_event.is_set():
        try:
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                sys.stdout.write(data.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            else:
                time.sleep(0.01)
        except Exception as e:
            if not stop_event.is_set():
                print(f"\n[Read error: {e}]")
            break

def main():
    print("=" * 70)
    print("     AEROEDGE-RT INTERACTIVE SERIAL TERMINAL (ESP32-WROOM / COM14)")
    print("=" * 70)
    print(" Available Commands:")
    print("   • <any prompt>       -> Runs TinyLlama-260K inference (e.g. 'hello')")
    print("   • BABY [N]           -> Runs BabyLLaMA-1M for N tokens (e.g. 'BABY 50')")
    print("   • STORIES [N]        -> Runs Stories-3.2M for N tokens (e.g. 'STORIES 50')")
    print("   • COMPARE [N]: text  -> Runs Baseline vs AeroEdge-RT side-by-side")
    print("   • C                  -> Runs full 3-SLM Scaling Matrix (260K, 1M, 3.2M)")
    print("   • exit / quit        -> Exits this terminal")
    print("=" * 70)
    print("[Connecting to COM14 @ 115200 baud...]\n")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except Exception as e:
        print(f"Error opening {PORT}: {e}")
        return

    stop_event = threading.Event()
    t = threading.Thread(target=reader_thread, args=(ser, stop_event), daemon=True)
    t.start()

    # Send a newline to get prompt
    time.sleep(0.5)
    ser.write(b"\r\n")

    try:
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip().lower() in ["exit", "quit"]:
                break
            ser.write((line + "\n").encode("utf-8"))
            ser.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        time.sleep(0.2)
        ser.close()
        print("\n[Terminal closed.]")

if __name__ == "__main__":
    main()
