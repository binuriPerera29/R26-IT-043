"""
run_all.py — Launch all 4 Flask backends simultaneously
Usage: python run_all.py
Stop:  Ctrl+C  (kills all child processes cleanly)
"""

import subprocess
import sys
import os
import signal
import time

APPS = [
    {
        "name":    "Glaucoma Detection",
        "path":    "glaucoma_app.py", 
        "port":    5004,
    },
    {
        "name":    "Diabetic Retinopathy",
        "path":    "dr.py",
        "port":    5001,
    },
    {
        "name":    "OCT Analysis",
        "path":    "oct.py",
        "port":    5000,
    },
    {
        "name":    "Cataract Detection",
        "path":    "cataract.py",
        "port":    5002,
    },
]

processes = []


def launch_all():
    for app in APPS:
        if not os.path.exists(app["path"]):
            print(f"[WARN] '{app['path']}' not found — skipping {app['name']}")
            continue

        proc = subprocess.Popen(
            [sys.executable, app["path"]],
            # Each app reads its own PORT env var if set; otherwise uses its default
            env={**os.environ, "PORT": str(app["port"])},
        )
        processes.append(proc)
        print(f"[OK]  Started '{app['name']}' (PID {proc.pid}) → http://localhost:{app['port']}")
        time.sleep(0.5)   # small stagger so log output doesn't interleave badly


def shutdown(sig=None, frame=None):
    print("\n[INFO] Shutting down all servers…")
    for proc in processes:
        proc.terminate()
    for proc in processes:
        proc.wait()
    print("[INFO] All servers stopped.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("=" * 55)
    print("  Eye Disease Analysis Platform — Multi-App Launcher")
    print("=" * 55)
    launch_all()
    print("-" * 55)
    print("  All servers running. Press Ctrl+C to stop all.\n")

    # Keep the launcher alive while children run
    for proc in processes:
        proc.wait()