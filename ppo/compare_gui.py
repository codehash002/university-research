"""Open PPO and SUMO built-in side by side in two GUIs.

    python compare_gui.py --model models/run_3lane_3h/best_model.zip --route 50
"""
import argparse, os, subprocess, sys, time


def main():
    ap = argparse.ArgumentParser(description="PPO vs SUMO built-in in two GUIs")
    ap.add_argument("--model", default="models/run_3lane_3h/best_model.zip")
    ap.add_argument("--route", type=int, default=50)
    ap.add_argument("--delay", type=int, default=120, help="ms per SUMO second")
    ap.add_argument("--horizon", type=int, default=900, help="sim seconds")
    a = ap.parse_args()
    py = sys.executable

    if not os.path.exists(a.model):
        sys.exit(f"Model not found: {a.model}")

    print("=" * 60)
    print("  1) PPO AI")
    print("  2) SUMO built-in")
    print(f"  Traffic: routes_{a.route}.rou.xml   delay={a.delay} ms")
    print("=" * 60)

    ppo = subprocess.Popen([
        py, "demo.py", "--model", a.model,
        "--route", str(a.route), "--delay", str(a.delay),
        "--horizon", str(a.horizon),
    ])
    time.sleep(2)
    builtin = subprocess.Popen([
        py, "preview.py",
        "--route", str(a.route), "--delay", str(a.delay),
        "--horizon", str(a.horizon),
    ])

    ppo.wait()
    builtin.wait()
    print("\nBoth windows finished.")


if __name__ == "__main__":
    main()
