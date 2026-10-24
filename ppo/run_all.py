"""Build, train, evaluate (optional demo).

    python run_all.py --hours 3 --run-name run_3lane_3h --demo
"""
import argparse, glob, os, subprocess, sys, time


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=3.0, help="training hours (real time)")
    ap.add_argument("--run-name", default=time.strftime("run_%Y%m%d_%H%M"))
    ap.add_argument("--n-routes", type=int, default=60)
    ap.add_argument("--new-traffic", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--queue-weight", type=float, default=0.5)
    ap.add_argument("--switch-weight", type=float, default=0.05)
    ap.add_argument("--ped-weight", type=float, default=0.2)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--demo-route", type=int, default=50)
    a = ap.parse_args()
    py = sys.executable

    have_net = os.path.exists("scenario/net.net.xml") and os.path.exists("scenario/phases.json")
    if a.rebuild or not have_net:
        run([py, "make_scenario.py", "--n-routes", str(a.n_routes), "--lanes", str(a.lanes)])
    elif a.new_traffic or not glob.glob("scenario/routes_*.rou.xml"):
        run([py, "make_scenario.py", "--demand-only", "--n-routes", str(a.n_routes),
             "--seed", str(int(time.time()) % 100000)])

    run([py, "train.py", "--hours", str(a.hours), "--run-name", a.run_name,
         "--queue-weight", str(a.queue_weight), "--switch-weight", str(a.switch_weight),
         "--ped-weight", str(a.ped_weight)])

    best = os.path.join("models", a.run_name, "best_model.zip")
    model = best if os.path.exists(best) else os.path.join("models", a.run_name, "final.zip")
    run([py, "evaluate.py", "--model", model, "--csv", os.path.join("results", a.run_name + ".csv")])
    print("\nAll done. Results table:", os.path.join("results", a.run_name + ".csv"))

    if a.demo:
        print("\nOpening SUMO-GUI to demo the trained model...")
        run([py, "demo.py", "--model", model, "--route", str(a.demo_route)])
    else:
        print("Watch the AI live:  python demo.py --model", model)


if __name__ == "__main__":
    main()
