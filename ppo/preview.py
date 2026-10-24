"""SUMO-GUI with built-in traffic-light timing (no AI).

    python preview.py --route 50 --delay 150
"""
import argparse, os, sys
from traffic_env import MultiTlEnv


def main():
    ap = argparse.ArgumentParser(description="Preview scenario in SUMO-GUI")
    ap.add_argument("--scenario", default="scenario")
    ap.add_argument("--route", type=int, default=50, help="traffic file index")
    ap.add_argument("--horizon", type=int, default=900, help="sim seconds")
    ap.add_argument("--delay", type=int, default=120, help="ms per SUMO second")
    a = ap.parse_args()

    net = os.path.join(a.scenario, "net.net.xml")
    route = os.path.join(a.scenario, f"routes_{a.route}.rou.xml")
    if not os.path.exists(net) or not os.path.exists(route):
        sys.exit("Missing scenario. Run: python make_scenario.py --lanes 3")

    print("=" * 50)
    print("  THIS WINDOW = SUMO BUILT-IN CONTROLLER")
    print(f"  Traffic: {os.path.basename(route)}")
    print("=" * 50)

    env = MultiTlEnv(net, [route], horizon=a.horizon, control=False,
                     gui=True, gui_delay=a.delay, label="builtin_gui")
    obs, _ = env.reset(options={"route": route}, seed=42)
    done = False
    while not done:
        obs, r, te, tr, info = env.step(env.action_space.sample())
        done = te or tr
    env.close()
    print("SUMO built-in preview finished.")


if __name__ == "__main__":
    main()
