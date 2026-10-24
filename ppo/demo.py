"""Run trained PPO in SUMO-GUI.

    python demo.py --model models/run_3lane_3h/best_model.zip --route 50
"""
import argparse, glob, os, sys
from stable_baselines3 import PPO
from traffic_env import MultiTlEnv


def main():
    ap = argparse.ArgumentParser(description="Run trained PPO with SUMO-GUI")
    ap.add_argument("--model", required=True, help="path to best_model.zip or final.zip")
    ap.add_argument("--scenario", default="scenario")
    ap.add_argument("--route", type=int, default=None, help="traffic file index (e.g. 50)")
    ap.add_argument("--n-test", type=int, default=10, help="must match train.py")
    ap.add_argument("--horizon", type=int, default=1800)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--delay", type=int, default=100, help="ms between each SUMO second")
    ap.add_argument("--no-gui", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(a.model):
        sys.exit(f"Model not found: {a.model}")

    net = os.path.join(a.scenario, "net.net.xml")
    all_routes = sorted(glob.glob(os.path.join(a.scenario, "routes_*.rou.xml")))
    if not all_routes:
        sys.exit("No traffic files. Run: python make_scenario.py")
    test_routes = all_routes[-a.n_test:]

    if a.route is not None:
        chosen = os.path.join(a.scenario, f"routes_{a.route}.rou.xml")
        if not os.path.exists(chosen):
            sys.exit(f"Missing {chosen}")
        routes = [chosen]
    else:
        routes = test_routes

    print("=" * 50)
    print("  THIS WINDOW = YOUR PPO AI CONTROLLER")
    print(f"  Model: {a.model}")
    print("=" * 50)
    model = PPO.load(a.model)
    if not a.no_gui:
        mins = a.horizon * max(a.delay, 1) / 1000 / 60
        print(f"GUI delay={a.delay} ms/step → ~{mins:.0f} real minutes "
              f"(or use the Delay slider in SUMO-GUI).")
    env = MultiTlEnv(net, routes, horizon=a.horizon, gui=not a.no_gui,
                     gui_delay=0 if a.no_gui else a.delay, label="ppo_gui")

    obs_size = env.observation_space.shape[0]
    if model.observation_space.shape[0] != obs_size:
        env.close()
        sys.exit(
            f"Model expects obs size {model.observation_space.shape[0]}, "
            f"but current network gives {obs_size}. Retrain after network changes."
        )

    for ep in range(a.episodes):
        route = routes[ep % len(routes)]
        print(f"\n=== Episode {ep + 1}/{a.episodes}  traffic={os.path.basename(route)} ===")
        if not a.no_gui:
            print("SUMO-GUI should open. Close the GUI window or wait for the episode to finish.")
        obs, _ = env.reset(options={"route": route}, seed=42)
        done, step, total_r = False, 0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, te, tr, info = env.step(action)
            total_r += reward
            step += 1
            done = te or tr
            if step % 30 == 0:
                print(f"  t={env.t:4d}s  reward_so_far={total_r:7.1f}  "
                      f"wait={info.get('waiting_time', 0):.0f}  queue={info.get('queue', 0):.0f}")
        m = info.get("episode_metrics", {})
        print(f"Done. steps={step}  total_reward={total_r:.1f}  "
              f"avg_wait={m.get('avg_wait', float('nan')):.1f}  "
              f"avg_queue={m.get('avg_queue', float('nan')):.2f}  "
              f"arrived={m.get('arrived', '?')}")

    env.close()
    print("\nDemo finished.")


if __name__ == "__main__":
    main()
