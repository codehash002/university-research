"""Compare PPO vs baselines on held-out traffic.

    python evaluate.py --model models/run_3lane_3h/best_model.zip
"""
import argparse, csv, glob, os
import numpy as np
from stable_baselines3 import PPO
from traffic_env import MultiTlEnv


def run(env, policy, seeds):
    rows = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done, step = False, 0
        while not done:
            obs, r, te, tr, info = env.step(policy(obs, step))
            step += 1
            done = te or tr
        m = info["episode_metrics"]
        rows.append((m["avg_wait"], m["avg_queue"], m["arrived"]))
    return np.array(rows).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--scenario", default="scenario")
    ap.add_argument("--horizon", type=int, default=1800)
    ap.add_argument("--n-test", type=int, default=10, help="must match train.py")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--csv", default=None, help="save table to this file")
    a = ap.parse_args()

    net = os.path.join(a.scenario, "net.net.xml")
    routes = sorted(glob.glob(os.path.join(a.scenario, "routes_*.rou.xml")))[-a.n_test:]
    seeds = range(1000, 1000 + a.episodes)
    ai_env = MultiTlEnv(net, routes, horizon=a.horizon)
    default_env = MultiTlEnv(net, routes, horizon=a.horizon, control=False)
    model = PPO.load(a.model)
    if model.observation_space.shape[0] != ai_env.observation_space.shape[0]:
        ai_env.close(); default_env.close()
        raise SystemExit(
            f"Model obs size {model.observation_space.shape[0]} != env "
            f"{ai_env.observation_space.shape[0]}. Rebuild/retrain after network changes."
        )
    n_tls = len(ai_env.tls_ids)
    n_phases = [len(g) for g in ai_env.greens]

    try:
        rows = {
            "AI (PPO)": run(ai_env, lambda o, t: model.predict(o, deterministic=True)[0], seeds),
            "SUMO built-in timing": run(default_env, lambda o, t: np.zeros(n_tls, int), seeds),
            "Fixed 30 s cycle": run(ai_env, lambda o, t: np.array([(t // 6) % k for k in n_phases]), seeds),
            "Random lights": run(ai_env, lambda o, t: ai_env.action_space.sample(), seeds),
        }
    finally:
        ai_env.close(); default_env.close()

    print(f"{'controller':24s} {'avg wait':>10s} {'avg queue':>10s} {'vehicles done':>14s}")
    for name, (w, q, arr) in rows.items():
        print(f"{name:24s} {w:10.1f} {q:10.2f} {arr:14.0f}")
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        with open(a.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["controller", "avg_wait", "avg_queue", "vehicles_finished"])
            for name, v in rows.items():
                wr.writerow([name, round(v[0], 2), round(v[1], 2), round(v[2], 1)])
        print("Saved", a.csv)


if __name__ == "__main__":
    main()
