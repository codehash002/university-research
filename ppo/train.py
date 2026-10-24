"""Train PPO traffic-light control.

    python train.py --hours 3
    python train.py --timesteps 500000
"""
import argparse, glob, os, time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from traffic_env import MultiTlEnv


class StopAfter(BaseCallback):
    def __init__(self, seconds):
        super().__init__()
        self.seconds, self.deadline = seconds, None

    def _on_training_start(self):
        self.deadline = time.time() + self.seconds

    def _on_step(self):
        return time.time() < self.deadline


def make_env(net, routes, horizon, rank, gui=False, **kw):
    def _init():
        return MultiTlEnv(net, routes, horizon=horizon, gui=gui,
                          label=f"env{rank}_{os.getpid()}", **kw)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenario")
    ap.add_argument("--hours", type=float, default=None, help="train for this many hours of real time")
    ap.add_argument("--timesteps", type=int, default=500_000, help="used when --hours is not given")
    ap.add_argument("--n-envs", type=int, default=max(1, min((os.cpu_count() or 2) - 1, 8)),
                    help="parallel SUMO copies")
    ap.add_argument("--horizon", type=int, default=1800, help="simulated seconds per episode")
    ap.add_argument("--n-test", type=int, default=10, help="traffic files kept aside for testing only")
    ap.add_argument("--run-name", default="ppo5_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--queue-weight", type=float, default=0.5)
    ap.add_argument("--switch-weight", type=float, default=0.05)
    ap.add_argument("--ped-weight", type=float, default=0.2,
                    help="penalty weight for waiting pedestrians")
    ap.add_argument("--gui", action="store_true",
                    help="open SUMO-GUI (forces --n-envs 1)")
    a = ap.parse_args()
    kw = dict(queue_weight=a.queue_weight, switch_weight=a.switch_weight,
              ped_weight=a.ped_weight)

    if a.gui:
        a.n_envs = 1
        print("GUI mode: using 1 simulation. Prefer demo.py after training.")

    net = os.path.join(a.scenario, "net.net.xml")
    routes = sorted(glob.glob(os.path.join(a.scenario, "routes_*.rou.xml")))
    if len(routes) <= a.n_test:
        raise SystemExit("Not enough traffic files. Run: python make_scenario.py")
    train_routes, test_routes = routes[:-a.n_test], routes[-a.n_test:]
    print(f"{len(train_routes)} traffic files for training, {len(test_routes)} kept for testing, "
          f"{a.n_envs} parallel simulations")

    Vec = DummyVecEnv if (a.gui or a.n_envs == 1) else SubprocVecEnv
    train_env = VecMonitor(Vec([make_env(net, train_routes, a.horizon, i, gui=a.gui, **kw)
                                for i in range(a.n_envs)]))
    eval_env = VecMonitor(DummyVecEnv([make_env(net, test_routes, a.horizon, 99, gui=False, **kw)]))

    model = PPO("MlpPolicy", train_env, seed=a.seed, verbose=1,
                learning_rate=3e-4, n_steps=512, batch_size=256, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[256, 256]), tensorboard_log="runs")

    out = os.path.join("models", a.run_name)
    os.makedirs(out, exist_ok=True)
    callbacks = [
        EvalCallback(eval_env, best_model_save_path=out, log_path=out,
                     eval_freq=max(25_000 // a.n_envs, 1), n_eval_episodes=3, deterministic=True),
        CheckpointCallback(save_freq=max(100_000 // a.n_envs, 1), save_path=out, name_prefix="ckpt"),
    ]
    total = a.timesteps
    if a.hours:
        callbacks.append(StopAfter(a.hours * 3600))
        total = 10 ** 10
    try:
        model.learn(total_timesteps=total, callback=callbacks, tb_log_name=a.run_name)
        model.save(os.path.join(out, "final"))
        print("Finished. Models saved in", out)
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
