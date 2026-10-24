"""Gymnasium env: TraCI bridge between SUMO and the PPO agent."""
import json, os, random
import numpy as np
import gymnasium as gym
import traci


class MultiTlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, net_file, route_files, phases_file=None, delta_t=5, yellow=3,
                 min_green=10, horizon=1800, wait_norm=200.0,
                 queue_weight=0.5, switch_weight=0.05, ped_weight=0.2,
                 ped_norm=8.0,
                 control=True, gui=False, gui_delay=0, label=None):
        super().__init__()
        self.net_file, self.route_files = net_file, list(route_files)
        phases_file = phases_file or os.path.join(os.path.dirname(net_file), "phases.json")
        with open(phases_file) as f:
            self.greens = json.load(f)
        self.tls_ids = list(self.greens.keys())
        self.greens = [self.greens[t] for t in self.tls_ids]
        self.delta_t, self.yellow, self.min_green, self.horizon = delta_t, yellow, min_green, horizon
        self.wait_norm = wait_norm
        self.queue_weight, self.switch_weight = queue_weight, switch_weight
        self.ped_weight, self.ped_norm = ped_weight, ped_norm
        self.control = control
        self.gui = gui
        self.gui_delay = int(gui_delay)
        self.label = label or f"sim{os.getpid()}_{id(self)}"
        self.conn = None

        self._start(self.route_files[0], seed=0)
        self._read_structure()
        n_obs = sum(len(g) + 1 + 2 * len(l) for g, l in zip(self.greens, self.lanes)) + 1 + 2
        self.observation_space = gym.spaces.Box(0.0, 1.0, (n_obs,), np.float32)
        self.action_space = gym.spaces.MultiDiscrete([len(g) for g in self.greens])
        self._close()

    def _start(self, route, seed):
        binary = "sumo-gui" if self.gui else "sumo"
        cmd = [binary, "-n", self.net_file, "-r", route,
               "--seed", str(seed), "--no-step-log", "true", "--no-warnings", "true",
               "--time-to-teleport", "-1", "--waiting-time-memory", "1000"]
        if self.gui:
            cmd += ["--start", "true", "--quit-on-end", "true"]
            if self.gui_delay > 0:
                cmd += ["--delay", str(self.gui_delay)]
            gui_cfg = os.path.join(os.path.dirname(self.net_file), "gui_settings.xml")
            if os.path.exists(gui_cfg):
                cmd += ["--gui-settings-file", gui_cfg]
        traci.start(cmd, label=self.label)
        self.conn = traci.getConnection(self.label)

    def _close(self):
        if self.conn is None:
            return
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = None
        try:
            if hasattr(traci, "getConnectionNames") and self.label in traci.getConnectionNames():
                try:
                    traci.getConnection(self.label).close()
                except Exception:
                    pass
        except Exception:
            pass

    def _read_structure(self):
        c = self.conn
        self.lanes, self.caps = [], []
        for t, g in zip(self.tls_ids, self.greens):
            n_links = len(c.trafficlight.getRedYellowGreenState(t))
            assert all(len(s) == n_links for s in g), f"phases.json does not match the network for {t}"
            lanes = sorted(set(c.trafficlight.getControlledLanes(t)))
            self.lanes.append(lanes)
            self.caps.append(np.array([max(1.0, min(c.lane.getLength(l) / 7.5, 12.0)) for l in lanes]))
        self.total_cap = float(sum(cap.sum() for cap in self.caps))

    @staticmethod
    def _yellow_state(a, b):
        return "".join("y" if x in "Gg" and y == "r" else x for x, y in zip(a, b))

    def _sim(self, seconds):
        for _ in range(int(seconds)):
            self.conn.simulationStep()
            self.arrived += self.conn.simulation.getArrivedNumber()
            if self.conn.simulation.getMinExpectedNumber() == 0:
                return False
        return True

    def _lane_numbers(self):
        queues, counts, wait = [], [], 0.0
        for lanes in self.lanes:
            queues.append(np.array([self.conn.lane.getLastStepHaltingNumber(l) for l in lanes]))
            counts.append(np.array([self.conn.lane.getLastStepVehicleNumber(l) for l in lanes]))
            wait += sum(self.conn.lane.getWaitingTime(l) for l in lanes)
        return queues, counts, wait

    def _ped_numbers(self):
        waiting, total = 0, 0
        try:
            for pid in self.conn.person.getIDList():
                total += 1
                if self.conn.person.getWaitingTime(pid) > 0.5:
                    waiting += 1
        except Exception:
            pass
        return waiting, total

    def _observe(self):
        queues, counts, _ = self._lane_numbers()
        ped_wait, ped_total = self._ped_numbers()
        parts = []
        for i in range(len(self.tls_ids)):
            phase = np.zeros(len(self.greens[i]), np.float32)
            phase[self.cur[i]] = 1.0
            parts += [phase, [min(self.elapsed[i] / 60.0, 1.0)],
                      np.clip(queues[i] / self.caps[i], 0, 1),
                      np.clip(counts[i] / self.caps[i], 0, 1)]
        parts.append([min(self.t / self.horizon, 1.0)])
        parts.append([min(ped_wait / self.ped_norm, 1.0),
                      min(ped_total / self.ped_norm, 1.0)])
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close()
        if options and options.get("route"):
            route = options["route"]
        elif seed is not None:
            route = random.Random(seed).choice(self.route_files)
        else:
            route = random.choice(self.route_files)
        self._start(route, seed=int(self.np_random.integers(1_000_000)))
        n = len(self.tls_ids)
        self.cur, self.elapsed = [0] * n, [0] * n
        self.t, self.arrived, self.steps = 0, 0, 0
        self.queue_sum, self.wait_sum, self.ped_wait_sum = 0.0, 0.0, 0.0
        if self.control:
            for i, t in enumerate(self.tls_ids):
                self.conn.trafficlight.setRedYellowGreenState(t, self.greens[i][0])
        self._sim(1)
        self.prev_wait = self._lane_numbers()[2]
        return self._observe(), {}

    def step(self, action):
        alive, switched = True, 0
        if self.control:
            change = [i for i, a in enumerate(action)
                      if int(a) != self.cur[i] and self.elapsed[i] >= self.min_green]
            if change:
                for i in change:
                    y = self._yellow_state(self.greens[i][self.cur[i]], self.greens[i][int(action[i])])
                    self.conn.trafficlight.setRedYellowGreenState(self.tls_ids[i], y)
                alive = self._sim(self.yellow)
                self.t += self.yellow
                for i in change:
                    self.cur[i] = int(action[i])
                    self.elapsed[i] = 0
                    self.conn.trafficlight.setRedYellowGreenState(self.tls_ids[i], self.greens[i][self.cur[i]])
                switched = len(change)
        if alive:
            alive = self._sim(self.delta_t)
        self.t += self.delta_t
        self.elapsed = [e + self.delta_t for e in self.elapsed]

        queues, _, wait = self._lane_numbers()
        ped_wait, ped_total = self._ped_numbers()
        queue_total = float(sum(q.sum() for q in queues))
        progress = float(np.clip((self.prev_wait - wait) / self.wait_norm, -1.0, 1.0))
        queue_penalty = self.queue_weight * min(queue_total / self.total_cap, 1.0)
        switch_penalty = self.switch_weight * switched
        ped_penalty = self.ped_weight * min(ped_wait / self.ped_norm, 1.0)
        reward = progress - queue_penalty - switch_penalty - ped_penalty
        self.prev_wait = wait
        self.queue_sum += queue_total
        self.wait_sum += wait
        self.ped_wait_sum = getattr(self, "ped_wait_sum", 0.0) + ped_wait
        self.steps += 1

        terminated, truncated = not alive, self.t >= self.horizon
        info = {"waiting_time": wait, "queue": queue_total,
                "ped_waiting": ped_wait, "ped_total": ped_total}
        if terminated or truncated:
            k = max(self.steps, 1)
            info["episode_metrics"] = {"avg_queue": self.queue_sum / k, "avg_wait": self.wait_sum / k,
                                       "arrived": self.arrived,
                                       "avg_ped_waiting": self.ped_wait_sum / k}
        return self._observe(), float(reward), terminated, truncated, info

    def close(self):
        self._close()
