"""SUMO TraCI → WebSocket bridge for the native Bevy viewer (no WebGL).

    python server.py --scenario ../ppo/scenario --route 50 --gui
    python server.py --scenario ../ppo/scenario --route 50 --headless --model ../ppo/models/run/best_model.zip
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import traci
from aiohttp import WSMsgType, web

from net_export import export_network

ROOT = Path(__file__).resolve().parent
PPO_DIR = ROOT.parent / "ppo"

# TraCI vehicle signal bits (sumolib / traci constants)
SIG_BLINKER_RIGHT = 1
SIG_BLINKER_LEFT = 2
SIG_BRAKELIGHT = 8


class PpoController:
    """Applies a trained PPO policy to traffic lights (logic mirrored from traffic_env)."""

    def __init__(self, model_path: str, net_file: str, conn, delta_t: int = 5, yellow: int = 3, min_green: int = 10):
        sys.path.insert(0, str(PPO_DIR))
        try:
            from stable_baselines3 import PPO  # noqa: WPS433
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "stable_baselines3 is required for PPO control. "
                "Install with:  .\\venv\\Scripts\\pip install stable-baselines3 gymnasium numpy"
            ) from e

        phases_file = Path(net_file).parent / "phases.json"
        with open(phases_file) as f:
            raw = json.load(f)
        self.tls_ids = list(raw.keys())
        self.greens = [raw[t] for t in self.tls_ids]
        self.delta_t = delta_t
        self.yellow = yellow
        self.min_green = min_green
        self.conn = conn
        self.model = PPO.load(model_path)
        self.cur = [0] * len(self.tls_ids)
        self.elapsed = [0] * len(self.tls_ids)
        self._t_in_phase = 0
        self._pending_yellow: list[int] | None = None
        self._yellow_left = 0
        self._pending_action: list[int] | None = None

        self.lanes, self.caps = [], []
        for t, g in zip(self.tls_ids, self.greens):
            lanes = sorted(set(conn.trafficlight.getControlledLanes(t)))
            self.lanes.append(lanes)
            self.caps.append(
                __import__("numpy").array(
                    [max(1.0, min(conn.lane.getLength(l) / 7.5, 12.0)) for l in lanes]
                )
            )
        import numpy as np

        self.np = np
        self.total_cap = float(sum(cap.sum() for cap in self.caps))
        for i, t in enumerate(self.tls_ids):
            conn.trafficlight.setRedYellowGreenState(t, self.greens[i][0])

        n_obs = sum(len(g) + 1 + 2 * len(l) for g, l in zip(self.greens, self.lanes)) + 1 + 2
        if self.model.observation_space.shape[0] != n_obs:
            raise RuntimeError(
                f"Model expects obs {self.model.observation_space.shape[0]}, network gives {n_obs}"
            )
        self.horizon = 1800
        self.t = 0  # whole simulation seconds processed by the policy

    def sync_to_time(self, sim_time: float):
        """Advance PPO by whole seconds even when SUMO uses sub-second steps."""
        target = int(sim_time)
        while self.t < target:
            self.on_sim_second()

    @staticmethod
    def _yellow_state(a, b):
        return "".join("y" if x in "Gg" and y == "r" else x for x, y in zip(a, b))

    def _observe(self):
        np = self.np
        c = self.conn
        queues, counts = [], []
        for lanes, caps in zip(self.lanes, self.caps):
            queues.append(np.array([c.lane.getLastStepHaltingNumber(l) for l in lanes]))
            counts.append(np.array([c.lane.getLastStepVehicleNumber(l) for l in lanes]))
        ped_wait = ped_total = 0
        try:
            for pid in c.person.getIDList():
                ped_total += 1
                if c.person.getWaitingTime(pid) > 0.5:
                    ped_wait += 1
        except Exception:
            pass
        parts = []
        for i in range(len(self.tls_ids)):
            phase = np.zeros(len(self.greens[i]), np.float32)
            phase[self.cur[i]] = 1.0
            parts += [
                phase,
                [min(self.elapsed[i] / 60.0, 1.0)],
                np.clip(queues[i] / self.caps[i], 0, 1),
                np.clip(counts[i] / self.caps[i], 0, 1),
            ]
        parts.append([min(self.t / self.horizon, 1.0)])
        parts.append([min(ped_wait / 8.0, 1.0), min(ped_total / 8.0, 1.0)])
        return np.concatenate(parts).astype(np.float32)

    def on_sim_second(self):
        """Call once per whole simulation second (not per TraCI step)."""
        self.t += 1
        self.elapsed = [e + 1 for e in self.elapsed]
        self._t_in_phase += 1

        if self._pending_yellow is not None:
            self._yellow_left -= 1
            if self._yellow_left <= 0:
                for i in self._pending_yellow:
                    self.cur[i] = self._pending_action[i]
                    self.elapsed[i] = 0
                    self.conn.trafficlight.setRedYellowGreenState(
                        self.tls_ids[i], self.greens[i][self.cur[i]]
                    )
                self._pending_yellow = None
                self._pending_action = None
                self._t_in_phase = 0
            return

        if self._t_in_phase < self.delta_t:
            return

        obs = self._observe()
        action, _ = self.model.predict(obs, deterministic=True)
        action = [int(a) for a in action]
        change = [
            i
            for i, a in enumerate(action)
            if a != self.cur[i] and self.elapsed[i] >= self.min_green
        ]
        self._t_in_phase = 0
        if not change:
            return
        for i in change:
            y = self._yellow_state(self.greens[i][self.cur[i]], self.greens[i][action[i]])
            self.conn.trafficlight.setRedYellowGreenState(self.tls_ids[i], y)
        self._pending_yellow = change
        self._pending_action = action
        self._yellow_left = self.yellow


class SumoBridge:
    """Owns one SUMO TraCI connection and broadcasts step state."""

    def __init__(
        self,
        net_file: str,
        route_file: str,
        step_length: float = 0.1,
        gui: bool = False,
        model_path: str | None = None,
    ):
        self.net_file = net_file
        self.route_file = route_file
        self.step_length = step_length
        self.gui = gui
        self.model_path = model_path
        self.network = export_network(net_file)
        self.clients: set[web.WebSocketResponse] = set()
        self.running = False
        self.paused = False
        self.speed = 1.0
        self._label = f"sumo3d_{os.getpid()}"
        self.conn = None
        self.sim_time = 0.0
        self.vehicle_count = 0
        self.done = False
        self.ppo: PpoController | None = None

    def start_sumo(self):
        binary = "sumo-gui" if self.gui else "sumo"
        cmd = [
            binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--step-length", str(self.step_length),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--waiting-time-memory", "1000",
            "--start", "true",
            "--quit-on-end", "true",
            # Don't abort the whole run if one flow has a bad edge after net rebuild
            "--ignore-route-errors", "true",
        ]
        if self.gui:
            gui_cfg = Path(self.net_file).parent / "gui_settings.xml"
            if gui_cfg.exists():
                cmd += ["--gui-settings-file", str(gui_cfg)]
            # Shorter delay so GUI keeps up with sub-second steps
            cmd += ["--delay", str(max(1, int(self.step_length * 80)))]
        try:
            traci.start(cmd, label=self._label)
            self.conn = traci.getConnection(self._label)
        except Exception as e:
            print(f"[sumo] failed to start: {e}", file=sys.stderr)
            self.conn = None
            self.running = False
            self.done = True
            self.ppo = None
            raise
        self.running = True
        self.done = False
        self.sim_time = 0.0
        self.ppo = None
        if self.model_path:
            self.ppo = PpoController(self.model_path, self.net_file, self.conn)

    def stop_sumo(self):
        self.running = False
        self.ppo = None
        if self.conn is None:
            return
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = None
        try:
            if self._label in traci.getConnectionNames():
                traci.getConnection(self._label).close()
        except Exception:
            pass

    def restart(self):
        self.stop_sumo()
        self.start_sumo()

    def reconfigure(self, route_file: str | None = None, gui: bool | None = None, model_path=...):
        if route_file is not None:
            self.route_file = route_file
        if gui is not None:
            self.gui = gui
        if model_path is not ...:
            self.model_path = model_path or None
        self.restart()

    def _mark_dead(self, reason: str = ""):
        if reason:
            print(f"[sumo] connection lost: {reason}", file=sys.stderr)
        self.running = False
        self.done = True
        self.conn = None
        self.ppo = None

    def snapshot(self) -> dict | None:
        c = self.conn
        if c is None:
            return None
        try:
            vehicles = []
            for vid in c.vehicle.getIDList():
                x, y = c.vehicle.getPosition(vid)
                try:
                    signals = int(c.vehicle.getSignals(vid))
                except Exception:
                    signals = 0
                vehicles.append({
                    "id": vid,
                    "x": float(x),
                    "y": float(y),
                    "angle": float(c.vehicle.getAngle(vid)),
                    "speed": float(c.vehicle.getSpeed(vid)),
                    "length": float(c.vehicle.getLength(vid)),
                    "width": float(c.vehicle.getWidth(vid)),
                    "type": c.vehicle.getTypeID(vid),
                    "signals": signals,
                })

            persons = []
            try:
                for pid in c.person.getIDList():
                    x, y = c.person.getPosition(pid)
                    persons.append({
                        "id": pid,
                        "x": float(x),
                        "y": float(y),
                        "angle": float(c.person.getAngle(pid)),
                        "speed": float(c.person.getSpeed(pid)),
                    })
            except Exception:
                pass

            lights = []
            for tid in c.trafficlight.getIDList():
                lights.append({
                    "id": tid,
                    "state": c.trafficlight.getRedYellowGreenState(tid),
                })

            self.sim_time = float(c.simulation.getTime())
            self.vehicle_count = len(vehicles)
            return {
                "type": "frame",
                "t": self.sim_time,
                "vehicles": vehicles,
                "persons": persons,
                "lights": lights,
                "arrived": int(c.simulation.getArrivedNumber()),
                "expected": int(c.simulation.getMinExpectedNumber()),
                "controller": "ppo" if self.ppo else "builtin",
            }
        except Exception as e:
            self._mark_dead(str(e))
            return None

    def step_once(self) -> dict | None:
        if self.conn is None or self.done:
            return None
        try:
            self.conn.simulationStep()
        except Exception as e:
            self._mark_dead(str(e))
            return None
        if self.ppo is not None:
            try:
                self.ppo.sync_to_time(float(self.conn.simulation.getTime()))
            except Exception as e:
                print(f"[ppo] step error: {e}", file=sys.stderr)
        frame = self.snapshot()
        if frame is None:
            return None
        if frame["expected"] == 0:
            self.done = True
            frame["done"] = True
        return frame

    async def broadcast(self, payload: dict):
        if not self.clients:
            return
        raw = json.dumps(payload, separators=(",", ":"))
        dead = []
        for ws in self.clients:
            try:
                await ws.send_str(raw)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def sim_loop(self):
        while True:
            if not self.running or self.conn is None:
                await asyncio.sleep(0.1)
                continue
            if self.paused or self.done:
                await asyncio.sleep(0.05)
                continue
            t0 = time.perf_counter()
            frame = self.step_once()
            if frame is not None:
                await self.broadcast(frame)
            target = max(0.001, self.step_length / max(0.05, self.speed))
            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, target - elapsed))


bridge: SumoBridge | None = None


async def ws_handler(request: web.Request):
    global bridge
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    bridge.clients.add(ws)

    await ws.send_str(json.dumps({"type": "network", "network": bridge.network}))
    await ws.send_str(json.dumps(status_payload()))
    if bridge.conn is not None and not bridge.done:
        snap = bridge.snapshot()
        if snap is not None:
            await ws.send_str(json.dumps(snap, separators=(",", ":")))

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            await handle_client_cmd(ws, data)
        elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
            break

    bridge.clients.discard(ws)
    return ws


def status_payload() -> dict:
    return {
        "type": "status",
        "paused": bridge.paused,
        "speed": bridge.speed,
        "route": os.path.basename(bridge.route_file),
        "t": bridge.sim_time,
        "done": bridge.done,
        "gui": bridge.gui,
        "controller": "ppo" if bridge.ppo else "builtin",
        "model": bridge.model_path or "",
    }


async def handle_client_cmd(_ws: web.WebSocketResponse, data: dict):
    cmd = data.get("cmd")
    if cmd == "pause":
        bridge.paused = True
    elif cmd == "play":
        bridge.paused = False
        if bridge.done:
            bridge.restart()
    elif cmd == "toggle":
        bridge.paused = not bridge.paused
        if not bridge.paused and bridge.done:
            bridge.restart()
    elif cmd == "speed":
        bridge.speed = float(max(0.05, min(20.0, data.get("value", 1.0))))
    elif cmd == "restart":
        bridge.restart()
        bridge.paused = False
        await bridge.broadcast({"type": "network", "network": bridge.network})
    elif cmd == "configure":
        # Native launcher: change route / gui / model and restart
        route = data.get("route")
        gui = data.get("gui")
        model = data.get("model")
        scenario = Path(bridge.net_file).parent
        route_file = None
        if route is not None:
            route_file = str(scenario / f"routes_{int(route)}.rou.xml")
            if not Path(route_file).exists():
                await _ws.send_str(json.dumps({"type": "error", "message": f"missing {route_file}"}))
                return
        model_path = ...
        if "model" in data:
            model_path = model if model else None
            if model_path and not Path(model_path).exists():
                await _ws.send_str(json.dumps({"type": "error", "message": f"missing model {model_path}"}))
                return
        bridge.reconfigure(
            route_file=route_file,
            gui=bool(gui) if gui is not None else None,
            model_path=model_path,
        )
        bridge.paused = False
        bridge.network = export_network(bridge.net_file)
        await bridge.broadcast({"type": "network", "network": bridge.network})
    elif cmd == "step":
        was = bridge.paused
        bridge.paused = True
        frame = bridge.step_once()
        if frame:
            await bridge.broadcast(frame)
        bridge.paused = was
    await bridge.broadcast(status_payload())


async def on_startup(app: web.Application):
    bridge.start_sumo()
    app["sim_task"] = asyncio.create_task(bridge.sim_loop())


async def on_cleanup(app: web.Application):
    task = app.get("sim_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    bridge.stop_sumo()


def main():
    global bridge
    ap = argparse.ArgumentParser(description="SUMO TraCI bridge for native 3D viewer")
    ap.add_argument("--scenario", default=os.path.join("..", "ppo", "scenario"))
    ap.add_argument("--route", type=int, default=50)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--step-length",
        type=float,
        default=0.1,
        help="SUMO step length in seconds (0.1 = smooth; 1.0 = jumpy)",
    )
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--gui", action="store_true", help="open SUMO-GUI window")
    ap.add_argument("--headless", action="store_true", help="force headless sumo (default if no --gui)")
    ap.add_argument("--model", default="", help="PPO .zip path (empty = built-in TLS)")
    a = ap.parse_args()

    scenario = Path(a.scenario).resolve()
    net = scenario / "net.net.xml"
    route = scenario / f"routes_{a.route}.rou.xml"
    if not net.exists() or not route.exists():
        sys.exit(
            f"Missing scenario files.\n  net:   {net}\n  route: {route}\n"
            "Build first:  cd ../ppo && python make_scenario.py --lanes 3"
        )

    gui = bool(a.gui) and not a.headless
    model_path = a.model.strip() or None
    if model_path:
        model_path = str(Path(model_path).resolve())
        if not Path(model_path).exists():
            sys.exit(f"Model not found: {model_path}")

    bridge = SumoBridge(
        str(net), str(route), step_length=a.step_length, gui=gui, model_path=model_path
    )
    bridge.speed = a.speed

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/", lambda _r: web.Response(text="SUMO TraCI bridge (native client). Connect via /ws\n"))
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    print("=" * 56)
    print("  SUMO TraCI bridge  ->  native 3D")
    print(f"  Network : {net.name}")
    print(f"  Traffic : {route.name}")
    print(f"  SUMO    : {'sumo-gui' if gui else 'sumo (headless)'}")
    print(f"  Control : {'PPO ' + model_path if model_path else 'built-in TLS'}")
    print(f"  WS      : ws://{a.host}:{a.port}/ws")
    print("=" * 56)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
