# SUMO TraCI bridge (native 3D)

Headless WebSocket bridge used by **`sumo3d-native`**. No browser / WebGL UI.

```
SUMO (gui|headless) ──TraCI──► server.py ──WebSocket──► sumo3d-native.exe
```

## Setup

```powershell
cd sumo3d
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
# For PPO control from the native launcher, also install ppo deps:
pip install -r ../ppo/requirements.txt
```

## Run (usually started by the native app)

```powershell
.\venv\Scripts\python.exe server.py --scenario ../ppo/scenario --route 50 --gui
.\venv\Scripts\python.exe server.py --scenario ../ppo/scenario --route 46 --headless --model ../ppo/models/<run>/best_model.zip
```

| Flag | Meaning |
|------|---------|
| `--gui` / `--headless` | Open SUMO-GUI or run headless |
| `--route N` | `routes_N.rou.xml` |
| `--model PATH` | PPO zip (omit = built-in TLS) |
| `--speed` | Sim speed multiplier |

Connect: `ws://127.0.0.1:8765/ws`

## Layout

```
sumo3d/
├── server.py       # TraCI + optional PPO + WS
├── net_export.py   # net.xml → JSON (per-lane TLS heads)
└── requirements.txt
```
