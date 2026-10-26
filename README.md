# Traffic AI Projects

Shared monorepo for traffic-control experiments. Each project lives in its own folder.

| Folder | Status | Description |
|--------|--------|-------------|
| [`ppo/`](ppo/) | Ready | PPO agent that controls traffic lights in SUMO |
| [`sumo3d/`](sumo3d/) | Ready | TraCI WebSocket bridge (GUI/headless + optional PPO) |
| [`sumo3d-native/`](sumo3d-native/) | Ready | Native GPU 3D control hub + approach cameras (prebuilt `.exe`) |
| `decision-model/` | Coming soon | Decision-model experiments |

---

## Prerequisites

- **Windows** (3D viewer ships as a Windows `.exe`)
- **Python 3.10+**
- **[SUMO](https://eclipse.dev/sumo/)** installed and on your `PATH`
  - Set `SUMO_HOME` to the SUMO install directory (e.g. `C:\Program Files (x86)\Eclipse\Sumo`)
  - Check: `sumo --version`
- A GPU/driver that can run Vulkan (Intel UHD / NVIDIA / AMD are fine)

You do **not** need Rust. The 3D app is the prebuilt binary in `sumo3d-native/bin/`.

---

## Clone

```powershell
git clone <REPO_URL>
cd <repo-name>
```

---

## One-time setup (venv + bridge deps)

The native viewer starts the Python TraCI bridge for you. Install bridge deps once:

```powershell
cd sumo3d

# Create venv
py -3 -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# Install bridge + optional PPO controller deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r ..\ppo\requirements.txt

cd ..
```

> `venv/` is gitignored — each teammate creates their own.

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Run the 3D hub (main workflow)

With SUMO installed and the `sumo3d` venv created as above:

```powershell
.\sumo3d-native\bin\sumo3d-native.exe
```

In the app:

1. **Left panel — SUMO 3D Control**
   - Pick a **route** (`ppo/scenario/routes_*.rou.xml`)
   - Pick a **controller**: Built-in SUMO TLS, or a PPO `.zip` under `ppo/models/`
   - Pick **SUMO-GUI** or **Headless**
   - Pick **3D models (GLB)** or **Boxes** for vehicles
   - Click **Start**
2. **Center** — free-look 3D map (LMB orbit · RMB pan · scroll zoom · WASD · Q/E height · Space pause)
3. **Right panel — Approach cameras** — live gantry-top feeds into incoming traffic; toggle per camera or All on/off

Collapse either side panel with the ◀ / ▶ buttons when you want more map space.

The hub launches `sumo3d/server.py` with the venv Python automatically. With **SUMO-GUI** selected, SUMO opens in a second window on the same simulation.

More detail: [`sumo3d-native/README.md`](sumo3d-native/README.md).

---

## PPO project (train / evaluate)

```powershell
cd ppo

py -3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

All commands below assume you are inside `ppo/` with the venv activated.

### Build scenario (if needed)

```powershell
python make_scenario.py --lanes 3 --n-routes 60
```

Demand only (keep network):

```powershell
python make_scenario.py --demand-only --n-routes 60
```

### Train

```powershell
python train.py --hours 3 --run-name run_3lane_3h
# or
python train.py --timesteps 500000 --run-name my_run
```

Models land in `models/<run-name>/`. TensorBoard logs in `runs/`.

### Evaluate / demo

```powershell
python evaluate.py --model models/run_3lane_3h/best_model.zip --csv results/run_3lane_3h.csv
python demo.py --model models/run_3lane_3h/best_model.zip --route 50
python preview.py --route 50
python compare_gui.py --model models/run_3lane_3h/best_model.zip --route 50
```

Full pipeline:

```powershell
python run_all.py --hours 3 --run-name run_3lane_3h --demo
```

### PPO layout

```
ppo/
├── make_scenario.py
├── train.py
├── evaluate.py
├── demo.py / preview.py / compare_gui.py
├── run_all.py
├── traffic_env.py
├── requirements.txt
├── scenario/          # Network + traffic demand
├── models/            # Trained checkpoints
├── results/           # Evaluation CSVs (gitignored)
└── runs/              # TensorBoard logs (gitignored)
```

---

## Tips

- Always **activate the matching venv** before running Python scripts by hand.
- Run PPO scripts from inside `ppo/` so `scenario/` and `models/` paths resolve.
- For a quick setup check, use a short train (`--hours 0.1`) before a long run.
- Training curves: `tensorboard --logdir runs` (from `ppo/` with venv on).
- If the 3D window fails to start the bridge, confirm `sumo3d\venv` exists and `sumo --version` works.
