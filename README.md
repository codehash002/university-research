# Traffic AI Projects

Shared repo for traffic-control experiments. Each project lives in its own folder.

| Folder | Status | Description |
|--------|--------|-------------|
| [`ppo/`](ppo/) | Ready | PPO agent that controls traffic lights in SUMO |
| `decision-model/` | Coming soon | Decision-model experiments |

---

## Prerequisites

- **Python 3.10+**
- **[SUMO](https://eclipse.dev/sumo/)** (traffic simulator) installed and on your `PATH`
  - After install, set `SUMO_HOME` to the SUMO install directory (e.g. `C:\Program Files (x86)\Eclipse\Sumo`)
  - Check: `sumo --version`

---

## Clone and set up

```bash
git clone <REPO_URL>
cd <repo-name>
```

### PPO project

```bash
cd ppo

# Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> `venv/` and `__pycache__/` are gitignored — each teammate creates their own venv.

---

## PPO quick start

All commands below assume you are inside `ppo/` with the venv activated.

### 1. Build the scenario (network + traffic)

Skip if `scenario/net.net.xml` and route files already exist.

```bash
python make_scenario.py --lanes 3 --n-routes 60
```

Regenerate only traffic (keep the network):

```bash
python make_scenario.py --demand-only --n-routes 60
```

### 2. Train

By wall-clock hours:

```bash
python train.py --hours 3 --run-name run_3lane_3h
```

Or by timesteps:

```bash
python train.py --timesteps 500000 --run-name my_run
```

Models are saved under `models/<run-name>/` (`best_model.zip`, `final.zip`, checkpoints). TensorBoard logs go to `runs/`.

### 3. Evaluate (PPO vs baselines)

```bash
python evaluate.py --model models/run_3lane_3h/best_model.zip --csv results/run_3lane_3h.csv
```

Prints average wait, queue, and finished vehicles for:

- AI (PPO)
- SUMO built-in timing
- Fixed 30 s cycle
- Random lights

### 4. Demo in SUMO-GUI

Watch the trained agent:

```bash
python demo.py --model models/run_3lane_3h/best_model.zip --route 50
```

SUMO built-in controller only (no AI):

```bash
python preview.py --route 50
```

Side-by-side (PPO + built-in):

```bash
python compare_gui.py --model models/run_3lane_3h/best_model.zip --route 50
```

### 5. Full pipeline (build → train → evaluate → optional demo)

```bash
python run_all.py --hours 3 --run-name run_3lane_3h --demo
```

---

## PPO layout

```
ppo/
├── make_scenario.py   # Build network + route files
├── train.py           # Train PPO
├── evaluate.py        # Compare vs baselines
├── demo.py            # Run PPO in SUMO-GUI
├── preview.py         # SUMO built-in GUI (no AI)
├── compare_gui.py     # PPO vs built-in side by side
├── run_all.py         # End-to-end pipeline
├── traffic_env.py     # Gymnasium / SUMO environment
├── requirements.txt
├── scenario/          # Network + traffic demand
├── models/            # Trained checkpoints
├── results/           # Evaluation CSVs
└── runs/              # TensorBoard logs
```

---

## Tips

- Always **activate the venv** before running scripts.
- Run scripts from inside `ppo/` so relative paths (`scenario/`, `models/`) resolve correctly.
- Training can take a long time; start with a short run (`--hours 0.1` or fewer `--timesteps`) to verify setup.
- View training curves: `tensorboard --logdir runs`
