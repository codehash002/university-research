# SUMO 3D Native

Desktop GPU viewer + control hub. Launches the TraCI bridge, optional **SUMO-GUI**, optional **PPO** traffic-light control, and **approach cameras**.

## Run (no Rust required)

From the repo root (after the one-time `sumo3d` venv setup in the root [README](../README.md)):

```powershell
.\sumo3d-native\bin\sumo3d-native.exe
```

Or from this folder:

```powershell
.\bin\sumo3d-native.exe
```

### Control panel (left)

1. Pick a **route** (`ppo/scenario/routes_*.rou.xml`)
2. Pick a **controller**: Built-in SUMO TLS or a PPO `.zip` under `ppo/models/`
3. Pick **SUMO-GUI** or **Headless**
4. Click **Start**

### Map (center)

| Input | Action |
|-------|--------|
| Left-drag | Orbit |
| Right-drag | Pan |
| Scroll | Zoom |
| WASD / arrows | Move on ground |
| Q / E | Raise / lower |
| Space | Pause / play |

### Approach cameras (right)

One live feed per TLS approach, mounted on top of each signal gantry arm looking into incoming traffic. Toggle each feed or use **All on** / **All off**. Collapse panels with ◀ / ▶.

## Notes

- **Vehicles**: **3D models (GLB)** (default) or **Boxes**. Both keep TraCI blinker/brake lights.
- Prefer the release `.exe` in `bin/` for normal use.
- Developers rebuilding from source: Rust + MSVC, then `cargo build --release` and copy `target\release\sumo3d-native.exe` → `bin\sumo3d-native.exe`.
