//! Native SUMO 3D control hub (Bevy + egui + TraCI bridge).

mod approach_cams;
mod geom;
mod protocol;
mod ws;

use approach_cams::{
    approach_cam_ui, spawn_approach_cameras, sync_approach_cam_active, ApproachCams, UiPanels,
};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use bevy::render::primitives::Aabb;
use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiUserTextures};
use geom::{
    dashed_line_mesh, junction_mesh, offset_polyline, ribbon_mesh, sample_at, solid_line_mesh,
    to_world, total_len, trim_shape_ends, zebra_mesh,
};
use protocol::{
    Lane, LightState, Network, ServerMsg, Tls, Vehicle, SIG_BLINKER_LEFT, SIG_BLINKER_RIGHT,
    SIG_BRAKELIGHT,
};
use std::collections::{HashMap, HashSet};
use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Duration;
use ws::{spawn_bridge, BridgeChannels, ClientCmd};

fn sumo_yaw(angle_deg: f32) -> f32 {
    (90.0 - angle_deg) * PI / 180.0
}

fn resolve_type(vtype: &str) -> &'static str {
    let t = vtype.to_lowercase();
    if t.contains("bus") {
        "bus"
    } else if t.contains("truck")
        || t.contains("trailer")
        || t.contains("hgv")
        || t.contains("lorry")
        || t.contains("delivery")
    {
        "truck"
    } else if t.contains("van") {
        "van"
    } else if t.contains("motorcycle") || t.contains("moped") || t.contains("bike") {
        "motorcycle"
    } else {
        "car"
    }
}

fn vehicle_size(kind: &str) -> Vec3 {
    match kind {
        "bus" => Vec3::new(12.0, 3.2, 2.5),
        "truck" => Vec3::new(8.0, 3.4, 2.4),
        "van" => Vec3::new(5.5, 2.4, 2.0),
        "motorcycle" => Vec3::new(2.2, 1.4, 0.9),
        _ => Vec3::new(4.5, 1.4, 1.8),
    }
}

fn vehicle_color(kind: &str, id: &str) -> Color {
    let mut h: u32 = 0;
    for b in id.bytes() {
        h = h.wrapping_mul(31).wrapping_add(b as u32);
    }
    let palette = [
        Color::srgb(0.92, 0.93, 0.95),
        Color::srgb(0.35, 0.55, 0.95),
        Color::srgb(0.90, 0.35, 0.35),
        Color::srgb(0.25, 0.80, 0.55),
        Color::srgb(0.90, 0.72, 0.30),
        Color::srgb(0.60, 0.45, 0.85),
        Color::srgb(0.30, 0.80, 0.75),
        Color::srgb(0.15, 0.15, 0.15),
    ];
    match kind {
        "bus" => Color::srgb(0.95, 0.80, 0.15),
        "truck" => Color::srgb(0.55, 0.35, 0.20),
        "van" => Color::srgb(0.20, 0.65, 0.50),
        "motorcycle" => Color::srgb(0.80, 0.20, 0.20),
        _ => palette[(h as usize) % palette.len()],
    }
}

#[derive(Resource)]
struct Bridge(BridgeChannels);

#[derive(Resource)]
struct BridgeProc(Option<Child>);

#[derive(Resource, Default)]
struct NetBuilt(bool);

#[derive(Resource, Default)]
struct VehicleMap(HashMap<String, Entity>);

#[derive(Resource)]
struct SharedVehicleMeshes {
    by_kind: HashMap<&'static str, Handle<Mesh>>,
    signal: Handle<Mesh>,
}

#[derive(Resource)]
struct GlbCatalog {
    scenes: HashMap<&'static str, Handle<Scene>>,
}

#[derive(Clone, Copy)]
struct GlbFix {
    scale: f32,
    yaw_extra: f32,
    /// Scene-local translation so the mesh sits centered on the entity (XZ) and on the ground (Y).
    offset: Vec3,
}

#[derive(Resource, Default)]
struct GlbNorm(HashMap<&'static str, GlbFix>);

#[derive(Resource, Default)]
struct GlbProbeQueued(HashSet<&'static str>);

/// Target length (m) + yawNose after length→X (matches former WebGL catalog).
fn glb_catalog_spec(kind: &str) -> (&'static str, f32, f32) {
    match kind {
        "bus" => ("models/bus.glb#Scene0", 12.0, PI),
        "truck" => ("models/truck.glb#Scene0", 8.0, 0.0),
        "van" => ("models/van.glb#Scene0", 5.5, 0.0),
        "motorcycle" => ("models/motorcycle.glb#Scene0", 2.2, 0.0),
        _ => ("models/car.glb#Scene0", 4.5, PI),
    }
}

#[derive(Component)]
struct VehicleTag {
    id: String,
}

/// Smooth visual motion toward the latest TraCI pose (avoids 1 Hz teleport look).
#[derive(Component)]
struct VehSmooth {
    target_pos: Vec3,
    target_yaw: f32,
}

#[derive(Component)]
struct VehSignals {
    bits: u32,
    mat_bl: Handle<StandardMaterial>,
    mat_br: Handle<StandardMaterial>,
    mat_brake: Handle<StandardMaterial>,
}

#[derive(Component)]
struct NeedsGlbNorm {
    kind: &'static str,
}

#[derive(Component)]
struct OrbitCam {
    target: Vec3,
    radius: f32,
    yaw: f32,
    pitch: f32,
}

#[derive(Component)]
struct TlsHeadTag {
    tls_id: String,
    links: Vec<usize>,
    mat_r: Handle<StandardMaterial>,
    mat_y: Handle<StandardMaterial>,
    mat_g: Handle<StandardMaterial>,
}

#[derive(Resource, Default)]
struct LightMats {
    off: Handle<StandardMaterial>,
}

#[derive(Resource)]
struct Launcher {
    routes: Vec<u32>,
    models: Vec<(String, PathBuf)>, // label, path
    route_idx: usize,
    model_idx: usize, // 0 = Built-in
    gui: bool,
    /// true = GLB models; false = colored boxes
    use_glb: bool,
    prev_use_glb: bool,
    speed: f32,
    status: String,
    running: bool,
}

fn repo_paths() -> (PathBuf, PathBuf, PathBuf) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let scenario = root.join("ppo").join("scenario");
    let models = root.join("ppo").join("models");
    let sumo3d = root.join("sumo3d");
    (scenario, models, sumo3d)
}

fn scan_launcher() -> Launcher {
    let (scenario, models_dir, _) = repo_paths();
    let mut routes = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&scenario) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(rest) = name.strip_prefix("routes_") {
                if let Some(n) = rest.strip_suffix(".rou.xml") {
                    if let Ok(i) = n.parse::<u32>() {
                        routes.push(i);
                    }
                }
            }
        }
    }
    routes.sort();
    let mut models = vec![("Built-in SUMO TLS".into(), PathBuf::new())];
    if models_dir.is_dir() {
        fn walk(dir: &Path, out: &mut Vec<(String, PathBuf)>) {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        walk(&p, out);
                    } else if p.extension().and_then(|x| x.to_str()) == Some("zip") {
                        let label = p.file_name().unwrap_or_default().to_string_lossy().into_owned();
                        let parent = p
                            .parent()
                            .and_then(|x| x.file_name())
                            .map(|x| x.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let label = if parent.is_empty() {
                            label
                        } else {
                            format!("{parent}/{label}")
                        };
                        out.push((label, p));
                    }
                }
            }
        }
        let mut found = Vec::new();
        walk(&models_dir, &mut found);
        found.sort_by(|a, b| a.0.cmp(&b.0));
        models.extend(found);
    }
    let route_idx = routes.iter().position(|&r| r == 50).unwrap_or(0);
    Launcher {
        routes,
        models,
        route_idx,
        model_idx: 0,
        gui: true,
        use_glb: true,
        prev_use_glb: true,
        speed: 3.0,
        status: "Pick route / controller / SUMO window, then Start.".into(),
        running: false,
    }
}

fn main() {
    let bridge = spawn_bridge("ws://127.0.0.1:8765/ws".into());

    // Release .exe lives in target/release/; Bevy would look there for assets/
    // unless we pin the project assets folder (where the GLBs actually are).
    let assets_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "SUMO 3D Native".into(),
                        resolution: (1400.0_f32, 800.0_f32).into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    file_path: assets_path.to_string_lossy().into_owned(),
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin)
        .insert_resource(ClearColor(Color::srgb(0.06, 0.09, 0.12)))
        .insert_resource(Bridge(bridge))
        .insert_resource(BridgeProc(None))
        .insert_resource(NetBuilt(false))
        .insert_resource(VehicleMap::default())
        .insert_resource(GlbNorm::default())
        .insert_resource(GlbProbeQueued::default())
        .insert_resource(scan_launcher())
        .insert_resource(ApproachCams::default())
        .insert_resource(UiPanels::default())
        .add_systems(Startup, (setup_scene, setup_shared_meshes, load_glb_catalog))
        .add_systems(
            Update,
            (
                launcher_ui,
                approach_cam_ui,
                sync_approach_cam_active,
                poll_bridge,
            ),
        )
        .add_systems(
            Update,
            (
                normalize_glb_probes,
                smooth_vehicles,
                update_signal_lights,
                free_camera,
                handle_keys,
            ),
        )
        .run();
}

fn setup_scene(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(200.0, 280.0, 320.0).looking_at(Vec3::new(750.0, 0.0, -250.0), Vec3::Y),
        OrbitCam {
            target: Vec3::new(750.0, 0.0, -250.0),
            radius: 420.0,
            yaw: 0.6,
            pitch: 0.95,
        },
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(200.0, 400.0, 120.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.45, 0.52, 0.58),
        brightness: 220.0,
    });

    let off = materials.add(StandardMaterial {
        base_color: Color::srgb(0.16, 0.19, 0.22),
        emissive: LinearRgba::BLACK,
        unlit: true,
        ..default()
    });
    commands.insert_resource(LightMats { off });
}

fn setup_shared_meshes(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    let mut by_kind = HashMap::new();
    for kind in ["car", "bus", "truck", "van", "motorcycle"] {
        let s = vehicle_size(kind);
        by_kind.insert(kind, meshes.add(Cuboid::new(s.x, s.y, s.z)));
    }
    let signal = meshes.add(Cuboid::new(0.12, 0.1, 0.18));
    commands.insert_resource(SharedVehicleMeshes { by_kind, signal });
}

fn load_glb_catalog(mut commands: Commands, asset_server: Res<AssetServer>) {
    let mut scenes = HashMap::new();
    for kind in ["car", "bus", "truck", "van", "motorcycle"] {
        let (path, _, _) = glb_catalog_spec(kind);
        scenes.insert(kind, asset_server.load(path));
    }
    commands.insert_resource(GlbCatalog { scenes });
    eprintln!("[assets] GLB catalog queued (assets/models/)");
}

/// Kill whatever is listening on the TraCI bridge port (Windows).
fn free_bridge_port(port: u16) {
    #[cfg(windows)]
    {
        let _ = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "$pids = Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique; foreach ($p in $pids) {{ Stop-Process -Id $p -Force -ErrorAction SilentlyContinue }}"
                ),
            ])
            .status();
        std::thread::sleep(Duration::from_millis(400));
    }
}

fn start_bridge_process(launcher: &Launcher) -> Result<Child, String> {
    free_bridge_port(8765);
    let (_, _, sumo3d) = repo_paths();
    let server = sumo3d.join("server.py");
    let venv_py = sumo3d.join("venv").join("Scripts").join("python.exe");
    let (scenario, _, _) = repo_paths();
    if !server.exists() {
        return Err("sumo3d/server.py not found".into());
    }
    let py = if venv_py.exists() {
        venv_py
    } else {
        PathBuf::from("python")
    };
    let route = launcher
        .routes
        .get(launcher.route_idx)
        .copied()
        .unwrap_or(50);
    let mut cmd = Command::new(&py);
    cmd.arg(&server)
        .arg("--scenario")
        .arg(&scenario)
        .arg("--route")
        .arg(route.to_string())
        .arg("--speed")
        .arg(launcher.speed.to_string())
        .arg("--step-length")
        .arg("0.1")
        .current_dir(&sumo3d);
    if launcher.gui {
        cmd.arg("--gui");
    } else {
        cmd.arg("--headless");
    }
    if launcher.model_idx > 0 {
        if let Some((_, path)) = launcher.models.get(launcher.model_idx) {
            cmd.arg("--model").arg(path);
        }
    }
    cmd.spawn().map_err(|e| e.to_string())
}

fn launcher_ui(
    mut contexts: EguiContexts,
    mut launcher: ResMut<Launcher>,
    mut bridge_proc: ResMut<BridgeProc>,
    mut panels: ResMut<UiPanels>,
    bridge: Res<Bridge>,
) {
    let ctx = contexts.ctx_mut();

    if !panels.control_open {
        egui::SidePanel::left("sumo_control_collapsed")
            .exact_width(28.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(8.0);
                    if ui
                        .add(
                            egui::Button::new("▶\nC\nt\nr\nl")
                                .min_size(egui::vec2(22.0, 120.0)),
                        )
                        .clicked()
                    {
                        panels.control_open = true;
                    }
                });
            });
        return;
    }

    egui::SidePanel::left("sumo_control_panel")
        .default_width(300.0)
        .width_range(240.0..=420.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("SUMO 3D Control");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("◀").on_hover_text("Collapse panel").clicked() {
                        panels.control_open = false;
                    }
                });
            });
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.label("Route");
                let route_labels: Vec<String> = launcher
                    .routes
                    .iter()
                    .map(|r| format!("routes_{r}.rou.xml"))
                    .collect();
                if route_labels.is_empty() {
                    ui.colored_label(egui::Color32::RED, "No routes in ppo/scenario");
                } else {
                    egui::ComboBox::from_id_salt("route")
                        .selected_text(
                            route_labels
                                .get(launcher.route_idx)
                                .cloned()
                                .unwrap_or_default(),
                        )
                        .show_ui(ui, |ui| {
                            for (i, label) in route_labels.iter().enumerate() {
                                ui.selectable_value(&mut launcher.route_idx, i, label);
                            }
                        });
                }

                ui.label("Controller (TLS)");
                let model_labels: Vec<String> =
                    launcher.models.iter().map(|(l, _)| l.clone()).collect();
                egui::ComboBox::from_id_salt("model")
                    .selected_text(
                        model_labels
                            .get(launcher.model_idx)
                            .cloned()
                            .unwrap_or_else(|| "Built-in".into()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, label) in model_labels.iter().enumerate() {
                            ui.selectable_value(&mut launcher.model_idx, i, label);
                        }
                    });
                if launcher.models.len() <= 1 {
                    ui.small("No PPO .zip under ppo/models — using built-in TLS.");
                }

                ui.label("SUMO window");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut launcher.gui, true, "SUMO-GUI");
                    ui.radio_value(&mut launcher.gui, false, "Headless");
                });

                ui.label("Vehicles");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut launcher.use_glb, true, "3D models (GLB)");
                    ui.radio_value(&mut launcher.use_glb, false, "Boxes");
                });
                if launcher.use_glb {
                    ui.small("Blinkers/brake stay on; GLBs load from assets/models/.");
                }

                ui.add(egui::Slider::new(&mut launcher.speed, 0.5..=10.0).text("Speed"));

                ui.separator();
                ui.horizontal(|ui| {
                    if !launcher.running {
                        if ui.button("Start").clicked() {
                            if let Some(mut old) = bridge_proc.0.take() {
                                let _ = old.kill();
                                let _ = old.wait();
                            }
                            match start_bridge_process(&launcher) {
                                Ok(child) => {
                                    launcher.status = format!(
                                        "Started pid={} — 3D will connect on :8765",
                                        child.id()
                                    );
                                    launcher.running = true;
                                    bridge_proc.0 = Some(child);
                                    std::thread::sleep(Duration::from_millis(2000));
                                    let route =
                                        launcher.routes.get(launcher.route_idx).copied().unwrap_or(50);
                                    let model = if launcher.model_idx > 0 {
                                        launcher.models[launcher.model_idx]
                                            .1
                                            .to_string_lossy()
                                            .replace('\\', "/")
                                    } else {
                                        String::new()
                                    };
                                    let _ = bridge.0.cmd_tx.send(ClientCmd::Configure {
                                        route,
                                        gui: launcher.gui,
                                        model,
                                    });
                                }
                                Err(e) => {
                                    launcher.status = format!("Start failed: {e}");
                                }
                            }
                        }
                    } else {
                        if ui.button("Stop").clicked() {
                            if let Some(mut c) = bridge_proc.0.take() {
                                let _ = c.kill();
                                let _ = c.wait();
                            }
                            launcher.running = false;
                            launcher.status = "Stopped.".into();
                        }
                        if ui.button("Apply settings").clicked() {
                            let route =
                                launcher.routes.get(launcher.route_idx).copied().unwrap_or(50);
                            let model = if launcher.model_idx > 0 {
                                launcher.models[launcher.model_idx]
                                    .1
                                    .to_string_lossy()
                                    .replace('\\', "/")
                            } else {
                                String::new()
                            };
                            let _ = bridge.0.cmd_tx.send(ClientCmd::Configure {
                                route,
                                gui: launcher.gui,
                                model,
                            });
                            launcher.status = "Sent configure to bridge.".into();
                        }
                    }
                    if ui.button("Pause/Play").clicked() {
                        let _ = bridge.0.cmd_tx.send(ClientCmd::Toggle);
                    }
                });

                ui.separator();
                ui.label(&launcher.status);
                ui.small("Map: LMB orbit · RMB pan · Scroll zoom · WASD · Q/E height");
            });
        });
}

#[derive(SystemParam)]
struct NetBuildAssets<'w> {
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    images: ResMut<'w, Assets<Image>>,
    egui_textures: ResMut<'w, EguiUserTextures>,
    approach_cams: ResMut<'w, ApproachCams>,
    light_mats: Res<'w, LightMats>,
}

fn poll_bridge(
    mut commands: Commands,
    mut net_assets: NetBuildAssets,
    bridge: Res<Bridge>,
    mut net_built: ResMut<NetBuilt>,
    mut vmap: ResMut<VehicleMap>,
    shared: Res<SharedVehicleMeshes>,
    glb: Option<Res<GlbCatalog>>,
    norms: Res<GlbNorm>,
    mut probes: ResMut<GlbProbeQueued>,
    mut q_orbit: Query<&mut OrbitCam>,
    mut q_veh: Query<(&VehicleTag, &mut Transform, &mut VehSignals, &mut VehSmooth)>,
    mut q_tls: Query<&TlsHeadTag>,
    mut launcher: ResMut<Launcher>,
) {
    let rx = bridge.0.rx.lock().expect("bridge rx");
    // Drain backlog — keep only the latest of each kind
    let mut last_network: Option<Network> = None;
    let mut last_frame: Option<(Vec<Vehicle>, Vec<LightState>)> = None;
    let mut last_status: Option<String> = None;
    while let Ok(msg) = rx.try_recv() {
        match msg {
            ServerMsg::Network { network } => last_network = Some(network),
            ServerMsg::Frame {
                vehicles, lights, ..
            } => last_frame = Some((vehicles, lights)),
            ServerMsg::Status {
                controller, done, ..
            } => {
                last_status = Some(format!(
                    "controller={controller}  done={done}"
                ));
            }
            ServerMsg::Error { message } => {
                last_status = Some(format!("error: {message}"));
            }
        }
    }
    drop(rx);

    if let Some(s) = last_status {
        launcher.status = s;
    }

    if let Some(network) = last_network {
        // Rebuild scene on configure
        // Despawn previous network entities is hard without tags — first time only for MVP
        // unless net_built reset
        if !net_built.0 {
            build_network(
                &mut commands,
                &mut net_assets.meshes,
                &mut net_assets.materials,
                &mut net_assets.images,
                &mut net_assets.egui_textures,
                &mut net_assets.approach_cams,
                &net_assets.light_mats,
                &network,
            );
            if let Ok(mut cam) = q_orbit.get_single_mut() {
                let b = &network.bounds;
                let cx = (b.xmin + b.xmax) * 0.5;
                let cy = (b.ymin + b.ymax) * 0.5;
                cam.target = to_world(cx, cy, 0.0);
                cam.radius = ((b.xmax - b.xmin).max(b.ymax - b.ymin)) * 0.55;
            }
            net_built.0 = true;
            eprintln!(
                "[scene] {} lanes, {} tls heads, {} approach cams",
                network.lanes.len(),
                network.tls.iter().map(|t| t.heads.len()).sum::<usize>(),
                net_assets.approach_cams.entries.len()
            );
        }
    }

    // Switching Boxes ↔ GLB: respawn all vehicles
    if launcher.use_glb != launcher.prev_use_glb {
        for (_, e) in vmap.0.drain() {
            commands.entity(e).despawn_recursive();
        }
        launcher.prev_use_glb = launcher.use_glb;
    }

    if let Some((vehicles, lights)) = last_frame {
        apply_vehicles(
            &mut commands,
            &mut net_assets.materials,
            &mut vmap,
            &mut q_veh,
            &shared,
            glb.as_deref(),
            &norms,
            &mut probes,
            launcher.use_glb,
            &vehicles,
        );
        apply_lights(&mut net_assets.materials, &mut q_tls, &lights);
    }
}

fn build_network(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    egui_textures: &mut EguiUserTextures,
    approach_cams: &mut ApproachCams,
    _light_mats: &LightMats,
    network: &Network,
) {
    let road_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.09, 0.10),
        perceptual_roughness: 0.95,
        ..default()
    });
    let junction_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.07, 0.08, 0.09),
        perceptual_roughness: 0.95,
        ..default()
    });
    let ground_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.32, 0.16),
        perceptual_roughness: 1.0,
        ..default()
    });
    let sidewalk_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.38, 0.40, 0.42),
        perceptual_roughness: 0.85,
        ..default()
    });
    let curb_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.50, 0.52, 0.54),
        perceptual_roughness: 0.8,
        ..default()
    });
    let line_white = materials.add(StandardMaterial {
        base_color: Color::srgba(0.91, 0.93, 0.96, 0.9),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let line_yellow = materials.add(StandardMaterial {
        base_color: Color::srgba(0.94, 0.75, 0.25, 0.9),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });
    let zebra_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.94, 0.94, 0.94, 0.92),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    let b = &network.bounds;
    let cx = (b.xmin + b.xmax) * 0.5;
    let cy = (b.ymin + b.ymax) * 0.5;
    let gw = (b.xmax - b.xmin) + 400.0;
    let gh = (b.ymax - b.ymin) + 400.0;
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(gw, gh))),
        MeshMaterial3d(ground_mat),
        Transform::from_translation(to_world(cx, cy, -0.05)),
    ));

    let mut edge_lanes: HashMap<&str, Vec<&Lane>> = HashMap::new();
    for lane in &network.lanes {
        edge_lanes.entry(lane.edge.as_str()).or_default().push(lane);
    }

    for lane in &network.lanes {
        if lane.kind == "ped" || lane.shape.len() < 2 {
            continue;
        }
        // Skip internal junction connectors — they cause z-fighting / “messy” intersection fill.
        // Junction polygons cover the intersection instead.
        if lane.internal {
            continue;
        }
        if lane.kind == "crossing" {
            // Asphalt under zebra so grass never shows at crosswalks
            if let Some(m) = ribbon_mesh(&lane.shape, lane.width.max(2.0) + 1.5, 0.022) {
                commands.spawn((
                    Mesh3d(meshes.add(m)),
                    MeshMaterial3d(junction_mat.clone()),
                    Transform::IDENTITY,
                ));
            }
            if let Some(z) = zebra_mesh(&lane.shape, lane.width) {
                commands.spawn((
                    Mesh3d(meshes.add(z)),
                    MeshMaterial3d(zebra_mat.clone()),
                    Transform::IDENTITY,
                ));
            }
            continue;
        }
        let y = 0.02;
        if let Some(m) = ribbon_mesh(&lane.shape, lane.width.max(0.5), y) {
            commands.spawn((
                Mesh3d(meshes.add(m)),
                MeshMaterial3d(road_mat.clone()),
                Transform::IDENTITY,
            ));
        }
        if lane.kind == "road" {
            let siblings: Vec<&&Lane> = edge_lanes
                .get(lane.edge.as_str())
                .map(|v| v.iter().filter(|s| s.kind == "road").collect())
                .unwrap_or_default();
            let max_idx = siblings.iter().map(|s| s.index).max().unwrap_or(0);
            // LHT: lane 0 = leftmost. +offset = left of travel, -offset = right.
            // Outer white on left of lane 0; yellow on right of rightmost; dashes between lanes.
            if lane.index == 0 {
                if let Some(m) = solid_line_mesh(&lane.shape, lane.width * 0.48, y + 0.005, 0.12) {
                    commands.spawn((
                        Mesh3d(meshes.add(m)),
                        MeshMaterial3d(line_white.clone()),
                        Transform::IDENTITY,
                    ));
                }
            }
            if lane.index == max_idx {
                if let Some(m) = solid_line_mesh(&lane.shape, -lane.width * 0.48, y + 0.005, 0.12) {
                    commands.spawn((
                        Mesh3d(meshes.add(m)),
                        MeshMaterial3d(line_yellow.clone()),
                        Transform::IDENTITY,
                    ));
                }
            }
            // Separator on the RIGHT edge of each lane except the rightmost
            if lane.index < max_idx {
                if let Some(m) =
                    dashed_line_mesh(&lane.shape, -lane.width * 0.48, y + 0.005, 3.0, 5.0, 0.12)
                {
                    commands.spawn((
                        Mesh3d(meshes.add(m)),
                        MeshMaterial3d(line_white.clone()),
                        Transform::IDENTITY,
                    ));
                }
            }
        }
    }

    spawn_sidewalks(commands, meshes, sidewalk_mat, curb_mat, &network.lanes);
    // Asphalt junction + crossing pads above the grass plane.
    for j in &network.junctions {
        if j.kind == "dead_end" {
            continue;
        }
        let radius = estimate_junction_radius(j.x, j.y, &network.lanes).max(10.0);
        // Always lay a solid disk so grass never shows through, even if shape is bad.
        if let Some(m) = junction_disk_mesh(j.x, j.y, radius) {
            commands.spawn((
                Mesh3d(meshes.add(m)),
                MeshMaterial3d(junction_mat.clone()),
                Transform::from_xyz(0.0, 0.03, 0.0),
            ));
        }
        if j.shape.len() >= 3 {
            if let Some(m) = junction_mesh(&j.shape) {
                commands.spawn((
                    Mesh3d(meshes.add(m)),
                    MeshMaterial3d(junction_mat.clone()),
                    Transform::from_xyz(0.0, 0.035, 0.0),
                ));
            }
        }
    }
    spawn_buildings_sparse(commands, meshes, materials, &network.lanes);
    spawn_tls_gantries(commands, meshes, materials, &network.lanes, &network.tls);
    spawn_approach_cameras(
        commands,
        images,
        egui_textures,
        approach_cams,
        &network.lanes,
        &network.tls,
    );
}

fn estimate_junction_radius(jx: f32, jy: f32, lanes: &[Lane]) -> f32 {
    let mut best = 6.0f32;
    for lane in lanes {
        if lane.internal || lane.shape.is_empty() {
            continue;
        }
        for p in [&lane.shape[0], lane.shape.last().unwrap()] {
            let dx = p[0] - jx;
            let dy = p[1] - jy;
            let d = (dx * dx + dy * dy).sqrt();
            if d < 45.0 {
                best = best.max(d + lane.width * 0.5 + 1.0);
            }
        }
    }
    best.min(40.0)
}

fn junction_disk_mesh(x: f32, y: f32, radius: f32) -> Option<Mesh> {
    let n = 20usize;
    let mut shape = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        shape.push([x + radius * a.cos(), y + radius * a.sin()]);
    }
    junction_mesh(&shape)
}

fn spawn_sidewalks(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    sidewalk_mat: Handle<StandardMaterial>,
    curb_mat: Handle<StandardMaterial>,
    lanes: &[Lane],
) {
    let mut by_edge: HashMap<&str, Vec<&Lane>> = HashMap::new();
    for lane in lanes {
        if lane.internal || lane.kind != "road" {
            continue;
        }
        by_edge.entry(lane.edge.as_str()).or_default().push(lane);
    }
    for list in by_edge.values() {
        let mut sorted = list.clone();
        sorted.sort_by_key(|l| l.index);
        // SUMO lane 0 = rightmost. Only place sidewalk on the RIGHT curb —
        // never on the left (that created the gray “ped strip” in the median).
        // Left-hand traffic: lane 0 = leftmost = outer curb (nearside).
        let Some(curb_lane) = sorted.first().copied() else {
            continue;
        };
        if curb_lane.shape.len() < 2 {
            continue;
        }
        let sw = 2.2;
        let y = 0.06;
        let side_sign = 1.0_f32; // left of travel (LHT curb)
        let shape = trim_shape_ends(&curb_lane.shape, 5.0);
        let half = curb_lane.width.max(0.5) * 0.5;
        let offset = (half + sw * 0.5 + 0.2) * side_sign;
        let pts = offset_polyline(&shape, offset);
        if let Some(m) = ribbon_mesh(&pts, sw, y) {
            commands.spawn((
                Mesh3d(meshes.add(m)),
                MeshMaterial3d(sidewalk_mat.clone()),
                Transform::IDENTITY,
            ));
        }
        let curb_off = offset - side_sign * (sw * 0.5 + 0.1);
        let curb_pts = offset_polyline(&shape, curb_off);
        if let Some(m) = ribbon_mesh(&curb_pts, 0.2, y + 0.1) {
            commands.spawn((
                Mesh3d(meshes.add(m)),
                MeshMaterial3d(curb_mat.clone()),
                Transform::IDENTITY,
            ));
        }
    }
}

fn spawn_buildings_sparse(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    lanes: &[Lane],
) {
    let colors = [
        Color::srgb(0.20, 0.24, 0.28),
        Color::srgb(0.28, 0.26, 0.22),
        Color::srgb(0.22, 0.28, 0.30),
        Color::srgb(0.26, 0.22, 0.24),
        Color::srgb(0.18, 0.22, 0.26),
    ];
    let body_mats: Vec<_> = colors
        .iter()
        .map(|c| {
            materials.add(StandardMaterial {
                base_color: *c,
                perceptual_roughness: 0.88,
                ..default()
            })
        })
        .collect();
    let roof_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.14, 0.16),
        perceptual_roughness: 0.92,
        ..default()
    });
    let window_on = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.88, 0.55),
        emissive: LinearRgba::from(Color::srgb(1.0, 0.85, 0.45)) * 6.0,
        unlit: true,
        ..default()
    });
    let window_off = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.10, 0.14),
        unlit: true,
        ..default()
    });
    let mut placed = 0usize;
    let max_buildings = 140usize;
    let spacing = 55.0;
    let mut rng = 0xC0FFEEu32;
    let mut next_f = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 8) as f32 / (1u32 << 24) as f32
    };

    for lane in lanes {
        if placed >= max_buildings {
            break;
        }
        if lane.shape.len() < 2 || lane.internal || lane.kind != "road" || lane.index != 0 {
            continue;
        }
        let len = total_len(&lane.shape);
        if len < 40.0 {
            continue;
        }
        // LHT: sidewalk is on the LEFT of travel (+nx). Keep buildings beyond it.
        let mut pos = 18.0 + next_f() * 16.0;
        while pos < len - 18.0 && placed < max_buildings {
            let pt = sample_at(&lane.shape, pos);
            let nxt = sample_at(&lane.shape, (pos + 2.0).min(len));
            let dx = nxt[0] - pt[0];
            let dy = nxt[1] - pt[1];
            let dl = (dx * dx + dy * dy).sqrt().max(1e-4);
            let nx = -dy / dl; // left of travel
            let ny = dx / dl;
            let bw = 8.0 + next_f() * 14.0;
            let bd = 8.0 + next_f() * 12.0;
            let bh = 10.0 + next_f() * 28.0;
            // Clearance: half lane + sidewalk 2.2 + gap + half building depth
            let min_clear = lane.width * 0.5 + 2.4 + 3.0 + bd * 0.5;
            let offset = min_clear + 4.0 + next_f() * 18.0;
            let bx = pt[0] + nx * offset;
            let by = pt[1] + ny * offset;
            let yaw = dx.atan2(dy);
            let mat = body_mats[placed % body_mats.len()].clone();
            let wp = to_world(bx, by, 0.0);
            let root = commands
                .spawn((
                    Transform::from_translation(wp).with_rotation(Quat::from_rotation_y(yaw)),
                    Visibility::default(),
                ))
                .id();
            commands.entity(root).with_children(|p| {
                p.spawn((
                    Mesh3d(meshes.add(Cuboid::new(bw, bh, bd))),
                    MeshMaterial3d(mat),
                    Transform::from_xyz(0.0, bh * 0.5, 0.0),
                ));
                p.spawn((
                    Mesh3d(meshes.add(Cuboid::new(bw * 0.92, 0.4, bd * 0.92))),
                    MeshMaterial3d(roof_mat.clone()),
                    Transform::from_xyz(0.0, bh + 0.2, 0.0),
                ));
                // Lit windows on street-facing facade
                let floors = ((bh / 3.2).floor() as i32).clamp(2, 8);
                let cols = ((bw / 2.8).floor() as i32).clamp(2, 5);
                for fi in 0..floors {
                    for ci in 0..cols {
                        if next_f() < 0.28 {
                            continue;
                        }
                        let lit = next_f() > 0.35;
                        let wx = -bw * 0.35 + (ci as f32 + 0.5) * (bw * 0.7 / cols as f32);
                        let wy = 2.0 + fi as f32 * (bh - 3.0) / floors as f32;
                        let wz = bd * 0.51;
                        p.spawn((
                            Mesh3d(meshes.add(Cuboid::new(0.7, 0.9, 0.08))),
                            MeshMaterial3d(if lit {
                                window_on.clone()
                            } else {
                                window_off.clone()
                            }),
                            Transform::from_xyz(wx, wy, wz),
                        ));
                    }
                }
            });
            placed += 1;
            pos += spacing + next_f() * 30.0;
        }
    }
    eprintln!("[scene] buildings: {placed}");
}

fn spawn_tls_gantries(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    lanes: &[Lane],
    tls_list: &[Tls],
) {
    // Edge → road lanes (for curb-accurate mast placement)
    let mut lanes_by_edge: HashMap<&str, Vec<&Lane>> = HashMap::new();
    for lane in lanes {
        if lane.internal || lane.kind != "road" {
            continue;
        }
        lanes_by_edge
            .entry(lane.edge.as_str())
            .or_default()
            .push(lane);
    }
    let pole_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.62, 0.64, 0.66),
        metallic: 0.7,
        perceptual_roughness: 0.35,
        ..default()
    });
    let housing_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.05, 0.05, 0.06),
        metallic: 0.25,
        perceptual_roughness: 0.55,
        ..default()
    });

    for tl in tls_list {
        let heads = if tl.heads.is_empty() {
            vec![protocol::TlsHead {
                x: tl.x,
                y: tl.y,
                facing: 0.0,
                travel: 0.0,
                links: vec![],
                edge: String::new(),
                lane_index: 0,
                turn: String::from("S"),
            }]
        } else {
            tl.heads.clone()
        };

        let mut by_edge: HashMap<String, Vec<protocol::TlsHead>> = HashMap::new();
        for h in heads {
            let key = if h.edge.is_empty() {
                format!("@{:.1},{:.1}", h.x, h.y)
            } else {
                h.edge.clone()
            };
            by_edge.entry(key).or_default().push(h);
        }

        for (edge_id, mut group) in by_edge {
            // LHT driver view: L near left curb pole → S → R into roadway
            group.sort_by(|a, b| {
                turn_rank(&a.turn)
                    .cmp(&turn_rank(&b.turn))
                    .then(a.lane_index.cmp(&b.lane_index))
            });
            let n = group.len().max(1);
            let travel = group[0].travel;
            let facing_yaw = sumo_yaw(group[0].facing);
            let gantry_yaw = sumo_yaw(travel);

            let trad = travel * PI / 180.0;
            let tdx = trad.sin();
            let tdy = trad.cos();
            let r2x = tdy; // right of travel
            let r2y = -tdx;
            let l2x = -r2x; // left of travel (sidewalk / nearside)
            let l2y = -r2y;

            let lane_w = 3.2_f32;
            let n_f = n as f32;
            let approach_w = lane_w * n_f;

            // Prefer real lane geometry: leftmost lane end → curb → sidewalk mast
            let (pole_x, pole_y) = if let Some(list) = lanes_by_edge.get(edge_id.as_str()) {
                let mut sorted = list.clone();
                sorted.sort_by_key(|l| l.index);
                if let Some(leftmost) = sorted.first() {
                    if leftmost.shape.len() >= 2 {
                        let p1 = *leftmost.shape.last().unwrap();
                        let p0 = leftmost.shape[leftmost.shape.len() - 2];
                        let dx = p1[0] - p0[0];
                        let dy = p1[1] - p0[1];
                        let len = (dx * dx + dy * dy).sqrt().max(1e-4);
                        let ux = dx / len;
                        let uy = dy / len;
                        let lx = -uy; // left of travel (LHT sidewalk)
                        let ly = ux;
                        let half = leftmost.width.max(0.5) * 0.5;
                        // Road edge (curb), then 0.9m onto the pedestrian walk
                        let curb_x = p1[0] - ux * 5.0 + lx * half;
                        let curb_y = p1[1] - uy * 5.0 + ly * half;
                        (curb_x + lx * 0.9, curb_y + ly * 0.9)
                    } else {
                        let cx: f32 = group.iter().map(|h| h.x).sum::<f32>() / n_f;
                        let cy: f32 = group.iter().map(|h| h.y).sum::<f32>() / n_f;
                        (
                            cx + l2x * (approach_w * 0.5 + 0.9) - tdx * 4.0,
                            cy + l2y * (approach_w * 0.5 + 0.9) - tdy * 4.0,
                        )
                    }
                } else {
                    let cx: f32 = group.iter().map(|h| h.x).sum::<f32>() / n_f;
                    let cy: f32 = group.iter().map(|h| h.y).sum::<f32>() / n_f;
                    (
                        cx + l2x * (approach_w * 0.5 + 0.9) - tdx * 4.0,
                        cy + l2y * (approach_w * 0.5 + 0.9) - tdy * 4.0,
                    )
                }
            } else {
                let cx: f32 = group.iter().map(|h| h.x).sum::<f32>() / n_f;
                let cy: f32 = group.iter().map(|h| h.y).sum::<f32>() / n_f;
                (
                    cx + l2x * (approach_w * 0.5 + 0.9) - tdx * 4.0,
                    cy + l2y * (approach_w * 0.5 + 0.9) - tdy * 4.0,
                )
            };
            let pole_pos = to_world(pole_x, pole_y, 0.0);

            // Mast sits 0.9m onto sidewalk; heads start after that at lane centers
            let sidewalk_inset = 0.9_f32;
            let arm_len = sidewalk_inset + approach_w + 0.25;
            let arm_h = 6.2;
            let pole_h = 7.0;

            let mut head_specs = Vec::with_capacity(n);
            for head in &group {
                let mat_r = materials.add(StandardMaterial {
                    base_color: Color::srgb(0.12, 0.12, 0.12),
                    unlit: true,
                    ..default()
                });
                let mat_y = materials.add(StandardMaterial {
                    base_color: Color::srgb(0.12, 0.12, 0.12),
                    unlit: true,
                    ..default()
                });
                let mat_g = materials.add(StandardMaterial {
                    base_color: Color::srgb(0.12, 0.12, 0.12),
                    unlit: true,
                    ..default()
                });
                head_specs.push((head.links.clone(), mat_r, mat_y, mat_g));
            }

            let mast_mesh = meshes.add(Cuboid::new(0.28, pole_h, 0.28));
            let arm_mesh = meshes.add(Cuboid::new(0.24, 0.24, arm_len));
            let rod_mesh = meshes.add(Cuboid::new(0.07, 0.55, 0.07));
            let box_mesh = meshes.add(Cuboid::new(0.42, 1.25, 0.34));
            let lens_mesh = meshes.add(Sphere::new(0.14));

            let gantry = commands
                .spawn((
                    Transform::from_translation(pole_pos)
                        .with_rotation(Quat::from_rotation_y(gantry_yaw)),
                    Visibility::default(),
                ))
                .id();

            commands.entity(gantry).with_children(|p| {
                // Mast flush with left roadside
                p.spawn((
                    Mesh3d(mast_mesh),
                    MeshMaterial3d(pole_mat.clone()),
                    Transform::from_xyz(0.0, pole_h * 0.5, 0.0),
                ));
                // Arm over lanes only (local +Z = into roadway)
                p.spawn((
                    Mesh3d(arm_mesh),
                    MeshMaterial3d(pole_mat.clone()),
                    Transform::from_xyz(0.0, arm_h, arm_len * 0.5),
                ));

                // Lane centers measured from curb (past the sidewalk inset)
                for (i, (links, mat_r, mat_y, mat_g)) in head_specs.into_iter().enumerate() {
                    let z = sidewalk_inset + (i as f32 + 0.5) * lane_w;
                    p.spawn((
                        Transform::from_xyz(0.0, arm_h, z)
                            .with_rotation(Quat::from_rotation_y(facing_yaw - gantry_yaw)),
                        Visibility::default(),
                        TlsHeadTag {
                            tls_id: tl.id.clone(),
                            links,
                            mat_r: mat_r.clone(),
                            mat_y: mat_y.clone(),
                            mat_g: mat_g.clone(),
                        },
                    ))
                    .with_children(|hp| {
                        hp.spawn((
                            Mesh3d(rod_mesh.clone()),
                            MeshMaterial3d(pole_mat.clone()),
                            Transform::from_xyz(0.0, -0.2, 0.0),
                        ));
                        hp.spawn((
                            Mesh3d(box_mesh.clone()),
                            MeshMaterial3d(housing_mat.clone()),
                            Transform::from_xyz(0.0, -1.0, 0.0),
                        ));
                        for (yy, mat) in [(-0.55, mat_r), (-1.0, mat_y), (-1.45, mat_g)] {
                            hp.spawn((
                                Mesh3d(lens_mesh.clone()),
                                MeshMaterial3d(mat),
                                Transform::from_xyz(0.2, yy, 0.0),
                            ));
                        }
                    });
                }
            });
        }
    }
}

fn turn_rank(turn: &str) -> u8 {
    match turn.to_uppercase().as_str() {
        "L" => 0,
        "S" => 1,
        "R" => 2,
        _ => 1,
    }
}

fn color_from_links(state: &str, links: &[usize]) -> char {
    if state.is_empty() {
        return 'o';
    }
    if links.is_empty() {
        if state.chars().any(|c| c == 'G' || c == 'g') && !state.chars().any(|c| c == 'y' || c == 'Y')
        {
            return 'G';
        }
        if state.chars().any(|c| c == 'y' || c == 'Y') {
            return 'y';
        }
        if state.contains('r') {
            return 'r';
        }
        return 'o';
    }
    let bytes: Vec<char> = state.chars().collect();
    let mut has_g = false;
    let mut has_y = false;
    let mut has_r = false;
    for &i in links {
        match bytes.get(i).copied().unwrap_or(' ') {
            'G' | 'g' => has_g = true,
            'y' | 'Y' => has_y = true,
            'r' => has_r = true,
            _ => {}
        }
    }
    if has_g {
        'G'
    } else if has_y {
        'y'
    } else if has_r {
        'r'
    } else {
        'o'
    }
}

fn apply_lights(
    materials: &mut Assets<StandardMaterial>,
    q_tls: &mut Query<&TlsHeadTag>,
    lights: &[LightState],
) {
    let by_id: HashMap<&str, &str> = lights
        .iter()
        .map(|l| (l.id.as_str(), l.state.as_str()))
        .collect();
    for head in q_tls.iter() {
        let state = by_id.get(head.tls_id.as_str()).copied().unwrap_or("");
        let mode = color_from_links(state, &head.links);
        set_lens(materials, &head.mat_r, mode == 'r', Color::srgb(1.0, 0.23, 0.23));
        set_lens(materials, &head.mat_y, mode == 'y', Color::srgb(1.0, 0.79, 0.20));
        set_lens(materials, &head.mat_g, mode == 'G', Color::srgb(0.24, 1.0, 0.54));
    }
}

fn set_lens(materials: &mut Assets<StandardMaterial>, handle: &Handle<StandardMaterial>, on: bool, color: Color) {
    if let Some(mat) = materials.get_mut(handle) {
        if on {
            mat.base_color = color;
            mat.emissive = LinearRgba::from(color) * 12.0;
            mat.unlit = true;
        } else {
            mat.base_color = Color::srgb(0.10, 0.10, 0.10);
            mat.emissive = LinearRgba::BLACK;
            mat.unlit = true;
        }
    }
}

fn spawn_signal_children(
    p: &mut ChildBuilder,
    sig_mesh: Handle<Mesh>,
    mat_bl: Handle<StandardMaterial>,
    mat_br: Handle<StandardMaterial>,
    mat_brake: Handle<StandardMaterial>,
    size: Vec3,
) {
    // Entity origin = vehicle center; length along +X (nose +X), rear = -X.
    let rear = -size.x * 0.42;
    let half_w = size.z * 0.32;
    let hy = (size.y * 0.45).clamp(0.45, 1.35);
    let s = 0.7;
    p.spawn((
        Mesh3d(sig_mesh.clone()),
        MeshMaterial3d(mat_bl),
        Transform::from_xyz(rear, hy, half_w).with_scale(Vec3::splat(s)),
    ));
    p.spawn((
        Mesh3d(sig_mesh.clone()),
        MeshMaterial3d(mat_br),
        Transform::from_xyz(rear, hy, -half_w).with_scale(Vec3::splat(s)),
    ));
    p.spawn((
        Mesh3d(sig_mesh),
        MeshMaterial3d(mat_brake),
        Transform::from_xyz(rear, hy * 0.9, 0.0)
            .with_scale(Vec3::new(s * 0.85, s * 0.65, s * 1.8)),
    ));
}

fn lerp_angle(from: f32, to: f32, t: f32) -> f32 {
    let mut d = (to - from) % (2.0 * PI);
    if d > PI {
        d -= 2.0 * PI;
    } else if d < -PI {
        d += 2.0 * PI;
    }
    from + d * t
}

fn smooth_vehicles(
    time: Res<Time>,
    mut q: Query<(&mut Transform, &VehSmooth)>,
) {
    let dt = time.delta_secs();
    // Higher = snappier follow; still smooth between 10 Hz TraCI poses
    let alpha = (1.0 - (-14.0 * dt).exp()).clamp(0.0, 1.0);
    for (mut tf, smooth) in &mut q {
        tf.translation = tf.translation.lerp(smooth.target_pos, alpha);
        let cur_yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
        let yaw = lerp_angle(cur_yaw, smooth.target_yaw, alpha);
        tf.rotation = Quat::from_rotation_y(yaw);
    }
}

fn apply_vehicles(
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    vmap: &mut VehicleMap,
    q_veh: &mut Query<(&VehicleTag, &mut Transform, &mut VehSignals, &mut VehSmooth)>,
    shared: &SharedVehicleMeshes,
    glb: Option<&GlbCatalog>,
    norms: &GlbNorm,
    probes: &mut GlbProbeQueued,
    use_glb: bool,
    vehicles: &[Vehicle],
) {
    let mut seen = HashSet::new();
    for v in vehicles {
        seen.insert(v.id.clone());
        let kind = resolve_type(&v.vtype);
        let size = vehicle_size(kind);
        let base_yaw = sumo_yaw(v.angle);

        let (pos, yaw) = if use_glb {
            if let Some(fix) = norms.0.get(kind).copied() {
                // Origin = ground center; mesh offset lives on the Scene child.
                (to_world(v.x, v.y, 0.0), base_yaw + fix.yaw_extra)
            } else {
                (to_world(v.x, v.y, size.y * 0.5), base_yaw)
            }
        } else {
            (to_world(v.x, v.y, size.y * 0.5), base_yaw)
        };

        if let Some(&entity) = vmap.0.get(&v.id) {
            if let Ok((_, _tf, mut sig, mut smooth)) = q_veh.get_mut(entity) {
                smooth.target_pos = pos;
                smooth.target_yaw = yaw;
                sig.bits = v.signals;
            }
            continue;
        }

        let mat_bl = materials.add(signal_mat_off());
        let mat_br = materials.add(signal_mat_off());
        let mat_brake = materials.add(signal_mat_off());
        let sig_mesh = shared.signal.clone();
        let color = vehicle_color(kind, &v.id);

        let use_scene = use_glb
            && norms.0.contains_key(kind)
            && glb.and_then(|g| g.scenes.get(kind)).is_some();

        let entity = if use_scene {
            let scene = glb.unwrap().scenes.get(kind).unwrap().clone();
            let fix = *norms.0.get(kind).unwrap();
            commands
                .spawn((
                    Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(yaw)),
                    Visibility::default(),
                    VehicleTag { id: v.id.clone() },
                    VehSmooth {
                        target_pos: pos,
                        target_yaw: yaw,
                    },
                    VehSignals {
                        bits: v.signals,
                        mat_bl: mat_bl.clone(),
                        mat_br: mat_br.clone(),
                        mat_brake: mat_brake.clone(),
                    },
                ))
                .with_children(|p| {
                    p.spawn((
                        SceneRoot(scene),
                        Transform::from_translation(fix.offset)
                            .with_scale(Vec3::splat(fix.scale)),
                    ));
                    spawn_signal_children(p, sig_mesh, mat_bl, mat_br, mat_brake, size);
                })
                .id()
        } else {
            let Some(mesh) = shared.by_kind.get(kind).cloned() else {
                continue;
            };
            let body_mat = materials.add(StandardMaterial {
                base_color: color,
                metallic: 0.25,
                perceptual_roughness: 0.45,
                ..default()
            });
            commands
                .spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(body_mat),
                    Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(yaw)),
                    VehicleTag { id: v.id.clone() },
                    VehSmooth {
                        target_pos: pos,
                        target_yaw: yaw,
                    },
                    VehSignals {
                        bits: v.signals,
                        mat_bl: mat_bl.clone(),
                        mat_br: mat_br.clone(),
                        mat_brake: mat_brake.clone(),
                    },
                ))
                .with_children(|p| {
                    spawn_signal_children(p, sig_mesh, mat_bl, mat_br, mat_brake, size);
                })
                .id()
        };
        vmap.0.insert(v.id.clone(), entity);

        // Queue one hidden probe per kind to measure GLB bounds
        if use_glb && !probes.0.contains(kind) {
            if let Some(scene) = glb.and_then(|g| g.scenes.get(kind)).cloned() {
                probes.0.insert(kind);
                commands.spawn((
                    SceneRoot(scene),
                    Transform::from_xyz(-12_000.0, -12_000.0, 0.0),
                    NeedsGlbNorm { kind },
                    Visibility::Hidden,
                ));
            }
        }
    }

    let stale: Vec<String> = vmap
        .0
        .keys()
        .filter(|k| !seen.contains(*k))
        .cloned()
        .collect();
    for id in stale {
        if let Some(e) = vmap.0.remove(&id) {
            commands.entity(e).despawn_recursive();
        }
    }
}

fn normalize_glb_probes(
    mut commands: Commands,
    mut norms: ResMut<GlbNorm>,
    q_needs: Query<(Entity, &NeedsGlbNorm, &GlobalTransform)>,
    children: Query<&Children>,
    aabbs: Query<(&Aabb, &GlobalTransform)>,
    mut vmap: ResMut<VehicleMap>,
) {
    let mut measured_any = false;
    for (entity, needs, root_gt) in &q_needs {
        if norms.0.contains_key(needs.kind) {
            commands.entity(entity).despawn_recursive();
            continue;
        }
        let root_pos = root_gt.translation();
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let mut found = false;
        let mut stack = vec![entity];
        while let Some(e) = stack.pop() {
            if let Ok((aabb, gt)) = aabbs.get(e) {
                let local_min = aabb.center - aabb.half_extents;
                let local_max = aabb.center + aabb.half_extents;
                for corner in [
                    Vec3::new(local_min.x, local_min.y, local_min.z),
                    Vec3::new(local_min.x, local_min.y, local_max.z),
                    Vec3::new(local_min.x, local_max.y, local_min.z),
                    Vec3::new(local_min.x, local_max.y, local_max.z),
                    Vec3::new(local_max.x, local_min.y, local_min.z),
                    Vec3::new(local_max.x, local_min.y, local_max.z),
                    Vec3::new(local_max.x, local_max.y, local_min.z),
                    Vec3::new(local_max.x, local_max.y, local_max.z),
                ] {
                    let w = gt.transform_point(corner) - root_pos;
                    min = min.min(w);
                    max = max.max(w);
                    found = true;
                }
            }
            if let Ok(ch) = children.get(e) {
                for &c in ch.iter() {
                    stack.push(c);
                }
            }
        }
        if !found {
            continue;
        }
        let size = max - min;
        let (_, target_len, yaw_nose) = glb_catalog_spec(needs.kind);
        let mut yaw_extra = yaw_nose;
        let mut len = size.x.max(size.z);
        if size.z > size.x {
            yaw_extra += PI / 2.0;
            len = size.z;
        }
        let scale = if len > 1e-4 { target_len / len } else { 1.0 };
        // Center mesh on entity origin so blinkers/brakes line up with the body.
        let mid = (min + max) * 0.5;
        let offset = Vec3::new(-mid.x * scale, -min.y * scale, -mid.z * scale);
        norms.0.insert(
            needs.kind,
            GlbFix {
                scale,
                yaw_extra,
                offset,
            },
        );
        eprintln!(
            "[glb] {} scale={:.3} yaw_extra={:.2} offset={:?} size={:?}",
            needs.kind, scale, yaw_extra, offset, size
        );
        commands.entity(entity).despawn_recursive();
        measured_any = true;
    }
    // Force vehicle respawn as GLB once norms arrive
    if measured_any {
        for (_, e) in vmap.0.drain() {
            commands.entity(e).despawn_recursive();
        }
    }
}

fn signal_mat_off() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(0.12, 0.12, 0.12),
        emissive: LinearRgba::BLACK,
        unlit: true,
        ..default()
    }
}

fn update_signal_lights(
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    q: Query<&VehSignals>,
) {
    let blink_on = (time.elapsed_secs() * 4.0).fract() < 0.5;
    let amber = Color::srgb(1.0, 0.65, 0.05);
    let red = Color::srgb(1.0, 0.08, 0.05);
    for sig in &q {
        let left = sig.bits & SIG_BLINKER_LEFT != 0 && blink_on;
        let right = sig.bits & SIG_BLINKER_RIGHT != 0 && blink_on;
        let brake = sig.bits & SIG_BRAKELIGHT != 0;
        set_lens(&mut materials, &sig.mat_bl, left, amber);
        set_lens(&mut materials, &sig.mat_br, right, amber);
        set_lens(&mut materials, &sig.mat_brake, brake, red);
    }
}

fn free_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut motion: EventReader<bevy::input::mouse::MouseMotion>,
    mut scroll: EventReader<bevy::input::mouse::MouseWheel>,
    mut contexts: EguiContexts,
    mut q: Query<(&mut Transform, &mut OrbitCam)>,
) {
    let Ok((mut tf, mut cam)) = q.get_single_mut() else {
        return;
    };
    let dt = time.delta_secs();
    let egui_blocks = contexts.ctx_mut().wants_pointer_input();

    // View-aligned ground axes: match looking_at(target) from the orbit offset below.
    let forward = Vec3::new(-cam.yaw.cos(), 0.0, -cam.yaw.sin());
    let right = Vec3::Y.cross(forward);

    for ev in motion.read() {
        if egui_blocks {
            continue;
        }
        if buttons.pressed(MouseButton::Left) {
            cam.yaw -= ev.delta.x * 0.005;
            cam.pitch = (cam.pitch + ev.delta.y * 0.005).clamp(0.12, 1.45);
        } else if buttons.pressed(MouseButton::Right) {
            let speed = cam.radius * 0.0015;
            // Match swapped strafe: drag right → pan view right
            cam.target += right * (ev.delta.x * speed) + Vec3::Y * (ev.delta.y * speed * 0.5);
        }
    }
    for ev in scroll.read() {
        if egui_blocks {
            continue;
        }
        cam.radius = (cam.radius * if ev.y > 0.0 { 0.90 } else { 1.11 }).clamp(15.0, 2000.0);
    }

    // Recompute after yaw changes from this frame's mouse look
    let forward = Vec3::new(-cam.yaw.cos(), 0.0, -cam.yaw.sin());
    let right = Vec3::Y.cross(forward);
    let move_speed = cam.radius * 0.7 * dt;

    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        cam.target += forward * move_speed;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        cam.target -= forward * move_speed;
    }
    // Screen-left / screen-right (Y.cross(forward) was flipped vs view)
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        cam.target += right * move_speed;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        cam.target -= right * move_speed;
    }
    if keys.pressed(KeyCode::KeyQ) {
        cam.target.y += move_speed * 0.65;
    }
    if keys.pressed(KeyCode::KeyE) {
        cam.target.y -= move_speed * 0.65;
    }

    let x = cam.radius * cam.yaw.cos() * cam.pitch.sin();
    let y = cam.radius * cam.pitch.cos();
    let z = cam.radius * cam.yaw.sin() * cam.pitch.sin();
    *tf = Transform::from_translation(cam.target + Vec3::new(x, y, z)).looking_at(cam.target, Vec3::Y);
}

fn handle_keys(keys: Res<ButtonInput<KeyCode>>, bridge: Res<Bridge>) {
    if keys.just_pressed(KeyCode::Space) {
        let _ = bridge.0.cmd_tx.send(ClientCmd::Toggle);
    }
    if keys.just_pressed(KeyCode::KeyR) && keys.pressed(KeyCode::ControlLeft) {
        let _ = bridge.0.cmd_tx.send(ClientCmd::Restart);
    }
}
