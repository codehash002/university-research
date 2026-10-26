//! Per-approach TLS cameras mounted on gantry arms + right-side live feeds.

use crate::geom::to_world;
use crate::protocol::{Lane, Tls};
use bevy::prelude::*;
use bevy::render::camera::{ClearColorConfig, RenderTarget};
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy_egui::{egui, EguiContexts, EguiUserTextures};
use std::collections::HashMap;
use std::f32::consts::PI;

const CAM_W: u32 = 320;
const CAM_H: u32 = 180;
const FEED_W: f32 = 292.0;
const FEED_H: f32 = 164.0;

/// Must match `spawn_tls_gantries` mast/arm heights.
const ARM_H: f32 = 6.2;
const SIDEWALK_INSET: f32 = 0.9;
const LANE_W: f32 = 3.2;

#[derive(Component)]
pub struct ApproachCamView {
    pub index: usize,
}

#[derive(Clone)]
pub struct ApproachCamEntry {
    pub label: String,
    #[allow(dead_code)]
    pub tls_id: String,
    #[allow(dead_code)]
    pub edge: String,
    pub enabled: bool,
    pub image: Handle<Image>,
    pub camera: Entity,
}

#[derive(Resource, Default)]
pub struct ApproachCams {
    pub entries: Vec<ApproachCamEntry>,
}

#[derive(Resource)]
pub struct UiPanels {
    pub control_open: bool,
    pub cameras_open: bool,
}

impl Default for UiPanels {
    fn default() -> Self {
        Self {
            control_open: true,
            cameras_open: true,
        }
    }
}

fn make_rtt_image() -> Image {
    let size = Extent3d {
        width: CAM_W,
        height: CAM_H,
        ..default()
    };
    let mut image = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("approach_cam_rtt"),
            size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Bgra8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        ..default()
    };
    image.resize(size);
    image
}

fn sumo_yaw(angle_deg: f32) -> f32 {
    (90.0 - angle_deg) * PI / 180.0
}

/// Same curb/pole placement as the visual gantries — camera sits on top of the signal arm.
fn gantry_cam_pose(
    edge_id: &str,
    group: &[&crate::protocol::TlsHead],
    lanes_by_edge: &HashMap<&str, Vec<&Lane>>,
) -> (Vec3, Vec3) {
    let n = group.len().max(1);
    let n_f = n as f32;
    let travel = group[0].travel;
    let gantry_yaw = sumo_yaw(travel);
    let trad = travel * PI / 180.0;
    let tdx = trad.sin();
    let tdy = trad.cos();
    let travel_w = Vec3::new(tdx, 0.0, -tdy);
    let r2x = tdy;
    let r2y = -tdx;
    let l2x = -r2x;
    let l2y = -r2y;
    let approach_w = LANE_W * n_f;

    let (pole_x, pole_y) = if let Some(list) = lanes_by_edge.get(edge_id) {
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
                let lx = -uy;
                let ly = ux;
                let half = leftmost.width.max(0.5) * 0.5;
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

    let pole_world = to_world(pole_x, pole_y, 0.0);
    let rot = Quat::from_rotation_y(gantry_yaw);
    // Mid-span on top of the horizontal signal bar (local +Z into roadway).
    let mid_z = SIDEWALK_INSET + approach_w * 0.5;
    let eye = pole_world + rot * Vec3::new(0.0, ARM_H + 0.55, mid_z);
    // Look into incoming traffic, slightly downward like a roadside CCTV.
    let look = eye - travel_w * 55.0 + Vec3::Y * (-5.0);
    (eye, look)
}

/// One camera per TLS approach edge, mounted on the gantry arm.
pub fn spawn_approach_cameras(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    egui_textures: &mut EguiUserTextures,
    approach_cams: &mut ApproachCams,
    lanes: &[Lane],
    tls_list: &[Tls],
) {
    for e in approach_cams.entries.drain(..) {
        commands.entity(e.camera).despawn_recursive();
    }

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

    let mut order = -1_isize;
    for tl in tls_list {
        if tl.heads.is_empty() {
            continue;
        }
        let heads = &tl.heads;

        let mut by_edge: HashMap<String, Vec<&crate::protocol::TlsHead>> = HashMap::new();
        for h in heads {
            let key = if h.edge.is_empty() {
                format!("@{:.0},{:.0}", h.x, h.y)
            } else {
                h.edge.clone()
            };
            by_edge.entry(key).or_default().push(h);
        }

        for (edge_id, group) in by_edge {
            let (eye, look) = gantry_cam_pose(&edge_id, &group, &lanes_by_edge);

            let image_handle = images.add(make_rtt_image());
            egui_textures.add_image(image_handle.clone());

            let idx = approach_cams.entries.len();
            let label = if edge_id.starts_with('@') {
                format!("{} · approach", tl.id)
            } else {
                format!("{} · {}", tl.id, edge_id)
            };

            let camera = commands
                .spawn((
                    Camera3d::default(),
                    Camera {
                        order,
                        is_active: true,
                        target: RenderTarget::Image(image_handle.clone()),
                        clear_color: ClearColorConfig::Custom(Color::srgb(0.12, 0.15, 0.18)),
                        ..default()
                    },
                    Projection::from(PerspectiveProjection {
                        fov: 50.0_f32.to_radians(),
                        aspect_ratio: CAM_W as f32 / CAM_H as f32,
                        ..default()
                    }),
                    Transform::from_translation(eye).looking_at(look, Vec3::Y),
                    ApproachCamView { index: idx },
                    Name::new(format!("ApproachCam/{label}")),
                ))
                .id();

            approach_cams.entries.push(ApproachCamEntry {
                label,
                tls_id: tl.id.clone(),
                edge: edge_id,
                enabled: true,
                image: image_handle,
                camera,
            });

            order -= 1;
        }
    }

    eprintln!(
        "[scene] {} approach cameras (gantry-top incoming views)",
        approach_cams.entries.len()
    );
}

pub fn sync_approach_cam_active(
    cams: Res<ApproachCams>,
    mut q: Query<(&ApproachCamView, &mut Camera)>,
) {
    for (view, mut cam) in q.iter_mut() {
        if let Some(entry) = cams.entries.get(view.index) {
            if cam.is_active != entry.enabled {
                cam.is_active = entry.enabled;
            }
        }
    }
}

pub fn approach_cam_ui(
    mut contexts: EguiContexts,
    mut cams: ResMut<ApproachCams>,
    mut panels: ResMut<UiPanels>,
) {
    let tex_ids: Vec<Option<egui::TextureId>> = cams
        .entries
        .iter()
        .map(|e| contexts.image_id(&e.image))
        .collect();

    let ctx = contexts.ctx_mut();

    if !panels.cameras_open {
        egui::SidePanel::right("approach_cams_collapsed")
            .exact_width(28.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(8.0);
                    if ui
                        .add(egui::Button::new("◀\nC\na\nm\ns").min_size(egui::vec2(22.0, 120.0)))
                        .clicked()
                    {
                        panels.cameras_open = true;
                    }
                });
            });
        return;
    }

    egui::SidePanel::right("approach_cam_panel")
        .default_width(318.0)
        .width_range(260.0..=480.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Approach cameras");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("▶").on_hover_text("Collapse panel").clicked() {
                        panels.cameras_open = false;
                    }
                });
            });
            ui.small("Gantry-top CCTV looking into incoming traffic.");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("All on").clicked() {
                    for e in &mut cams.entries {
                        e.enabled = true;
                    }
                }
                if ui.button("All off").clicked() {
                    for e in &mut cams.entries {
                        e.enabled = false;
                    }
                }
                ui.label(format!(
                    "{}/{} on",
                    cams.entries.iter().filter(|e| e.enabled).count(),
                    cams.entries.len()
                ));
            });
            ui.separator();

            if cams.entries.is_empty() {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "No cameras yet — click Start so the network loads.",
                );
                return;
            }

            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                for i in 0..cams.entries.len() {
                    let label = cams.entries[i].label.clone();
                    let enabled = cams.entries[i].enabled;
                    ui.checkbox(&mut cams.entries[i].enabled, label);
                    if enabled {
                        if let Some(tex_id) = tex_ids.get(i).copied().flatten() {
                            let avail = ui.available_width();
                            let w = avail.min(FEED_W);
                            let h = w * (FEED_H / FEED_W);
                            ui.add(
                                egui::Image::new(egui::load::SizedTexture::new(
                                    tex_id,
                                    egui::vec2(w, h),
                                ))
                                .rounding(4.0),
                            );
                        } else {
                            ui.colored_label(egui::Color32::YELLOW, "Waiting for texture…");
                        }
                    }
                    ui.add_space(8.0);
                }
            });
        });
}
