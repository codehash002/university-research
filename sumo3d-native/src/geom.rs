//! Ribbon / marking geometry helpers (ported from sumo3d engine.js).

use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;

pub fn to_world(x: f32, y: f32, h: f32) -> Vec3 {
    Vec3::new(x, h, -y)
}

fn mesh_from_buffers(positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>, indices: Vec<u32>) -> Mesh {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

pub fn ribbon_mesh(shape: &[[f32; 2]], width: f32, y: f32) -> Option<Mesh> {
    if shape.len() < 2 {
        return None;
    }
    let half = width * 0.5;
    let n = shape.len();
    let mut positions = Vec::with_capacity(n * 2);
    let mut normals = Vec::with_capacity(n * 2);
    let mut indices: Vec<u32> = Vec::with_capacity((n - 1) * 6);

    for i in 0..n {
        let (x, y2) = (shape[i][0], shape[i][1]);
        let (dx, dy) = if i == 0 {
            (shape[1][0] - x, shape[1][1] - y2)
        } else if i + 1 == n {
            (x - shape[i - 1][0], y2 - shape[i - 1][1])
        } else {
            (shape[i + 1][0] - shape[i - 1][0], shape[i + 1][1] - shape[i - 1][1])
        };
        let len = (dx * dx + dy * dy).sqrt().max(1e-4);
        let nx = -dy / len;
        let ny = dx / len;
        let l = to_world(x + nx * half, y2 + ny * half, y);
        let r = to_world(x - nx * half, y2 - ny * half, y);
        positions.push([l.x, l.y, l.z]);
        positions.push([r.x, r.y, r.z]);
        normals.push([0.0, 1.0, 0.0]);
        normals.push([0.0, 1.0, 0.0]);
    }
    for i in 0..(n - 1) {
        let a = (i * 2) as u32;
        indices.extend_from_slice(&[a, a + 1, a + 2, a + 1, a + 3, a + 2]);
    }
    Some(mesh_from_buffers(positions, normals, indices))
}

fn polyline_dists(shape: &[[f32; 2]]) -> Vec<f32> {
    let mut dists = vec![0.0];
    for i in 1..shape.len() {
        let d = ((shape[i][0] - shape[i - 1][0]).powi(2) + (shape[i][1] - shape[i - 1][1]).powi(2))
            .sqrt();
        dists.push(dists[i - 1] + d);
    }
    dists
}

fn sample_polyline(shape: &[[f32; 2]], dists: &[f32], t: f32) -> [f32; 2] {
    if t <= 0.0 {
        return shape[0];
    }
    let last = dists.len() - 1;
    if t >= dists[last] {
        return shape[last];
    }
    for i in 1..dists.len() {
        if dists[i] >= t {
            let seg = dists[i] - dists[i - 1];
            let frac = if seg > 0.0 {
                (t - dists[i - 1]) / seg
            } else {
                0.0
            };
            return [
                shape[i - 1][0] + (shape[i][0] - shape[i - 1][0]) * frac,
                shape[i - 1][1] + (shape[i][1] - shape[i - 1][1]) * frac,
            ];
        }
    }
    shape[0]
}

pub fn solid_line_mesh(shape: &[[f32; 2]], offset: f32, y: f32, line_width: f32) -> Option<Mesh> {
    if shape.len() < 2 {
        return None;
    }
    let mut pts = Vec::with_capacity(shape.len());
    for i in 0..shape.len() {
        let (x, y2) = (shape[i][0], shape[i][1]);
        let (dx, dy) = if i == 0 {
            (shape[1][0] - x, shape[1][1] - y2)
        } else if i + 1 == shape.len() {
            (x - shape[i - 1][0], y2 - shape[i - 1][1])
        } else {
            (shape[i + 1][0] - shape[i - 1][0], shape[i + 1][1] - shape[i - 1][1])
        };
        let len = (dx * dx + dy * dy).sqrt().max(1e-4);
        let nx = -dy / len;
        let ny = dx / len;
        pts.push([x + nx * offset, y2 + ny * offset]);
    }
    ribbon_mesh(&pts, line_width, y)
}

pub fn dashed_line_mesh(
    shape: &[[f32; 2]],
    offset: f32,
    y: f32,
    dash_len: f32,
    gap_len: f32,
    line_width: f32,
) -> Option<Mesh> {
    if shape.len() < 2 {
        return None;
    }
    let half_w = line_width * 0.5;
    let dists = polyline_dists(shape);
    let total = *dists.last()?;
    let cycle = dash_len + gap_len;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut vert = 0u32;
    let mut pos = 0.0;
    while pos < total {
        let start = pos;
        let end = (pos + dash_len).min(total);
        if end - start < 0.3 {
            break;
        }
        let p0 = sample_polyline(shape, &dists, start);
        let p1 = sample_polyline(shape, &dists, end);
        let dx = p1[0] - p0[0];
        let dy = p1[1] - p0[1];
        let dl = (dx * dx + dy * dy).sqrt().max(1e-4);
        let nx = -dy / dl;
        let ny = dx / dl;
        let ox0 = p0[0] + nx * offset;
        let oy0 = p0[1] + ny * offset;
        let ox1 = p1[0] + nx * offset;
        let oy1 = p1[1] + ny * offset;
        let c0 = to_world(ox0 + nx * half_w, oy0 + ny * half_w, y);
        let c1 = to_world(ox0 - nx * half_w, oy0 - ny * half_w, y);
        let c2 = to_world(ox1 + nx * half_w, oy1 + ny * half_w, y);
        let c3 = to_world(ox1 - nx * half_w, oy1 - ny * half_w, y);
        positions.extend_from_slice(&[
            [c0.x, c0.y, c0.z],
            [c1.x, c1.y, c1.z],
            [c2.x, c2.y, c2.z],
            [c3.x, c3.y, c3.z],
        ]);
        normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 4]);
        indices.extend_from_slice(&[vert, vert + 1, vert + 2, vert + 1, vert + 3, vert + 2]);
        vert += 4;
        pos += cycle;
    }
    if vert == 0 {
        return None;
    }
    Some(mesh_from_buffers(positions, normals, indices))
}

pub fn zebra_mesh(shape: &[[f32; 2]], width: f32) -> Option<Mesh> {
    if shape.len() < 2 {
        return None;
    }
    let stripe_w = 0.5;
    let gap = 0.6;
    let half = width * 0.5;
    let dx = shape[shape.len() - 1][0] - shape[0][0];
    let dy = shape[shape.len() - 1][1] - shape[0][1];
    let len = (dx * dx + dy * dy).sqrt().max(1e-4);
    let tx = dx / len;
    let ty = dy / len;
    let nx = -ty;
    let ny = tx;
    let y = 0.05;
    let cx = (shape[0][0] + shape[shape.len() - 1][0]) * 0.5;
    let cy = (shape[0][1] + shape[shape.len() - 1][1]) * 0.5;
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut vert = 0u32;
    let mut pos = -half + 0.3;
    while pos < half - 0.3 {
        let sx = cx + nx * pos;
        let sy = cy + ny * pos;
        let half_len = len * 0.45;
        let hw = stripe_w * 0.5;
        let c0 = to_world(sx - tx * half_len - nx * hw, sy - ty * half_len - ny * hw, y);
        let c1 = to_world(sx - tx * half_len + nx * hw, sy - ty * half_len + ny * hw, y);
        let c2 = to_world(sx + tx * half_len - nx * hw, sy + ty * half_len - ny * hw, y);
        let c3 = to_world(sx + tx * half_len + nx * hw, sy + ty * half_len + ny * hw, y);
        positions.extend_from_slice(&[
            [c0.x, c0.y, c0.z],
            [c1.x, c1.y, c1.z],
            [c2.x, c2.y, c2.z],
            [c3.x, c3.y, c3.z],
        ]);
        normals.extend_from_slice(&[[0.0, 1.0, 0.0]; 4]);
        indices.extend_from_slice(&[vert, vert + 1, vert + 2, vert + 1, vert + 3, vert + 2]);
        vert += 4;
        pos += stripe_w + gap;
    }
    if vert == 0 {
        return None;
    }
    Some(mesh_from_buffers(positions, normals, indices))
}

pub fn junction_mesh(shape: &[[f32; 2]]) -> Option<Mesh> {
    if shape.len() < 3 {
        return None;
    }
    // Fan triangulate in XZ (Bevy Y-up): world x = sumo x, world z = -sumo y
    let mut positions = Vec::with_capacity(shape.len());
    let mut normals = Vec::with_capacity(shape.len());
    let y = 0.015;
    let mut cx = 0.0;
    let mut cz = 0.0;
    for p in shape {
        let w = to_world(p[0], p[1], y);
        positions.push([w.x, w.y, w.z]);
        normals.push([0.0, 1.0, 0.0]);
        cx += w.x;
        cz += w.z;
    }
    cx /= shape.len() as f32;
    cz /= shape.len() as f32;
    // insert centroid as vertex 0 for fan
    positions.insert(0, [cx, y, cz]);
    normals.insert(0, [0.0, 1.0, 0.0]);
    let mut indices = Vec::new();
    let n = shape.len() as u32;
    for i in 0..n {
        let a = i + 1;
        let b = if i + 1 == n { 1 } else { i + 2 };
        // Both windings — SUMO shapes vary; never cull the asphalt pad from above.
        indices.extend_from_slice(&[0, a, b, 0, b, a]);
    }
    Some(mesh_from_buffers(positions, normals, indices))
}

pub fn offset_polyline(shape: &[[f32; 2]], offset: f32) -> Vec<[f32; 2]> {
    let mut pts = Vec::with_capacity(shape.len());
    for i in 0..shape.len() {
        let (x, y2) = (shape[i][0], shape[i][1]);
        let (dx, dy) = if i == 0 {
            (shape[1][0] - x, shape[1][1] - y2)
        } else if i + 1 == shape.len() {
            (x - shape[i - 1][0], y2 - shape[i - 1][1])
        } else {
            (shape[i + 1][0] - shape[i - 1][0], shape[i + 1][1] - shape[i - 1][1])
        };
        let len = (dx * dx + dy * dy).sqrt().max(1e-4);
        let nx = -dy / len;
        let ny = dx / len;
        pts.push([x + nx * offset, y2 + ny * offset]);
    }
    pts
}

pub fn trim_shape_ends(shape: &[[f32; 2]], trim: f32) -> Vec<[f32; 2]> {
    let mut out: Vec<[f32; 2]> = shape.to_vec();
    if out.len() < 2 {
        return out;
    }
    let d0 = ((out[1][0] - out[0][0]).powi(2) + (out[1][1] - out[0][1]).powi(2)).sqrt().max(1.0);
    if d0 > trim + 1.0 {
        out[0] = [
            out[0][0] + (out[1][0] - out[0][0]) / d0 * trim,
            out[0][1] + (out[1][1] - out[0][1]) / d0 * trim,
        ];
    }
    let n = out.len();
    let d1 = ((out[n - 1][0] - out[n - 2][0]).powi(2) + (out[n - 1][1] - out[n - 2][1]).powi(2))
        .sqrt()
        .max(1.0);
    if d1 > trim + 1.0 {
        out[n - 1] = [
            out[n - 1][0] - (out[n - 1][0] - out[n - 2][0]) / d1 * trim,
            out[n - 1][1] - (out[n - 1][1] - out[n - 2][1]) / d1 * trim,
        ];
    }
    out
}

pub fn sample_at(shape: &[[f32; 2]], t: f32) -> [f32; 2] {
    let dists = polyline_dists(shape);
    sample_polyline(shape, &dists, t)
}

pub fn total_len(shape: &[[f32; 2]]) -> f32 {
    *polyline_dists(shape).last().unwrap_or(&0.0)
}
