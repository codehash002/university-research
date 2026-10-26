//! Shared TraCI bridge JSON protocol.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Bounds {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Lane {
    pub id: String,
    pub edge: String,
    #[serde(default)]
    pub index: i32,
    pub width: f32,
    pub shape: Vec<[f32; 2]>,
    pub kind: String,
    #[serde(default)]
    pub internal: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Junction {
    pub id: String,
    pub x: f32,
    pub y: f32,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub shape: Vec<[f32; 2]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TlsHead {
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub facing: f32,
    #[serde(default)]
    pub travel: f32,
    #[serde(default)]
    pub links: Vec<usize>,
    #[serde(default)]
    pub edge: String,
    #[serde(default)]
    pub lane_index: i32,
    /// "L" | "S" | "R" for gantry head role.
    #[serde(default)]
    pub turn: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Tls {
    pub id: String,
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub heads: Vec<TlsHead>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Network {
    pub bounds: Bounds,
    pub lanes: Vec<Lane>,
    #[serde(default)]
    pub junctions: Vec<Junction>,
    #[serde(default)]
    pub tls: Vec<Tls>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Vehicle {
    pub id: String,
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    #[serde(default)]
    pub speed: f32,
    #[serde(default)]
    pub length: f32,
    #[serde(default)]
    pub width: f32,
    #[serde(default)]
    #[serde(rename = "type")]
    pub vtype: String,
    /// TraCI getSignals bitmask (blinker L/R, brake, …).
    #[serde(default)]
    pub signals: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Person {
    pub id: String,
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub angle: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LightState {
    pub id: String,
    pub state: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    #[serde(rename = "network")]
    Network { network: Network },
    #[serde(rename = "frame")]
    Frame {
        t: f32,
        vehicles: Vec<Vehicle>,
        #[serde(default)]
        persons: Vec<Person>,
        #[serde(default)]
        lights: Vec<LightState>,
        #[serde(default)]
        done: bool,
    },
    #[serde(rename = "status")]
    Status {
        #[serde(default)]
        paused: bool,
        #[serde(default)]
        speed: f32,
        #[serde(default)]
        done: bool,
        #[serde(default)]
        gui: bool,
        #[serde(default)]
        controller: String,
    },
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        message: String,
    },
}

/// TraCI signal bits
pub const SIG_BLINKER_RIGHT: u32 = 1;
pub const SIG_BLINKER_LEFT: u32 = 2;
pub const SIG_BRAKELIGHT: u32 = 8;
