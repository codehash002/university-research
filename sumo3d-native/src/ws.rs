//! Background WebSocket client → channel of ServerMsg.

use crate::protocol::ServerMsg;
use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use std::sync::mpsc::{self, Receiver, Sender};
use tokio_tungstenite::{connect_async, tungstenite::Message};

pub enum ClientCmd {
    Pause,
    Play,
    Toggle,
    Restart,
    Speed(f32),
    /// route index, gui flag, model path (empty = builtin)
    Configure {
        route: u32,
        gui: bool,
        model: String,
    },
}

pub struct BridgeChannels {
    pub rx: std::sync::Mutex<Receiver<ServerMsg>>,
    pub cmd_tx: Sender<ClientCmd>,
}

pub fn spawn_bridge(url: String) -> BridgeChannels {
    let (tx, rx) = mpsc::channel::<ServerMsg>();
    let (cmd_tx, cmd_rx) = mpsc::channel::<ClientCmd>();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        rt.block_on(async move {
            let mut refused_logged = false;
            loop {
                match run_session(&url, &tx, &cmd_rx).await {
                    Ok(()) => {
                        refused_logged = false;
                        eprintln!("[ws] disconnected — retrying…");
                    }
                    Err(e) => {
                        let msg = format!("{e:#}");
                        let refused = msg.contains("10061")
                            || msg.contains("Connection refused")
                            || msg.contains("actively refused");
                        if refused {
                            if !refused_logged {
                                eprintln!(
                                    "[ws] waiting for bridge on {url} (click Start in the hub)"
                                );
                                refused_logged = true;
                            }
                        } else {
                            eprintln!("[ws] session ended: {msg}");
                            refused_logged = false;
                        }
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        });
    });

    BridgeChannels {
        rx: std::sync::Mutex::new(rx),
        cmd_tx,
    }
}

async fn run_session(
    url: &str,
    tx: &Sender<ServerMsg>,
    cmd_rx: &Receiver<ClientCmd>,
) -> Result<()> {
    let (ws, _) = connect_async(url).await?;
    eprintln!("[ws] connected {url}");
    let (mut write, mut read) = ws.split();

    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        match serde_json::from_str::<ServerMsg>(&text) {
                            Ok(m) => { let _ = tx.send(m); }
                            Err(e) => eprintln!("[ws] parse: {e}"),
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(e)) => return Err(e.into()),
                    _ => {}
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(16)) => {
                while let Ok(cmd) = cmd_rx.try_recv() {
                    let payload = match cmd {
                        ClientCmd::Pause => r#"{"cmd":"pause"}"#.to_string(),
                        ClientCmd::Play => r#"{"cmd":"play"}"#.to_string(),
                        ClientCmd::Toggle => r#"{"cmd":"toggle"}"#.to_string(),
                        ClientCmd::Restart => r#"{"cmd":"restart"}"#.to_string(),
                        ClientCmd::Speed(v) => format!(r#"{{"cmd":"speed","value":{v}}}"#),
                        ClientCmd::Configure { route, gui, model } => {
                            format!(
                                r#"{{"cmd":"configure","route":{route},"gui":{gui},"model":{}}}"#,
                                serde_json::to_string(&model).unwrap_or_else(|_| "\"\"".into())
                            )
                        }
                    };
                    write.send(Message::Text(payload.into())).await?;
                }
            }
        }
    }
    Ok(())
}
