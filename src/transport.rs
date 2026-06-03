#[cfg(feature = "server")]
pub mod server {
    use crate::protocol::SwarmMessage;
    use crate::Orchestrator;
    use axum::{
        extract::{
            ws::{Message, WebSocket, WebSocketUpgrade},
            State,
        },
        response::IntoResponse,
    };
    use futures::{sink::SinkExt, stream::StreamExt};
    use std::sync::Arc;

    /// Axum handler for NeuralSwarmAI WebSocket connections.
    pub async fn swarm_handler(
        ws: WebSocketUpgrade,
        State(orchestrator): State<Arc<Orchestrator>>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| handle_socket(socket, orchestrator))
    }

    async fn handle_socket(socket: WebSocket, orchestrator: Arc<Orchestrator>) {
        let (mut sender, mut receiver) = socket.split();
        let mut node_id: Option<String> = None;

        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Binary(bin) = msg {
                if let Ok(swarm_msg) = bincode::deserialize::<SwarmMessage>(&bin) {
                    let response = match swarm_msg {
                        SwarmMessage::NodeAnnounce {
                            device_id,
                            profile,
                            initial_status,
                        } => {
                            node_id = Some(device_id.clone());
                            println!("📡 [Server] Node {} connected!", device_id);
                            orchestrator
                                .handle_announce(device_id, profile, initial_status)
                                .ok()
                        }
                        SwarmMessage::StatusUpdate { status } => {
                            if let Some(ref id) = node_id {
                                orchestrator.handle_status_update(id, status).ok().flatten()
                            } else {
                                None
                            }
                        }
                        SwarmMessage::Heartbeat => {
                            if let Some(ref id) = node_id {
                                let _ = orchestrator.handle_heartbeat(id);
                            }
                            None
                        }
                        SwarmMessage::DrainRequest { .. } => {
                            if let Some(ref id) = node_id {
                                let _ = orchestrator.handle_drain(id);
                            }
                            None
                        }
                        // Tasks are handled dynamically, not implemented in this mock loop
                        _ => None,
                    };

                    // Send response if any (e.g., JoinResponse or RebalanceCommand)
                    if let Some(resp) = response {
                        if let Ok(resp_bin) = bincode::serialize(&resp) {
                            let _ = sender.send(Message::Binary(resp_bin.into())).await;
                        }
                    }
                }
            }
        }

        // Socket closed
        if let Some(id) = node_id {
            println!("📡 [Server] Node {} disconnected.", id);
            let _ = orchestrator.handle_drain(&id);
        }
    }
}

#[cfg(feature = "client")]
pub mod client {
    use crate::compute::{NodeProfile, NodeStatus};
    use crate::protocol::SwarmMessage;
    use anyhow::Result;
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

    /// Connects to a NeuralSwarmAI cluster.
    pub async fn connect_to_cluster(
        url: &str,
        profile: NodeProfile,
        status: NodeStatus,
    ) -> Result<()> {
        let (ws_stream, _) = connect_async(url).await?;
        println!("🔗 Connected to swarm at {}", url);

        let (mut write, mut read) = ws_stream.split();

        // 1. Send NodeAnnounce
        let announce = SwarmMessage::NodeAnnounce {
            device_id: profile.hostname.clone(),
            profile,
            initial_status: status,
        };
        let bin = bincode::serialize(&announce)?;
        write.send(Message::Binary(bin.into())).await?;

        // 2. Wait for JoinResponse
        if let Some(Ok(Message::Binary(bin))) = read.next().await {
            if let Ok(SwarmMessage::JoinResponse {
                assigned_layers,
                total_layers,
            }) = bincode::deserialize::<SwarmMessage>(&bin)
            {
                println!(
                    "🎉 Handshake success! Assigned {} / {} layers: {:?}",
                    assigned_layers.len(),
                    total_layers,
                    assigned_layers
                );
            }
        }

        // 3. Keep connection open for the PoC
        println!("⏳ Waiting for tasks...");
        while read.next().await.is_some() {}

        Ok(())
    }
}
