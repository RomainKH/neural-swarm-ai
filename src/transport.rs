#[cfg(feature = "server")]
pub mod server {
    use crate::crypto::{generate_nonce, verify_hmac};
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

    /// State required for the WebSocket server handler.
    #[derive(Clone)]
    pub struct ServerState {
        pub orchestrator: Arc<Orchestrator>,
        pub shared_secret: String,
    }

    /// Axum handler for NeuralSwarmAI WebSocket connections.
    pub async fn swarm_handler(
        ws: WebSocketUpgrade,
        State(state): State<ServerState>,
    ) -> impl IntoResponse {
        ws.on_upgrade(move |socket| handle_socket(socket, state))
    }

    async fn handle_socket(socket: WebSocket, state: ServerState) {
        let (mut sender, mut receiver) = socket.split();
        let mut node_id: Option<String> = None;

        // 1. Authentication Phase
        let nonce = generate_nonce();
        let challenge = SwarmMessage::AuthChallenge { nonce };
        if let Ok(bin) = bincode::serialize(&challenge) {
            if sender.send(Message::Binary(bin.into())).await.is_err() {
                return;
            }
        }

        let mut authenticated = false;

        // Wait for AuthResponse
        if let Some(Ok(Message::Binary(bin))) = receiver.next().await {
            if let Ok(SwarmMessage::AuthResponse { token_hash }) =
                bincode::deserialize::<SwarmMessage>(&bin)
            {
                if verify_hmac(&nonce, &state.shared_secret, &token_hash) {
                    authenticated = true;
                }
            }
        }

        if !authenticated {
            println!("🔒 [Server] Authentication failed. Dropping connection.");
            return;
        }

        // 2. Main Event Loop
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
                            println!(
                                "📡 [Server] Node {} connected and authenticated!",
                                device_id
                            );
                            state
                                .orchestrator
                                .handle_announce(device_id, profile, initial_status)
                                .ok()
                        }
                        SwarmMessage::StatusUpdate { status } => {
                            if let Some(ref id) = node_id {
                                state
                                    .orchestrator
                                    .handle_status_update(id, status)
                                    .ok()
                                    .flatten()
                            } else {
                                None
                            }
                        }
                        SwarmMessage::Heartbeat => {
                            if let Some(ref id) = node_id {
                                let _ = state.orchestrator.handle_heartbeat(id);
                            }
                            None
                        }
                        SwarmMessage::DrainRequest { .. } => {
                            if let Some(ref id) = node_id {
                                let _ = state.orchestrator.handle_drain(id);
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
            let _ = state.orchestrator.handle_drain(&id);
        }
    }
}

#[cfg(feature = "client")]
pub mod client {
    use crate::compute::{NodeProfile, NodeStatus};
    use crate::crypto::sign_hmac;
    use crate::protocol::SwarmMessage;
    use anyhow::Result;
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};

    /// Connects to a NeuralSwarmAI cluster.
    pub async fn connect_to_cluster(
        url: &str,
        shared_secret: &str,
        profile: NodeProfile,
        status: NodeStatus,
    ) -> Result<()> {
        let (ws_stream, _) = connect_async(url).await?;
        println!("🔗 Connected to swarm at {}", url);

        let (mut write, mut read) = ws_stream.split();

        // 1. Wait for AuthChallenge
        if let Some(Ok(Message::Binary(bin))) = read.next().await {
            if let Ok(SwarmMessage::AuthChallenge { nonce }) =
                bincode::deserialize::<SwarmMessage>(&bin)
            {
                // Respond with HMAC
                let token_hash = sign_hmac(&nonce, shared_secret);
                let auth_resp = SwarmMessage::AuthResponse { token_hash };
                let auth_bin = bincode::serialize(&auth_resp)?;
                write.send(Message::Binary(auth_bin.into())).await?;
            } else {
                anyhow::bail!("Expected AuthChallenge from server");
            }
        } else {
            anyhow::bail!("Connection closed before AuthChallenge");
        }

        // 2. Send NodeAnnounce
        let announce = SwarmMessage::NodeAnnounce {
            device_id: profile.hostname.clone(),
            profile,
            initial_status: status,
        };
        let bin = bincode::serialize(&announce)?;
        write.send(Message::Binary(bin.into())).await?;

        // 3. Wait for JoinResponse
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

        // 4. Keep connection open for the PoC
        println!("⏳ Waiting for tasks...");
        while read.next().await.is_some() {}

        Ok(())
    }
}
