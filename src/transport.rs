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

        // 1. Authentication & ECDH Phase
        let nonce = generate_nonce();
        let (my_secret, my_public) = crate::crypto::generate_ecdh_keys();

        let challenge = SwarmMessage::AuthChallenge {
            nonce,
            public_key: my_public,
        };

        if let Ok(bin) = bincode::serialize(&challenge) {
            if sender.send(Message::Binary(bin.into())).await.is_err() {
                return;
            }
        }

        let mut authenticated = false;
        let mut session_key: Option<[u8; 32]> = None;

        // Wait for AuthResponse
        if let Some(Ok(Message::Binary(bin))) = receiver.next().await {
            if let Ok(SwarmMessage::AuthResponse {
                node_id: id,
                token_hash,
                public_key: their_public,
            }) = bincode::deserialize::<SwarmMessage>(&bin)
            {
                if verify_hmac(&nonce, &state.shared_secret, &token_hash) {
                    authenticated = true;
                    node_id = Some(id);
                    // Derive session key for PFS
                    session_key = Some(crate::crypto::derive_session_key(
                        &my_secret,
                        &their_public,
                        &nonce,
                    ));
                }
            }
        }

        if !authenticated || session_key.is_none() {
            println!("🔒 [Server] Authentication failed. Dropping connection.");
            return;
        }

        let session_key = session_key.unwrap();

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
                            // Ensure node_id matches the authenticated one
                            if Some(&device_id) != node_id.as_ref() {
                                println!("🔒 [Server] Node ID mismatch. Dropping.");
                                return;
                            }
                            println!(
                                "📡 [Server] Node {} connected and authenticated!",
                                device_id
                            );

                            // Get join response from orchestrator
                            let mut resp = state
                                .orchestrator
                                .handle_announce(device_id, profile, initial_status)
                                .ok();

                            // Encrypt the cluster key for this node using the session key
                            if let Some(SwarmMessage::JoinResponse {
                                ref mut encrypted_cluster_key,
                                ..
                            }) = resp
                            {
                                if let Ok(enc) = crate::crypto::encrypt_with_aad(
                                    &state.orchestrator.cluster_key,
                                    &session_key,
                                    b"cluster-key-handshake",
                                ) {
                                    *encrypted_cluster_key = enc;
                                }
                            }
                            resp
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

    /// Connects to a NeuralSwarmAI cluster. Returns the decrypted ClusterKey and the WebSocket stream.
    pub async fn connect_to_cluster(
        url: &str,
        shared_secret: &str,
        profile: NodeProfile,
        mut status: NodeStatus,
    ) -> Result<(
        [u8; 32],
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    )> {
        let start_connect = std::time::Instant::now();
        let (ws_stream, _) = connect_async(url).await?;
        let connection_latency = start_connect.elapsed().as_millis() as u32;
        status.latency_ms = Some(connection_latency);

        println!("🔗 Connected to swarm at {} (latency: {}ms)", url, connection_latency);

        let (mut write, mut read) = ws_stream.split();

        let mut cluster_key: Option<[u8; 32]> = None;

        // 1. Wait for AuthChallenge & ECDH
        if let Some(Ok(Message::Binary(bin))) = read.next().await {
            if let Ok(SwarmMessage::AuthChallenge {
                nonce,
                public_key: master_public,
            }) = bincode::deserialize::<SwarmMessage>(&bin)
            {
                // Generate our keys
                let (my_secret, my_public) = crate::crypto::generate_ecdh_keys();
                let session_key =
                    crate::crypto::derive_session_key(&my_secret, &master_public, &nonce);

                // Respond with HMAC + our Public Key
                let token_hash = sign_hmac(&nonce, shared_secret);
                let auth_resp = SwarmMessage::AuthResponse {
                    node_id: profile.hostname.clone(),
                    token_hash,
                    public_key: my_public,
                };
                let auth_bin = bincode::serialize(&auth_resp)?;
                write.send(Message::Binary(auth_bin.into())).await?;

                // 2. Send NodeAnnounce
                let announce = SwarmMessage::NodeAnnounce {
                    device_id: profile.hostname.clone(),
                    profile,
                    initial_status: status,
                };
                let bin = bincode::serialize(&announce)?;
                write.send(Message::Binary(bin.into())).await?;

                // 3. Wait for JoinResponse & Decrypt ClusterKey
                if let Some(Ok(Message::Binary(bin))) = read.next().await {
                    if let Ok(SwarmMessage::JoinResponse {
                        assigned_layers,
                        total_layers,
                        encrypted_cluster_key,
                    }) = bincode::deserialize::<SwarmMessage>(&bin)
                    {
                        // Decrypt cluster key
                        let dec = crate::crypto::decrypt_with_aad(
                            &encrypted_cluster_key,
                            &session_key,
                            b"cluster-key-handshake",
                        )
                        .map_err(|e| anyhow::anyhow!("Failed to decrypt cluster key: {}", e))?;
                        let mut key = [0u8; 32];
                        key.copy_from_slice(&dec);
                        cluster_key = Some(key);

                        println!(
                            "🎉 Handshake success! Assigned {} / {} layers: {:?}",
                            assigned_layers.len(),
                            total_layers,
                            assigned_layers
                        );
                    }
                }
            } else {
                anyhow::bail!("Expected AuthChallenge from server");
            }
        } else {
            anyhow::bail!("Connection closed before AuthChallenge");
        }

        if let Some(key) = cluster_key {
            // Reconstruct the stream from the split parts
            let ws_stream = write
                .reunite(read)
                .map_err(|e| anyhow::anyhow!("Failed to reunite stream: {}", e))?;
            Ok((key, ws_stream))
        } else {
            anyhow::bail!("Failed to obtain cluster key");
        }
    }
}
