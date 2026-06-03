#[cfg(feature = "server")]
pub mod server {
    use axum::{
        extract::{ws::{Message, WebSocket, WebSocketUpgrade}, State},
        response::IntoResponse,
    };
    use futures::{sink::SinkExt, stream::StreamExt};
    use std::sync::Arc;
    use crate::protocol::SwarmMessage;

    /// Axum handler for NeuralSwarmAI WebSocket connections.
    pub async fn swarm_handler<S>(
        ws: WebSocketUpgrade,
        State(state): State<Arc<S>>,
        handle_msg: fn(SwarmMessage, Arc<S>) -> Option<SwarmMessage>,
    ) -> impl IntoResponse 
    where S: Send + Sync + 'static
    {
        ws.on_upgrade(move |socket| handle_socket(socket, state, handle_msg))
    }

    async fn handle_socket<S>(socket: WebSocket, state: Arc<S>, handle_msg: fn(SwarmMessage, Arc<S>) -> Option<SwarmMessage>) 
    where S: Send + Sync + 'static
    {
        let (mut sender, mut receiver) = socket.split();

        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Binary(bin) = msg {
                if let Ok(swarm_msg) = serde_json::from_slice::<SwarmMessage>(&bin) {
                    if let Some(response) = handle_msg(swarm_msg, Arc::clone(&state)) {
                        if let Ok(resp_bin) = serde_json::to_vec(&response) {
                            let _ = sender.send(Message::Binary(resp_bin.into())).await;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(feature = "client")]
pub mod client {
    use tokio_tungstenite::connect_async;
    use futures::stream::StreamExt;
    use anyhow::Result;

    /// Connects to a NeuralSwarmAI cluster.
    pub async fn connect_to_cluster(url: &str) -> Result<()> {
        let (ws_stream, _) = connect_async(url).await?;
        let (mut _write, mut _read) = ws_stream.split();
        
        // Protocol logic for client goes here.
        Ok(())
    }
}
