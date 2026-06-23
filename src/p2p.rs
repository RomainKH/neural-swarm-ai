use libp2p::{
    dcutr, identify, identity, noise, ping, relay,
    request_response::{self, ProtocolSupport},
    tcp, yamux, Multiaddr, PeerId, StreamProtocol, Swarm,
};
use std::error::Error;
use std::time::Duration;
use tokio::sync::mpsc;
use futures::StreamExt;
use crate::protocol::SwarmMessage;

#[derive(libp2p::swarm::NetworkBehaviour)]
pub struct SwarmBehaviour {
    pub ping: ping::Behaviour,
    pub identify: identify::Behaviour,
    pub relay_client: relay::client::Behaviour,
    pub dcutr: dcutr::Behaviour,
    pub req_resp: request_response::cbor::Behaviour<SwarmMessage, SwarmMessage>,
}

pub struct P2pNode {
    pub peer_id: PeerId,
    pub sender: mpsc::Sender<(PeerId, SwarmMessage)>,
    pub receiver: mpsc::Receiver<(PeerId, SwarmMessage)>,
}

pub async fn setup_p2p_node() -> Result<P2pNode, Box<dyn Error>> {
    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = PeerId::from(local_key.public());
    println!("🔑 Local Peer ID: {}", local_peer_id);

    let (_relay_transport, relay_client) = relay::client::new(local_peer_id);

    let mut swarm = libp2p::SwarmBuilder::with_existing_identity(local_key)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|key| {
            let identify = identify::Behaviour::new(identify::Config::new(
                "/neural-swarm/1.0.0".into(),
                key.public(),
            ));
            let ping = ping::Behaviour::new(ping::Config::new().with_interval(Duration::from_secs(15)));
            let dcutr = dcutr::Behaviour::new(key.public().to_peer_id());
            
            let req_resp = request_response::cbor::Behaviour::<SwarmMessage, SwarmMessage>::new(
                [(StreamProtocol::new("/neural-swarm/req-resp/1.0.0"), ProtocolSupport::Full)],
                request_response::Config::default().with_request_timeout(Duration::from_secs(30)),
            );

            SwarmBehaviour { ping, identify, relay_client, dcutr, req_resp }
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // Listen on all interfaces, random port
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    let (tx_out, mut rx_out) = mpsc::channel::<(PeerId, SwarmMessage)>(100);
    let (tx_in, rx_in) = mpsc::channel::<(PeerId, SwarmMessage)>(100);

    tokio::spawn(async move {
        loop {
            tokio::select! {
                event = swarm.select_next_some() => {
                    match event {
                        libp2p::swarm::SwarmEvent::Behaviour(SwarmBehaviourEvent::ReqResp(
                            request_response::Event::Message { peer, message }
                        )) => {
                            if let request_response::Message::Request { request, .. } = message {
                                let _ = tx_in.send((peer, request)).await;
                            }
                        }
                        // Ignore other events for now
                        _ => {}
                    }
                }
                Some((peer, msg)) = rx_out.recv() => {
                    swarm.behaviour_mut().req_resp.send_request(&peer, msg);
                }
            }
        }
    });

    Ok(P2pNode {
        peer_id: local_peer_id,
        sender: tx_out,
        receiver: rx_in,
    })
}
