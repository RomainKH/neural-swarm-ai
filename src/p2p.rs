use libp2p::{
    identify,
    identity, noise, ping,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux, Swarm,
};
use std::error::Error;
use std::time::Duration;

#[derive(libp2p::swarm::NetworkBehaviour)]
pub struct SwarmBehaviour {
    pub ping: ping::Behaviour,
    pub identify: identify::Behaviour,
}

pub async fn setup_p2p_node() -> Result<Swarm<SwarmBehaviour>, Box<dyn Error>> {
    let local_key = identity::Keypair::generate_ed25519();
    let local_peer_id = libp2p::PeerId::from(local_key.public());
    println!("🔑 Local Peer ID: {}", local_peer_id);

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

            SwarmBehaviour { ping, identify }
        })?
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // Listen on all interfaces, random port
    swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

    Ok(swarm)
}
