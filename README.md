# 🧠 NeuralSwarmAI

![Rust](https://img.shields.io/badge/rust-1.75%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

NeuralSwarmAI is a high-performance, lightweight Rust library for distributed Large Language Model (LLM) inference using **Pipeline Parallelism**.

It enables running massive models (e.g., 70B+ parameters) on a network of consumer-grade devices (Raspberry Pis, Smartphones, PCs) by splitting the model layers across the swarm.

## 📑 Table of Contents

- [✨ Key Features](#-key-features)
- [📦 Installation](#-installation)
- [🛠️ Development & Local Setup](#️-development--local-setup)
- [🚀 Quick Start](#-quick-start)
- [🔌 Custom Backend](#-custom-backend)
- [🏗️ Architecture](#️-architecture)
- [🛡️ Security](#️-security)
- [🤝 Contributing](#-contributing)

## ✨ Key Features

- **Pipeline Parallelism**: Distribute LLM layers across multiple nodes.
- **Dynamic Orchestration**: Real-time resource monitoring and automatic workload rebalancing based on current usage.
- **Pure Rust ML (v0.3)**: Built-in support for `candle` (HuggingFace) for 100% Rust, portable, and fast inference.
- **Topology-Aware Routing (v0.3)**: Pipeline organization optimized by network latency (Ping-aware) to minimize inter-node delay.
- **Pipeline Overlap (v0.2)**: Asynchronous network I/O allows receiving the next layer's KV Cache while the GPU is still computing the current one.
- **Security-First (v0.2)**: 
  - **Perfect Forward Secrecy**: ECDH (X25519) key exchange for every session.
  - **Authenticated Encryption**: AES-256-GCM with AAD (Task-binding) to prevent replay attacks.
- **Compression (v0.2)**: High-speed `zstd` compression of KV Cache tensors to reduce bandwidth usage by up to 60%.
- **Heterogeneous Support**: Seamlessly mix CPU (ARM/x86) and GPU (Metal/CUDA/WGPU) nodes.

## ⚡ Performance & Specs

NeuralSwarmAI is engineered for maximum throughput in unstable P2P environments.

### Optimized for Latency
| Feature | Impact | Technology |
|---------|--------|------------|
| **Pipeline Overlap** | -30% Latency | Async MPSC Channels |
| **KV Compression** | -60% Bandwidth | Zstd (Level 3) |
| **Topology Routing** | Minimized Jitter | Latency-aware sorting |
| **Zero-Copy** | 0ms Memcpy | `Bytes` reference counting |

### Resource Management
The orchestrator calculates a **Composite Capacity Score** for each node:
$$\text{Capacity} = (\text{Cores} \times \text{Clock}) + \frac{\text{Free RAM}}{\text{Model Size}} - \text{Latency Penalty}$$
This ensures that a high-latency node doesn't become the bottleneck of the entire cluster.

## 📦 Installation

To use NeuralSwarmAI as a dependency in your Rust project, add it to your `Cargo.toml`.

### Basic (no ML runtime)

The core library compiles in pure Rust with no C/C++ dependencies:

```toml
[dependencies]
neural-swarm-ai = "0.1.0"
```

### With Candle backend (Recommended for v0.3+)

For 100% Rust and easy cross-platform support:

```toml
[dependencies]
neural-swarm-ai = { version = "0.1.0", features = ["candle"] }
```

### With llama.cpp backend

Still supported via bindings:

```toml
[dependencies]
neural-swarm-ai = { version = "0.1.0", features = ["llama"] }
```

### From GitHub

To always use the latest commit from the main branch:

```toml
[dependencies]
neural-swarm-ai = { git = "https://github.com/RomainKH/neural-swarm-ai.git", branch = "main" }
```

## 🛠️ Development & Local Setup

Want to build or contribute to NeuralSwarmAI? Here is how to set up your environment.

### Prerequisites
- **Rust**: Ensure you have Rust installed. The easiest way is via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### Build from source
1. Clone the repository:
   ```bash
   git clone https://github.com/RomainKH/neural-swarm-ai.git
   cd neural-swarm-ai
   ```
2. Build the library (fast, pure Rust):
   ```bash
   cargo build --release
   ```
3. Build with llama.cpp backend (requires cmake):
   ```bash
   cargo build --release --features llama
   ```
4. Run the tests:
   ```bash
   cargo test
   ```

## 🚀 Quick Start

Here is a minimal example of how to use the library to create a Master orchestrator and a Worker node.

### 1. Orchestration (Master)

The Master node manages the cluster and distributes computation tasks proportionally to each node's compute power.

```rust
use neural_swarm_ai::Orchestrator;
use neural_swarm_ai::compute::{NodeProfile, DeviceType, NodeStatus};

fn main() {
    // Initialize an orchestrator for a 32-layer model with a shared secret
    let orchestrator = Orchestrator::new(32, "my-shared-secret".into());

    // Nodes announce themselves to the swarm
    let profile = NodeProfile::custom(DeviceType::Desktop, 8, 16384, "gpu-node".into());
    let status = NodeStatus::unknown();

    let resp = orchestrator.handle_announce("gpu-node".into(), profile, status).unwrap();
    println!("Node assigned layers: {:?}", resp);
}
```

### 2. Define a Compute Node (Worker)

The worker processes the layers assigned to it using any inference backend.

```rust
use neural_swarm_ai::Executor;
use neural_swarm_ai::compute::{ComputeMonitor, NodeProfile};

#[tokio::main]
async fn main() {
    // Auto-detect the hardware profile
    let profile = NodeProfile::detect();
    
    // Cluster key is obtained during handshake (see examples/worker.rs)
    let cluster_key = [0u8; 32]; 
    let executor = Executor::new(profile.hostname.clone(), cluster_key);

    // Start background resource monitoring
    let (monitor, _status_rx) = ComputeMonitor::new(Default::default());
    tokio::spawn(monitor.run());

    // Use any backend implementing InferenceBackend:
    // let result = executor.run_task(&mut my_backend, task_message).unwrap();
}
```

### 3. Using the Candle backend (v0.3+)

NeuralSwarmAI provides a built-in Candle backend for 100% Rust inference:

```rust
use neural_swarm_ai::{Executor, CandleBackend};

fn main() {
    // Load your model via Candle...
    // let mut backend = CandleBackend::new(model, device);
}
```

### 4. Using the llama.cpp backend

Still available for GGUF models:

```rust
use neural_swarm_ai::{Executor, LlamaBackend};

fn main() {
    // Wrap an existing LlamaContext:
    // let mut backend = LlamaBackend::new(&mut llama_ctx);
}
```

## 🔌 Custom Backend

NeuralSwarmAI is designed to work with **any** inference engine. Implement the `InferenceBackend` trait to plug in your own:

```rust
use neural_swarm_ai::InferenceBackend;
use anyhow::Result;

struct MyCustomBackend {
    // Your model state...
}

impl InferenceBackend for MyCustomBackend {
    fn set_state(&mut self, state: &[u8]) -> Result<()> {
        // Restore KV cache from serialized bytes
        Ok(())
    }

    fn get_state(&self) -> Result<Vec<u8>> {
        // Serialize current KV cache
        Ok(vec![])
    }

    fn run_layers(&mut self, start: u32, end: u32, tokens: &[i32]) -> Result<Vec<f32>> {
        // Run inference on layers [start, end) and return logits
        Ok(vec![])
    }
}
```

This enables integrations with frameworks like [candle](https://github.com/huggingface/candle), [burn](https://github.com/tracel-ai/burn), or any custom GGUF/ONNX runtime.

## 🏗️ Architecture

NeuralSwarmAI implements a "Pause-and-Forward" mechanism:
1. **Master** starts inference on the first $N$ layers.
2. **State** (KV Cache) is serialized and forwarded to the next **Worker**.
3. **Worker** injects state, computes next $M$ layers, and forwards the new state.
4. **Final Node** returns the logits for token sampling.

### Feature Flags

| Feature  | Default | Description                            |
|----------|---------|----------------------------------------|
| `server` | ✅      | Axum WebSocket server for the Master   |
| `client` | ✅      | WebSocket client for Worker nodes      |
| `candle` | ❌      | Pure Rust inference backend (Candle)   |
| `llama`  | ❌      | llama.cpp inference backend (Bindings) |
| `metal`  | ❌      | Apple Silicon GPU acceleration         |
| `cuda`   | ❌      | Nvidia GPU acceleration                |

## 🛡️ Security

Data privacy is guaranteed by:
- **Local Execution**: No cloud dependency. Your data never leaves your swarm.
- **End-to-End Encryption**: All computation state in transit is encrypted.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve NeuralSwarmAI, please open an issue or submit a pull request. 
Make sure to format your code with `cargo fmt` and check for lints with `cargo clippy`.

---
*Created by Romain Khanoyan.*
