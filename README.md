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
- [🏗️ Architecture](#️-architecture)
- [🛡️ Security](#️-security)
- [🤝 Contributing](#-contributing)

## ✨ Key Features

- **Pipeline Parallelism**: Distribute LLM layers across multiple nodes.
- **Dynamic Orchestration**: Automatically assign layers based on node compute power.
- **Zero-Copy Optimization**: Uses `Bytes` for efficient memory management during network transfers.
- **Security-First**: Native support for mTLS and encrypted tensor transfers (AES-256-GCM).
- **Heterogeneous Support**: Seamlessly mix CPU (ARM/x86) and GPU (Metal/CUDA) nodes.

## 📦 Installation

To use NeuralSwarmAI as a dependency in your Rust project, add it to your `Cargo.toml`.

### From Crates.io (If published)
```toml
[dependencies]
neural-swarm-ai = "0.1.0"
```

### From a local path (for development)
If you are developing your project alongside `neural-swarm-ai`:
```toml
[dependencies]
neural-swarm-ai = { path = "../neural-swarm-ai" }
```

### From GitHub
To always use the latest commit from the main branch:
```toml
[dependencies]
neural-swarm-ai = { git = "https://github.com/RomainKH/neural-swarm-ai.git", branch = "main" }
```

Alternatively, you can install it directly via the command line using Cargo:
```bash
cargo add --git https://github.com/RomainKH/neural-swarm-ai.git
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
2. Build the library:
   ```bash
   cargo build --release
   ```
3. Run the tests:
   ```bash
   cargo test
   ```

## 🚀 Quick Start

Here is a minimal example of how to use the library to create a Master orchestrator and a Worker node.

### 1. Orchestration (Master)

The Master node manages the cluster and distributes computation tasks.

```rust
use neural_swarm_ai::Orchestrator;

fn main() {
    // Initialize an orchestrator for a 32-layer model
    let mut orchestrator = Orchestrator::new(32); 
    
    // Example: A node joins the swarm
    let response = orchestrator.handle_join("macbook-pro".into(), 100);
    println!("Node joined with response: {:?}", response);
}
```

### 2. Define a Compute Node (Worker)

The worker processes the layers assigned to it and forwards the results.

```rust
use neural_swarm_ai::{Executor, SwarmMessage};

fn main() {
    let executor = Executor::new("node-1".into());

    // Pseudo-code for handling a task
    // On receiving a ProcessTask message:
    /*
    unsafe {
        if let Some(result) = executor.run_task(&mut llama_ctx, task_message) {
            // Send TaskResult back to Master or forward to the next node
        }
    }
    */
}
```

## 🏗️ Architecture

NeuralSwarmAI implements a "Pause-and-Forward" mechanism:
1. **Master** starts inference on the first $N$ layers.
2. **State** (KV Cache) is serialized and forwarded to the next **Worker**.
3. **Worker** injects state, computes next $M$ layers, and forwards the new state.
4. **Final Node** returns the logits for token sampling.

## 🛡️ Security

Data privacy is guaranteed by:
- **Local Execution**: No cloud dependency. Your data never leaves your swarm.
- **End-to-End Encryption**: All computation state in transit is encrypted.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve NeuralSwarmAI, please open an issue or submit a pull request. 
Make sure to format your code with `cargo fmt` and check for lints with `cargo clippy`.

---
*Created by Romain Khanoyan.*
