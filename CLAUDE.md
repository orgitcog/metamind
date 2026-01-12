# CLAUDE.md - hypATen Project Guide

## Project Overview

**hypATen** is an open-source EMG (electromyography) signal processing platform that combines:
- **myopen**: Hardware and software for EMG acquisition and prosthetic control
- **hypermind**: Hyperparameter optimization for neural networks
- **ATen**: High-performance C++ tensor operations (PyTorch core)
- **Torch ecosystem**: Deep learning modules (nn, rnn, nngraph)

Licensed under GPL v3.

## Architecture

```
hypATen/
├── MiniVIE/                    # MATLAB framework for EMG processing
│   ├── +PatternRecognition/    # Classification algorithms
│   ├── +SignalAnalysis/        # Feature extraction
│   ├── +Inputs/                # Hardware interfaces (EMG, motion capture)
│   ├── +Controls/              # Prosthetic grasp control
│   └── +GUIs/                  # User interfaces
├── torch_integration/          # Deep learning components
│   ├── hypermind/              # Hyperparameter optimization
│   ├── aten/                   # C++ tensor library
│   ├── nn/                     # Neural network layers
│   ├── rnn/                    # LSTM/GRU modules
│   ├── nngraph/                # Graph-based networks
│   └── examples/               # EMG classifier examples
├── gtkclient_tdt/              # GTK client for TDT systems
└── Documentation/              # Research papers and designs
```

## Key Technologies

| Component | Language | Purpose |
|-----------|----------|---------|
| MiniVIE | MATLAB | Signal acquisition, feature extraction, real-time control |
| Torch Integration | Lua/C++ | Deep learning models, GPU acceleration |
| Hardware Interfaces | C/MATLAB | EMG hardware communication |

## Development Commands

```bash
# Setup Torch integration
cd torch_integration && ./setup.sh

# Run EMG classifier examples
th examples/emg_classifier.lua
th examples/emg_rnn_classifier.lua
th examples/multi_channel_fusion.lua
```

## Signal Processing Pipeline

1. **Acquisition**: EMG signals from hardware (1000 Hz sampling)
2. **Windowing**: 200-300ms windows with 50-100ms overlap
3. **Feature Extraction**: MAV, RMS, variance, frequency domain
4. **Classification**: Pattern recognition or deep learning
5. **Control Output**: Prosthetic grasp commands

## Integration Patterns

### MATLAB ↔ Torch Communication
- **File-based**: Save `.mat` files, load in Torch via `matio`
- **Socket-based**: UDP streaming for real-time (<10ms latency)
- **Hybrid**: MATLAB for acquisition, Torch for inference

### Feature Format Convention
```
[MAV_ch1, ..., MAV_chN, RMS_ch1, ..., RMS_chN, VAR_ch1, ..., VAR_chN]
```

## Important Guidelines

- Maintain feature extraction consistency between MATLAB and Torch
- Apply identical normalization (zero-mean, unit variance per channel)
- Use CUDA for GPU acceleration when available
- Buffer 10-20 windows for batch processing efficiency

---

## Hyperd Integration Evaluation

### What is hyperd?

[hyperd](https://github.com/o9nn/hyperd) (HyperContainer Daemon) is a hypervisor-agnostic container runtime that runs Docker images on plain hypervisors, combining VM-level isolation with container-like performance.

### Potential Benefits for hypATen

#### 1. Isolated Training Environments
| Benefit | Description |
|---------|-------------|
| **Strong isolation** | Hardware-enforced security boundaries prevent training workloads from affecting host systems |
| **Reproducibility** | Containerized Torch environments ensure consistent model training across machines |
| **Resource control** | Fine-grained CPU/memory allocation for GPU-intensive training jobs |

#### 2. Deployment Advantages
| Benefit | Description |
|---------|-------------|
| **Portable inference** | Deploy trained EMG classifiers as lightweight hypercontainers |
| **Fast startup** | Sub-second boot times suitable for real-time prosthetic control applications |
| **Hypervisor agnostic** | Run on QEMU, Xen, or other hypervisors without code changes |

#### 3. Multi-Tenant EMG Processing
| Benefit | Description |
|---------|-------------|
| **Patient data isolation** | VM-level isolation for HIPAA/medical device compliance |
| **Concurrent processing** | Run multiple EMG classification models in isolated containers |
| **Resource efficiency** | Lower overhead than full VMs while maintaining isolation |

### Integration Considerations

**Pros:**
- Stronger security model than Docker for medical/research data
- Existing Torch Docker images work unchanged
- Good fit for edge deployment on prosthetic control systems
- Kubernetes-compatible for cloud-based training infrastructure

**Cons:**
- Additional complexity vs. native Docker
- Requires hypervisor support (QEMU 2.0+ or Xen 4.5+)
- Linux kernel 3.8+ requirement limits some embedded platforms
- Maintenance of additional infrastructure component

### Recommended Integration Path

1. **Phase 1**: Containerize Torch training environment with Docker
2. **Phase 2**: Test hyperd for isolated inference on edge devices
3. **Phase 3**: Deploy hyperd for multi-tenant cloud training if needed

### Technical Requirements

```bash
# hyperd dependencies
- Linux kernel 3.8+
- QEMU 2.0+ (2.6+ for ARM64)
- Go 1.7+
- Device-mapper-devel
```

### Verdict

**Recommended for**: Production deployments requiring medical-grade isolation, multi-tenant cloud infrastructure, or edge devices where VM-level security is mandated.

**Not recommended for**: Development environments, single-user research setups, or platforms without hypervisor support.

---

## Quick Reference

| Task | Command/Location |
|------|------------------|
| Train EMG classifier | `th examples/emg_classifier.lua` |
| MATLAB feature extraction | `MiniVIE/+SignalAnalysis/` |
| Pattern recognition | `MiniVIE/+PatternRecognition/` |
| Hardware interfaces | `MiniVIE/+Inputs/` |
| Torch setup | `torch_integration/setup.sh` |
