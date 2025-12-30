# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## CLI Usage (Non-Obvious)

- **List targets**: `cargo run -p neuro-compiler-cli -- list-targets`
- **Lower with pipeline**: `cargo run -p neuro-compiler-cli -- lower --pipeline partition,placement,routing --dump-dir target/dumps --dump-format json`
- **Compile to target**: `cargo run -p neuro-compiler-cli -- compile --input model.json --target riscv64gcv_linux`
- **Simulate with features**: `cargo run -p neuro-compiler-cli --features sim-neuron -- simulate --simulator neuron --input model.json`

## Code Style (Non-Obvious)

- **Clippy config**: `allow-private-module-inception = true` (allows nested private modules)
- **Rustfmt config**: `group_imports = "StdExternalCrate"` and `imports_granularity = "Module"` (non-standard import grouping)

## Target Configuration (Non-Obvious)

- **Manifest location**: Target manifests are in `targets/<name>.toml`
- **Manifest attachment**: Graph attributes must include `"hal_manifest_path"` pointing to the manifest file

## Pass Pipeline (Non-Obvious)

- **Available passes**: `noop`, `validate`, `quantize4`, `quantize8`, `quantize16`, `partition`, `placement`, `routing`, `timing`, `resource-check`, `tn-core-mapping`, `tn-weight-programming`, `tn-crossbar-config`, `sn-core-allocation`, `sn-aer-routing`, `sn-synapse-programming`, `loihi-core-mapping`, `loihi-synapse-programming`, `loihi-learning-rule`, `akida-layer-mapping`, `akida-weight-programming`, `akida-event-routing`
- **Pipeline format**: Comma-separated string (e.g., `"partition,placement,routing"`)
- **Dump formats**: `json`, `yaml`, `bin` (bin requires `--features bin-artifacts`)

## Features (Non-Obvious)

- **Backend features**: `backend-loihi`, `backend-riscv`, `backend-spinnaker`, etc.
- **Simulator features**: `sim-neuron`, `sim-arbor`, `sim-hw-specific`, etc.
- **Telemetry**: `--features telemetry` enables profiling and metrics
- **MLIR**: `--features mlir` enables MLIR bridge

## File Formats (Non-Obvious)

- **NIR format detection**: Based on file extension (.json = JSON, .yaml/.yml = YAML)
- **Graph versioning**: Must call `graph.ensure_version_tag()` before compilation/simulation
- **Serialization**: Supports JSON, YAML, and binary (feature-gated)