# neuro-compiler

A universal neuromorphic compiler, written in Rust with optional Python bindings. It takes spiking neural network models and lowers them through a common intermediate representation to a wide range of neuromorphic hardware backends, simulators, and RISC-V targets.

## How it fits together

```
Frontends (PyNN, Nengo, NEST, Brian, ...)
        │
        ▼
     NIR  ──────────────►  Pass pipeline  ──────────────►  Backends / Simulators
(Neuromorphic IR)      (validate, quantize,          (Loihi2, TrueNorth, Akida,
                        partition, placement,          SpiNNaker2, RISC-V, ...
                        routing, timing, ...)           NEURON, Arbor, ...)
        ▲
        │
       HAL (targets/*.toml — per-chip capacity & timing manifests)
```

- **NIR** (`crates/nir`) — a serializable (JSON/YAML) graph format describing populations, connections, and probes, independent of any target chip.
- **HAL** (`crates/hal`) — loads TOML target manifests (`targets/*.toml`) describing memory limits, fan-in/out, and timing per hardware target.
- **Passes** (`crates/passes`) — a pipeline of named transforms over NIR (`validate`, `quantize4/8/16`, `partition`, `placement`, `routing`, `timing`, `resource-check`, plus hardware-specific passes like `tn-core-mapping`, `loihi-learning-rule`, `akida-event-routing`, etc.).
- **Backends** — one crate per hardware target, each implementing `compile(&NIR, &Manifest) → artifact`.
- **Simulators** — one crate per simulator, each implementing `emit_artifacts(&NIR, out_dir)`.
- **Orchestrator / Runtime / MLOpt** — higher-level partitioning coordination, deploy/start/stop stubs, and placeholder cost-model/search hooks for ML-driven optimization.
- **Telemetry** — JSONL profiling (timers/counters) with an optional OTLP exporter.

See [docs/architecture/overview.md](docs/architecture/overview.md) for the full data-flow writeup.

## Supported targets

| Category | Targets |
|---|---|
| Neuromorphic backends | Loihi2, TrueNorth, Akida, SpiNNaker2, NeuroGrid, DYNAP-SE, MemXbar, BrainScaleS-2, SynSense Speck, SynSense Xylo, Custom ASIC |
| RISC-V backend | `linux_user` (RV64GCV), `bare_metal` (RV32IMAC), `control_plane` (RV64G/MMIO) |
| Simulators | NEURON, CoreNEURON, Arbor, CPU reference sim, HW-specific test adapter |
| Frontends (import) | PyNN, Nengo, NEST, Brian, BindsNET, CARLsim, GeNN, Rockpool |

Run `cargo run -p neuro-compiler-cli -- list-targets` for the authoritative, current list — target manifests live under [`targets/`](targets).

> **Note:** per the project roadmap, several backends and frontends are still being built out toward full hardware-specific code generation. Check a given crate under `crates/` for its current state before relying on it.

## Quick start

```bash
# Build the default workspace (core crates only — lean by default)
cargo build --workspace

# List built-in targets
cargo run -p neuro-compiler-cli -- list-targets

# Run a no-op lowering pipeline and dump intermediate NIR artifacts
cargo run -p neuro-compiler-cli -- lower --pipeline noop --dump-dir ./out

# Compile an example graph for a RISC-V Linux target
cargo run -p neuro-compiler-cli -- compile \
  ./examples/nir/simple.json --target riscv64gcv_linux -o ./tmp/riscv_output
```

### CLI commands

`list-targets`, `import`, `lower`, `compile`, `simulate`, `profile`, `package`, `deploy`, `export-mlir`, `run` (alias `exec`). See [`crates/cli/src/main.rs`](crates/cli/src/main.rs) for full argument details, and [`AGENTS.md`](AGENTS.md) for non-obvious CLI/pipeline notes.

### GUI

A PyQt5 desktop app wraps the full CLI surface with a tab per command:

```bash
cd neuro-compiler-gui
pip install -r requirements.txt
python main.py
```

See [`neuro-compiler-gui/README.md`](neuro-compiler-gui/README.md) for details.

## Feature-gated builds

The default `cargo build --workspace` only builds the core crates (NIR, HAL, passes, runtime, telemetry, orchestrator, mlopt, CLI, Python, xtask) — backend/frontend/simulator crates are opt-in via Cargo features to keep build times down.

```bash
# Only Loihi backend + Arbor simulator
cargo run -p neuro-compiler-cli -F backend-loihi -F sim-arbor -- list-targets

# Every backend
cargo build -p neuro-compiler-cli -F backends-all

# Everything (all frontends, backends, simulators)
cargo build -p neuro-compiler-cli -F all-surfaces
```

Full feature list: [`crates/cli/Cargo.toml`](crates/cli/Cargo.toml).

## Python bindings

Feature-gated PyO3 bindings (stable ABI3, Python ≥ 3.8) expose `list_targets`, `import`, `compile`, `simulate`, and profiling summaries.

```bash
cargo build -p neuro-compiler-py
maturin build -m pyproject.toml --features python
pip install "neuro-compiler>=0.0.1,<0.1.0"
```

The package is pre-1.0, so breaking changes can happen between minor versions until the API stabilizes. See [`docs/python/usage.md`](docs/python/usage.md).

## MLIR bridge (optional)

```bash
cargo build -p nc-mlir-bridge -F mlir
```

Exports NIR to MLIR for interoperability with the broader MLIR compiler ecosystem. See [`crates/mlir-bridge`](crates/mlir-bridge).

## Telemetry & profiling

Pass-level and backend/simulator timers and counters can be written as JSON Lines:

```bash
cargo run -p neuro-compiler-cli --profile-jsonl ./profile.jsonl -- \
  simulate --simulator neuron --input examples/nir/simple.json
```

Or via env vars: `NC_PROFILE_JSONL`, `NC_OTLP_ENDPOINT` (requires the `telemetry-otlp` feature). Schema and visualization notes: [`docs/metrics/profiling.md`](docs/metrics/profiling.md).

## Documentation

Full docs are built with mdBook:

```bash
cargo install mdbook
mdbook build docs
```

Start at [`docs/src/quickstart.md`](docs/src/quickstart.md). Also see:
- [`docs/architecture/overview.md`](docs/architecture/overview.md) — architecture & data flow
- [`docs/spec/`](docs/spec) — NIR, HAL, passes, and error taxonomy specs
- [`docs/backends/riscv.md`](docs/backends/riscv.md) — RISC-V backend docs
- [`docs/backends/authoring.md`](docs/backends/authoring.md) — how to add a new backend
- [`docs/tutorials/`](docs/tutorials) — step-by-step tutorials, including the RISC-V Python SDK quickstart

## Development

```bash
# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Test
cargo test --workspace

# Coverage (requires cargo-llvm-cov + nextest)
cargo llvm-cov clean --workspace
cargo llvm-cov nextest --workspace --no-report --doctests
cargo llvm-cov report --html --output-path coverage/html

# Benchmarks (produces target/criterion)
cargo bench --workspace
```

CI ([`.github/workflows/`](.github/workflows)) runs the feature matrix, RISC-V runtime jobs (QEMU user/system, Renode), Python wheel builds, `cargo-audit`, minimal-versions builds, dependency-duplicate checks, mdBook builds, coverage, and benchmark baselines. See [`CHANGELOG.md`](CHANGELOG.md) for release history and [`ROADMAP_UPDATED.md`](ROADMAP_UPDATED.md) for where the project is headed.

## License

UNLICENSED — All Rights Reserved. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
