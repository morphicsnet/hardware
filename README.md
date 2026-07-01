# Universal Neuromorphic Artifact Compiler

A Rust-first compiler workspace for lowering spiking neural network models into deterministic compile-time artifacts across heterogeneous neuromorphic hardware, simulation surfaces, RISC-V targets, and future custom accelerator backends.

This repository is no longer just a scaffold. It is organized as a multi-surface compiler stack with:

- a canonical Neuromorphic Intermediate Representation (`NIR`),
- a mid-level representation layer (`MIR`),
- a target-manifest-driven Hardware Abstraction Layer (`HAL`),
- deterministic lowering passes,
- feature-gated frontend importers,
- feature-gated backend emitters,
- simulator adapters,
- telemetry/profiling hooks,
- CLI, API, Python bindings, and GUI entry points,
- CI, documentation, fixtures, generated dumps, and roadmap material.

The core design principle is simple:

> Define a neuromorphic model once, validate and lower it deterministically, then emit the right artifact for the selected execution substrate.

Those artifacts may be backend configuration files, generated C, firmware-oriented outputs, simulator run directories, MLIR exports, profiling traces, or deployment/runtime packages.

---

## Repository map

```text
hardware/
├── crates/
│   ├── nir/                    # Canonical Neuromorphic IR
│   ├── mir/                    # Mid-level compiler representation
│   ├── hal/                    # Target manifests, capabilities, constraints
│   ├── passes/                 # Validate, quantize, partition, place, route, timing, checks
│   ├── runtime/                # Deployment/start/stop/status integration surface
│   ├── telemetry/              # JSONL profiling and optional OTLP tracing
│   ├── orchestrator/           # Multi-target and multi-chip coordination surface
│   ├── mlopt/                  # Cost-model and optimization hooks
│   ├── mlir-bridge/            # Optional MLIR export/lowering bridge
│   ├── api/                    # Programmatic API surface
│   ├── cli/                    # `neuro-compiler` command-line interface
│   ├── py/                     # PyO3 Python bindings
│   ├── frontend_*/             # PyNN, Nengo, NEST, Brian, BindsNET, CARLsim, GeNN, Rockpool
│   ├── sim_*/                  # NEURON, CoreNEURON, Arbor, hardware-specific simulator adapters
│   ├── backend_*/              # Hardware, simulator, RISC-V, and accelerator emitters
│   └── xtask/                  # Repository maintenance tasks
├── targets/                    # Declarative target profiles / HAL manifests
├── examples/                   # Example NIR graphs and usage inputs
├── fixtures/                   # Test fixtures
├── docs/                       # Architecture, specs, metrics, backend, Python, release docs
├── neuro-compiler-gui/         # Desktop GUI surface
├── out/                        # Example/generated compiler outputs
├── test-dumps/                 # Lowering dump examples
├── test-dumps-telemetry/       # Telemetry dump examples
└── .github/                    # CI, coverage, benchmark, issue/PR automation
```

---

## Core compiler architecture

The compiler follows a layered flow:

```text
Frontend importer / NIR file
        ↓
NIR graph validation
        ↓
Lowering pipeline
  validate → quantize → partition → placement → routing → timing → resource-check
        ↓
HAL-aware target selection
        ↓
Backend or simulator artifact emission
        ↓
Telemetry, profiling, packaging, deployment hooks
```

### NIR: target-independent model representation

`crates/nir` defines the canonical model graph. It represents populations, neuron models, parameters, connections, delays, plasticity metadata, probes, and cross-cutting attributes. NIR is serializable as JSON/YAML and is intended to be the stable interchange layer across compiler passes, importers, backends, and simulators.

### HAL: target constraints as compiler inputs

`crates/hal` and `targets/` define the Hardware Abstraction Layer. Target manifests encode hardware constraints such as memory limits, neurons/synapses per core, fan-in/fan-out boundaries, interconnect limits, timing resolution, weight precision, and device-specific capability metadata.

The compiler treats these constraints as first-class inputs during partitioning, placement, routing, timing, and resource checking.

### Passes: deterministic lowering

`crates/passes` provides the pass framework and built-in lowering passes:

- `validate`
- `quantize`
- `partition`
- `placement`
- `routing`
- `timing`
- `resource-check`

Passes can dump intermediate IR artifacts in JSON/YAML/bin formats for inspection, regression testing, reproducibility, and paper-quality traceability.

---

## Supported surfaces

The workspace is intentionally broad. Most surfaces are feature-gated so default builds remain lean while specialized builds can enable exactly the needed importers, backends, and simulators.

### Frontend importers

Feature-gated frontend crates are present for:

- PyNN
- Nengo
- NEST
- Brian
- BindsNET
- CARLsim
- GeNN
- Rockpool

These are intended to translate framework-specific models into the canonical NIR layer.

### Hardware and accelerator backends

Feature-gated backend crates are present for:

- Intel Loihi / Loihi-style targets
- IBM TrueNorth
- BrainChip Akida
- SpiNNaker
- Stanford Neurogrid
- DYNAPs / mixed-signal event-based systems
- Memristive crossbars (`MemXbar`)
- Custom ASIC targets
- RISC-V targets
- Speck
- Xylo
- BrainScaleS
- CPU reference simulation

### Simulator adapters

Simulator crates are present for:

- NEURON
- CoreNEURON
- Arbor
- hardware-specific simulator/test adapters

The simulator path is treated as another artifact generation surface rather than a separate modeling universe.

---

## RISC-V backend

The RISC-V backend is the most developed hardware/software co-design path in the current repository. It supports three deployment profiles:

| Target profile | Purpose |
|---|---|
| `riscv64gcv_linux` | RV64GCV Linux userspace with optional vector-aware codegen paths |
| `riscv32imac_bare` | RV32IMAC bare-metal / RTOS-style firmware generation |
| `riscv64gc_ctrl` | RV64GC Linux control plane for MMIO/DMA accelerator control |

The backend emits C/runtime artifacts, pass metadata, warning files, and optional profiling output. Depending on the profile and installed tools, it can integrate with QEMU user-mode, QEMU system-mode, or Renode-style control-plane simulation.

See:

- [`docs/backends/riscv.md`](docs/backends/riscv.md)
- [`crates/backend_riscv/`](crates/backend_riscv/)

---

## CLI quick start

Build the default workspace members:

```bash
cargo build --workspace
```

List known targets:

```bash
cargo run -p neuro-compiler-cli -- list-targets
```

Run a lowering pipeline and dump intermediate artifacts:

```bash
cargo run -p neuro-compiler-cli -- lower \
  --pipeline noop \
  --dump-dir ./out
```

Dump YAML instead of JSON:

```bash
cargo run -p neuro-compiler-cli -- lower \
  --pipeline noop \
  --dump-dir ./out \
  --dump-format yaml
```

Compile with a specific backend feature enabled:

```bash
cargo run -p neuro-compiler-cli \
  --features backend-riscv \
  -- compile \
  --input examples/nir/simple.json \
  --target riscv64gcv_linux
```

Build with all backend crates enabled:

```bash
cargo build -p neuro-compiler-cli --features backends-all
```

Build with all frontends, backends, and simulators enabled:

```bash
cargo build -p neuro-compiler-cli --features all-surfaces
```

---

## Feature flags

The CLI keeps most surfaces opt-in. Important aggregate flags include:

| Feature | Enables |
|---|---|
| `frontends-all` | All frontend importer crates |
| `backends-all` | All hardware/backend emitter crates |
| `sims-all` | All simulator adapter crates |
| `all-surfaces` | Frontends + backends + simulators |
| `mlir` | Optional MLIR bridge |
| `telemetry` | JSONL profiling hooks across supported crates |
| `telemetry-otlp` | OTLP tracing export path |
| `bin-artifacts` | Binary dump support in passes |

For the complete feature matrix, see [`crates/cli/Cargo.toml`](crates/cli/Cargo.toml).

---

## Python bindings

The Python surface is implemented with PyO3 and maturin.

Build the Python crate:

```bash
cargo build -p neuro-compiler-py
```

Build a wheel:

```bash
maturin build -m pyproject.toml --features python
```

The package targets Python 3.8+ through the stable ABI3 path. See:

- [`crates/py/`](crates/py/)
- [`docs/python/usage.md`](docs/python/usage.md)

---

## GUI

A desktop GUI is available under:

```bash
cd neuro-compiler-gui
python main.py
```

The GUI is intended to expose compiler operations visually: target inspection, model loading, lowering, compilation, simulation, telemetry, and artifact browsing.

---

## Documentation

The documentation tree contains architecture, backend, metrics, Python, release, and specification material.

Start with:

- [`docs/architecture/overview.md`](docs/architecture/overview.md)
- [`docs/spec/`](docs/spec/)
- [`docs/backends/riscv.md`](docs/backends/riscv.md)
- [`docs/metrics/profiling.md`](docs/metrics/profiling.md)
- [`docs/python/usage.md`](docs/python/usage.md)
- [`ROADMAP_UPDATED.md`](ROADMAP_UPDATED.md)

Build the docs with mdBook if the book configuration is present in your checkout:

```bash
cargo install mdbook
mdbook build docs
```

---

## Telemetry and profiling

The telemetry layer supports JSONL profiling and optional OTLP export.

Common controls:

```bash
export NC_PROFILE_JSONL=target/profile.jsonl
export NC_OTLP_ENDPOINT=http://localhost:4317
```

Relevant CLI flags and environment variables include:

- `--profile-jsonl` / `NC_PROFILE_JSONL`
- `--otlp-endpoint` / `NC_OTLP_ENDPOINT`

Profiling captures compiler, backend, simulator, graph, pass, and target labels using a consistent schema. See [`docs/metrics/profiling.md`](docs/metrics/profiling.md).

---

## CI, coverage, and benchmarks

The repository includes GitHub workflows for CI, coverage, and benchmarking. Coverage artifacts include LCOV and HTML reports; benchmark artifacts use Criterion output.

Local coverage reproduction:

```bash
cargo llvm-cov clean --workspace
cargo llvm-cov nextest --workspace --no-report --doctests
cargo llvm-cov report --html --output-path coverage/html
```

Local benchmarks:

```bash
cargo bench --workspace
```

---

## Development model

The workspace is designed around additive extension points:

### Add a backend

1. Create a new `crates/backend_*` crate.
2. Implement the backend compile surface.
3. Add or update a target manifest under `targets/`.
4. Wire the backend into CLI feature flags.
5. Add docs, examples, and tests.

### Add a frontend

1. Create a new `crates/frontend_*` crate.
2. Translate the external framework representation into NIR.
3. Register the importer through the CLI/API/Python surfaces as needed.

### Add a simulator adapter

1. Create a new `crates/sim_*` crate.
2. Emit runnable simulator artifacts.
3. Integrate telemetry and artifact directory conventions.
4. Add examples and documentation.

### Add a compiler pass

1. Implement the pass trait in `crates/passes` or a backend-specific pipeline.
2. Register it in the relevant pipeline builder.
3. Add dump/round-trip tests if the pass mutates NIR.
4. Keep iteration order deterministic.

---

## Research framing

This repository supports research in:

- deterministic compilation for spiking neural networks,
- hardware-aware lowering and partitioning,
- neuromorphic target description languages,
- heterogeneous backend artifact generation,
- simulator/hardware equivalence testing,
- RISC-V control-plane generation for neuromorphic accelerators,
- FPGA/ASIC-oriented compiler pathways,
- telemetry-driven optimization and profiling,
- MLIR interoperability for neuromorphic compiler infrastructure.

The long-term goal is a universal artifact compiler: one model representation, many physically distinct execution substrates, deterministic lowering, inspectable artifacts, and reproducible traces.

---

## License

UNLICENSED - All Rights Reserved. See [`LICENSE`](LICENSE).
