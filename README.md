# neuro-compiler 

A universal neuromorphic compiler scaffold with Rust core and optional Python bindings.

## Features

This repo currently provides:
- **Canonical IR**: NIR (Neuromorphic Intermediate Representation) for cross-platform model definitions
- **Hardware Abstraction Layer**: Unified interface supporting multiple neuromorphic targets
- **Pass System**: Extensible optimization and transformation pipeline
- **Runtime Support**: Telemetry, orchestration, and ML optimization capabilities
- **CLI Interface**: Command-line tools with target listing and lowering pipeline controls
- **Python Bindings**: Optional Python API with maturin integration (feature-gated)
- **MLIR Bridge**: Integration with MLIR infrastructure (gated behind 'mlir' feature)
- **RISC-V Backend**: Compile neuromorphic models to C code for a variety of RISC-V targets, from high-performance `RV64GCV` Linux systems to `RV32IMAC` bare-metal microcontrollers. Supported via the `backend-riscv` feature.
- **Documentation**: Complete mdBook documentation and examples

Quick start (build workspace):
- cargo build --workspace

Try CLI:
- cargo run -p neuro-compiler-cli -- list-targets

Enable MLIR bridge (example exporter, feature-gated):
- cargo build -p nc-mlir-bridge -F mlir

Python bindings (feature-gated; build wheels with maturin):
- cargo build -p neuro-compiler-py
- Build wheel:
  maturin build -m pyproject.toml --features python

Docs (mdBook):
- cargo install mdbook
- mdbook build docs
- Start at docs/src/quickstart.md

License: UNLICENSED - All Rights Reserved — see [LICENSE](LICENSE).

## Backend: RISC-V

The neuromorphic compiler can now target the RISC-V architecture, enabling deployment on open-standard hardware ranging from high-performance Linux systems to resource-constrained bare-metal microcontrollers.

### Supported Profiles

The RISC-V backend supports three distinct deployment profiles:

- **`linux_user`**: For general-purpose RV64GCV cores running Linux. Optimizes for performance using the Vector 'V' extension.
- **`bare_metal`**: For resource-constrained RV32IMAC microcontrollers. Generates a self-contained firmware image.
- **`control_plane`**: For RV64G cores acting as a control plane for custom accelerators, generating code that uses MMIO.

### CLI Usage Example

```bash
# Compile a simple NIR graph for a 64-bit Linux RISC-V target
neuro-compiler compile ./examples/nir/simple.json --target riscv64gcv_linux -o ./tmp/riscv_output
```

### Documentation

- For more information on toolchains and usage, see the [RISC-V Backend Docs](docs/backends/riscv.md).
- To get started with the Python SDK, see the [RISC-V Python Quickstart](docs/tutorials/riscv_pysdk_quickstart.md).

## Minimal builds via feature flags (per surface)

The CLI supports opt-in features for each frontend, backend, and simulator crate. This minimizes build time and dependency footprint. See [crates/cli/Cargo.toml](crates/cli/Cargo.toml) for the complete feature list and aggregate flags.

Examples:
- Build with only Loihi backend and Arbor sim:
  - cargo run -p neuro-compiler-cli -F backend-loihi -F sim-arbor -- list-targets
- Build with all backends:
  - cargo build -p neuro-compiler-cli -F backends-all
- Build with all surfaces (frontends/backends/sims):
  - cargo build -p neuro-compiler-cli -F all-surfaces

Note: Workspace default-members exclude most surface crates to keep the default `cargo build --workspace` lean. You can still build any crate explicitly by selecting it with `-p` or enabling the CLI features above.

## Structured lowering pipeline and intermediate artifacts

Lowering supports a structured pipeline config and intermediate artifact dumping via the CLI. See [crates/cli/src/main.rs](crates/cli/src/main.rs) and [crates/passes/src/lib.rs](crates/passes/src/lib.rs).

- Run a no-op pipeline and dump artifacts (JSON by default):
  - cargo run -p neuro-compiler-cli -- lower --pipeline noop --dump-dir ./out
- Dump YAML instead (or both, comma separated):
  - cargo run -p neuro-compiler-cli -- lower --pipeline noop --dump-dir ./out --dump-format yaml
  - cargo run -p neuro-compiler-cli -- lower --pipeline noop --dump-dir ./out --dump-format json,yaml

The IR graph dumps are round-trippable via [crates/nir/src/lib.rs](crates/nir/src/lib.rs) JSON/YAML serializers.

## Profiling schema and quick visualization

A JSON Lines schema for profiling is defined in [crates/telemetry/src/lib.rs](crates/telemetry/src/lib.rs) (module `profiling`) and documented in [docs/metrics/profiling.md](docs/metrics/profiling.md). Emit profile records as JSONL using the helper and visualize with Python (Altair/Matplotlib) following the doc’s example.

CLI telemetry flags:
- --profile-jsonl / NC_PROFILE_JSONL to write JSONL timers/counters
- --otlp-endpoint / NC_OTLP_ENDPOINT (when built with 'telemetry-otlp') to export traces via OTLP

## Python API versioning policy

- The Python package uses a stable ABI3 wheel (built with pyo3 `abi3-py38`), targeting Python ≥ 3.8.
- Versioning aligns with the Rust workspace version in [Cargo.toml](Cargo.toml) and [pyproject.toml](pyproject.toml).
- Within 0.x, breaking changes may occur between minor versions. We recommend pinning a compatible range:
  - pip install "neuro-compiler>=0.0.1,<0.1.0"
- Once the API stabilizes at 1.0+, semantic versioning will be followed with deprecation windows noted in the docs/book.
- Packaging constraints:
  - Wheels are built via maturin; see [pyproject.toml](pyproject.toml) for classifiers and `requires-python`.
  - The Rust extension uses the stable CPython ABI (no per-Python rebuilds required for supported versions).

## CI artifacts: Coverage and Benchmarks

Where to find workflow artifacts
- In GitHub, open the Actions tab.
- Choose the "Coverage" or "Bench" workflow.
- Click the latest successful run on the default branch.
- In the run summary, locate the "Artifacts" panel.

Artifact names and contents
- coverage-lcov → lcov.info (repo root)
- coverage-html → coverage/html (HTML report directory)
- criterion-baseline → criterion-baseline.tar.gz (tar of target/criterion)

Viewing instructions

Coverage HTML
- Download coverage-html and extract it, then open the report entry point:

```bash
open coverage/html/index.html
# or on Linux:
# xdg-open coverage/html/index.html
```

LCOV (lcov.info -> HTML via genhtml)
- macOS/Homebrew:

```bash
brew install lcov
genhtml lcov.info -o coverage/html
open coverage/html/index.html
```

Criterion baseline
- Download criterion-baseline, extract the tarball, then open the reports:

```bash
tar -xzf criterion-baseline.tar.gz
open target/criterion/index.html
# or open a specific benchmark:
# open target/criterion/<bench-name>/report/index.html
```

Local reproduction (run locally without CI)
```bash
# Coverage (requires cargo-llvm-cov and nextest installed)
cargo llvm-cov clean --workspace
cargo llvm-cov nextest --workspace --no-report --doctests
cargo llvm-cov report --html --output-path coverage/html

# Benchmarks (produces target/criterion)
cargo bench --workspace
```
