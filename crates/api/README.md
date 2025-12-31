# Neuro Compiler API

A REST API server for the Neuro Compiler toolchain, providing HTTP endpoints to access compilation, simulation, and analysis functionalities.

## Features

- RESTful API for neuro compiler operations
- JSON-based request/response format
- CORS support for web applications
- Comprehensive error handling
- Feature-gated backend and simulator support

## Quick Start

### Building and Running

```bash
# Build the API server
cargo build --package nc-api --release

# Run with default settings (port 3000)
cargo run --package nc-api --bin neuro-compiler-api

# Run with custom port
cargo run --package nc-api --bin neuro-compiler-api -- --port 8080

# Enable verbose logging
cargo run --package nc-api --bin neuro-compiler-api -- --verbose
```

### API Endpoints

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

#### GET /targets
List available compilation targets.

**Response:**
```json
[
  {
    "name": "riscv64gcv_linux",
    "description": "RISC-V 64-bit Linux user space"
  }
]
```

#### POST /compile
Compile a neural network model to a target platform.

**Request Body:**
```json
{
  "input": "model.json",
  "target": "riscv64gcv_linux",
  "output_dir": "compiled_output"
}
```

**Response:**
```json
{
  "status": "success",
  "output_path": "compiled_output/",
  "artifacts": ["main.c", "Makefile"]
}
```

#### POST /lower
Apply optimization passes to a model.

**Request Body:**
```json
{
  "input": "model.json",
  "pipeline": "validate,partition,placement,routing",
  "output_dir": "dumps"
}
```

#### GET /passes
List available optimization passes.

#### POST /simulate
Run simulation on a model.

**Request Body:**
```json
{
  "input": "model.json",
  "simulator": "neuron",
  "output_dir": "sim_output"
}
```

#### GET /simulators
List available simulators.

## Architecture

The API is built with:
- **Axum**: High-performance async web framework
- **Tokio**: Async runtime
- **Tower**: Middleware framework with CORS support
- **Serde**: JSON serialization/deserialization
- **Clap**: Command-line argument parsing

## Feature Gating

The API supports conditional compilation for different backends and simulators:

- `backend-riscv`: RISC-V target support
- `backend-brainscales`: BrainScaleS-2 support
- `backend-speck`: Speck backend support
- `backend-xylo`: Xylo backend support
- `sim-neuron`: NEURON simulator support

## Error Handling

All endpoints return standardized error responses:

```json
{
  "error": {
    "code": "COMPILATION_FAILED",
    "message": "Failed to compile model",
    "details": "Invalid target specification"
  }
}
```

## Development

### Adding New Endpoints

1. Add route handlers in `src/lib.rs`
2. Update request/response types
3. Add feature gating if backend-specific
4. Update documentation

### Testing

```bash
# Run unit tests
cargo test --package nc-api

# Run integration tests
cargo test --package nc-api -- --test integration
```

## Security

- Input validation on all endpoints
- CORS configuration for web clients
- No authentication (intended for development/local use)