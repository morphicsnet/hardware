//! Neuro-compiler API server
//!
//! Web API server for neuro-compiler functionality, enabling web applications
//! to access compilation and simulation capabilities.

use clap::Parser;
use nc_api::start_server;
use std::io;
use tracing_subscriber;

#[derive(Parser)]
#[command(name = "neuro-compiler-api")]
#[command(about = "Neuro-compiler REST API server")]
struct Args {
    /// Port to bind the server to
    #[arg(short, long, default_value = "3000")]
    port: u16,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(log_level)
        .init();

    println!("🧠 Neuro-compiler API Server");
    println!("📡 Starting server on port {}", args.port);
    println!("🌐 API endpoints:");
    println!("  GET  /health     - Health check");
    println!("  GET  /targets    - List available targets");
    println!("  POST /compile    - Compile neural network");
    println!("  POST /simulate   - Run simulation");
    println!();

    // Start the server
    if let Err(e) = start_server(args.port).await {
        eprintln!("❌ Server error: {}", e);
        std::process::exit(1);
    }

    Ok(())
}