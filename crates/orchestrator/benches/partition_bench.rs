#![cfg(feature = "orchestrator_partition")]

use criterion::{criterion_group, criterion_main, Criterion};
use nc_orchestrator::{nir, GreedyRefineBuilder, PartitionPlan};
use std::fs::{create_dir_all, OpenOptions};
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

// Resolve ${TRACE_DIR} as:
//  - $CRIT_TRACE_DIR if set; else
//  - $CARGO_TARGET_DIR if set; else "target"
// Final: ${resolved}/criterion/traces/partition_bench
fn resolve_trace_subdir() -> Option<PathBuf> {
    let base = std::env::var("CRIT_TRACE_DIR")
        .ok()
        .or_else(|| std::env::var("CARGO_TARGET_DIR").ok())
        .unwrap_or_else(|| "target".to_string());
    let dir = PathBuf::from(base)
        .join("criterion")
        .join("traces")
        .join("partition_bench");
    if create_dir_all(&dir).is_ok() {
        Some(dir)
    } else {
        None
    }
}

fn open_trace(dir: &PathBuf, fname: &str) -> Option<std::io::BufWriter<std::fs::File>> {
    let path = dir.join(fname);
    let file = OpenOptions::new().create(true).append(true).open(path).ok()?;
    Some(std::io::BufWriter::new(file))
}

// Benchmark 1: sizes [16, 64, 256], fixed seed 0xC0FFEE
fn bench_plan_chain_sizes(c: &mut Criterion) {
    const FIXED_SEED: u64 = 0xC0FFEEu64;
    let sizes: [usize; 3] = [16, 64, 256];
    let targets = ["riscv64gcv_linux"];

    let trace_dir = resolve_trace_subdir();
    let trace_file = "bench_plan_chain_sizes.jsonl";

    for &size in &sizes {
        // Build a tiny chain with `size` nodes (each population size = 1)
        let layer_sizes: Vec<u32> = vec![1u32; size];
        let g = nir::fixtures::chain(&layer_sizes);
        let mut builder = GreedyRefineBuilder::new(FIXED_SEED);
        let mut writer = trace_dir.as_ref().and_then(|d| open_trace(d, trace_file));

        c.bench_function(&format!("bench_plan_chain_sizes/size_{}", size), |b| {
            b.iter(|| {
                // Local timing per iteration for trace (Criterion still does its own timing)
                let start = Instant::now();
                let plan: PartitionPlan =
                    black_box(&mut builder).plan(black_box(&g), black_box(&targets));
                let elapsed_us = start.elapsed().as_micros();

                // Append minimal JSON line
                if let Some(w) = writer.as_mut() {
                    let _ = writeln!(
                        w,
                        "{{\"run\":\"bench_plan_chain_sizes\",\"size\":{},\"seed\":{},\"parts\":{},\"elapsed_us\":{}}}",
                        size, FIXED_SEED, plan.parts, elapsed_us
                    );
                }

                black_box(plan);
            })
        });
    }
}

// Benchmark 2: size = 128, seeds [1,2,3,5,8,13,21,34]
fn bench_plan_seed_variants(c: &mut Criterion) {
    let size: usize = 128;
    let seeds: [u64; 8] = [1, 2, 3, 5, 8, 13, 21, 34];
    let targets = ["riscv64gcv_linux"];

    // Build a single graph of 128 nodes (each population size = 1)
    let layer_sizes: Vec<u32> = vec![1u32; size];
    let g = nir::fixtures::chain(&layer_sizes);

    let trace_dir = resolve_trace_subdir();
    let trace_file = "bench_plan_seed_variants.jsonl";

    for &seed in &seeds {
        let mut builder = GreedyRefineBuilder::new(seed);
        let mut writer = trace_dir.as_ref().and_then(|d| open_trace(d, trace_file));

        c.bench_function(&format!("bench_plan_seed_variants/seed_{}", seed), |b| {
            b.iter(|| {
                let start = Instant::now();
                let plan: PartitionPlan =
                    black_box(&mut builder).plan(black_box(&g), black_box(&targets));
                let elapsed_us = start.elapsed().as_micros();

                if let Some(w) = writer.as_mut() {
                    let _ = writeln!(
                        w,
                        "{{\"run\":\"bench_plan_seed_variants\",\"size\":{},\"seed\":{},\"parts\":{},\"elapsed_us\":{}}}",
                        size, seed, plan.parts, elapsed_us
                    );
                }

                black_box(plan);
            })
        });
    }
}

criterion_group!(partition_benches, bench_plan_chain_sizes, bench_plan_seed_variants);
criterion_main!(partition_benches);