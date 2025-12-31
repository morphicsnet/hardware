use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use nc_nir as nir;
#[cfg(feature = "telemetry")]
use nc_telemetry as telemetry;

/// Emit CoreNEURON-compatible artifacts and optionally run simulation
pub fn emit_artifacts(g: &nir::Graph, out_dir: &Path) -> Result<PathBuf> {
    if !out_dir.exists() {
        fs::create_dir_all(out_dir)?;
    }

    #[cfg(feature = "telemetry")]
    let app = std::env::var("NC_PROFILE_JSONL")
        .ok()
        .and_then(|p| telemetry::profiling::Appender::open(p).ok());

    #[cfg(feature = "telemetry")]
    let _timer = {
        if let Some(a) = app.as_ref() {
            let labels = telemetry::labels::simulator(&g.name, "coreneuron");
            Some(a.start_timer("sim.emit_ms", labels))
        } else {
            None
        }
    };

    // Generate CoreNEURON Python model file
    generate_coreneuron_model(g, out_dir)?;

    // Generate simulation script
    generate_coreneuron_script(g, out_dir)?;

    // Generate model summary
    let summary = serde_json::json!({
        "simulator": "coreneuron",
        "name": g.name,
        "populations": g.populations.len(),
        "connections": g.connections.len(),
        "probes": g.probes.len(),
        "generated_files": ["model.py", "run_simulation.py", "model_summary.json"]
    });
    fs::write(out_dir.join("model_summary.json"), serde_json::to_string_pretty(&summary)?)?;

    // Generate RUN instructions
    let run_instructions = format!(
        "CoreNEURON Simulation Instructions\n\
        ===============================\n\
        Model: {}\n\
        Generated: {}\n\
        \nTo run the simulation:\n\
        1. Ensure CoreNEURON is installed: pip install neuron coreneuron\n\
        2. Run: python run_simulation.py\n\
        3. Results will be saved to: results/\n\
        \nNote: CoreNEURON provides optimized parallel execution of NEURON models.\n",
        g.name,
        chrono::Utc::now().to_rfc3339()
    );
    fs::write(out_dir.join("RUN.txt"), run_instructions)?;

    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let l = telemetry::labels::simulator(&g.name, "coreneuron");
        let _ = a.counter("graph.populations", g.populations.len() as f64, l.clone());
        let _ = a.counter("graph.connections", g.connections.len() as f64, l.clone());
        let _ = a.counter("graph.probes", g.probes.len() as f64, l.clone());
        let _ = a.counter("artifacts.generated", 4.0, l);
    }

    Ok(out_dir.to_path_buf())
}

/// Generate CoreNEURON-compatible Python model
fn generate_coreneuron_model(g: &nir::Graph, out_dir: &Path) -> Result<()> {
    let mut model_py = String::new();
    model_py.push_str(&format!(
        "# CoreNEURON Model: {}\n\
        # Generated: {}\n\
        # Populations: {}\n\
        # Connections: {}\n\
        \n",
        g.name,
        chrono::Utc::now().to_rfc3339(),
        g.populations.len(),
        g.connections.len()
    ));

    model_py.push_str("import neuron\n\
from neuron import h, coreneuron\n\
import numpy as np\n\
\n\
# Initialize CoreNEURON\n\
h.load_file('stdrun.hoc')\n\
coreneuron.enable = True\n\
coreneuron.gpu = False  # Can be enabled if GPU available\n\
\n\
# Create sections for each population\n\
sections = []\n\
populations = []\n\
\n");

    // Generate sections for populations
    for (i, pop) in g.populations.iter().enumerate() {
        model_py.push_str(&format!(
            "# Population {}: {}\n\
            pop_{} = []\n\
            for j in range({}):\n\
                sec = h.Section(name='pop_{}_{}')\n\
                sec.insert('hh')  # Hodgkin-Huxley channels\n\
                sec.L = 100  # length (um)\n\
                sec.diam = 10  # diameter (um)\n\
                pop_{}.append(sec)\n\
                sections.append(sec)\n\
            populations.append(pop_{})\n\
            \n",
            i, pop.name, i, pop.size, i, 0, i, i
        ));
    }

    // Generate synapses for connections
    model_py.push_str("// Setup synapses\n\
synapses = []\n\
netcons = []\n\
\n");

    for (i, conn) in g.connections.iter().enumerate() {
        model_py.push_str(&format!(
            "# Connection {}: {} -> {}\n\
            syn_{} = h.ExpSyn(pop_{}[{}](0.5))\n\
            syn_{}.tau = 2.0  # synapse time constant (ms)\n\
            synapses.append(syn_{})\n\
            \n\
            # NetCon for presynaptic spiking\n\
            nc_{} = h.NetCon(pop_{}[{}](0.5)._ref_v, syn_{}, sec=pop_{}[{}])\n\
            nc_{}.weight[0] = {:.6}\n\
            nc_{}.delay = 1.0  # synaptic delay (ms)\n\
            netcons.append(nc_{})\n\
            \n",
            i, conn.pre, conn.post, i, conn.pre, 0, i, i,
            i, conn.pre, 0, i, conn.post, 0, i,
            conn.weight, i, i
        ));
    }

    // Setup recording
    model_py.push_str("// Setup recording\n\
voltage_recordings = []\n\
spike_times = []\n\
spike_ids = []\n\
\n");

    for (i, _pop) in g.populations.iter().enumerate() {
        model_py.push_str(&format!(
            "# Record from population {}\n\
            v_vec_{} = h.Vector()\n\
            v_vec_{}.record(pop_{}[0](0.5)._ref_v)\n\
            voltage_recordings.append(v_vec_{})\n\
            \n\
            # Spike recording\n\
            nc_spike_{} = h.NetCon(pop_{}[0](0.5)._ref_v, None, sec=pop_{}[0])\n\
            nc_spike_{}.threshold = -20  # spike threshold (mV)\n\
            spike_vec_{} = h.Vector()\n\
            nc_spike_{}.record(spike_vec_{})\n\
            spike_times.append(spike_vec_{})\n\
            spike_ids.append(h.Vector())\n\
            spike_ids[-1].append({})\n\
            \n",
            i, i, i, i, i, i, i, i, i, i, i, i, i, i
        ));
    }

    model_py.push_str("print('CoreNEURON model loaded successfully')\n\
print(f'Populations: {len(populations)}')\n\
print(f'Connections: {len(synapses)}')\n");

    fs::write(out_dir.join("model.py"), model_py)?;
    Ok(())
}

/// Generate simulation execution script
fn generate_coreneuron_script(g: &nir::Graph, out_dir: &Path) -> Result<()> {
    let script = format!(
        "#!/usr/bin/env python3
# CoreNEURON Simulation Script: {}
# Generated: {}

import os
import sys
import time
import json
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

# Import the generated model
exec(open('model.py').read())

def run_simulation():
    \"\"\"Run CoreNEURON simulation and collect results\"\"\"

    print('Starting CoreNEURON simulation...')
    start_time = time.time()

    # Simulation parameters
    h.tstop = 1000.0  # simulation time (ms)
    h.dt = 0.025      # time step (ms)

    # Initialize
    h.finitialize(-65)  # resting potential (mV)

    # Run simulation
    print('Running simulation...')
    h.run()

    end_time = time.time()
    sim_time = end_time - start_time

    print(f'Simulation completed in {{:.2f}} seconds', sim_time)
    print(f'CoreNEURON enabled: {{}}', coreneuron.enable)

    # Collect results
    results = {{
        'simulator': 'coreneuron',
        'model': '{}',
        'simulation_time_ms': h.tstop,
        'time_step_ms': h.dt,
        'execution_time_s': sim_time,
        'populations': len(populations),
        'connections': len(synapses),
        'coreneuron_enabled': coreneuron.enable,
        'coreneuron_gpu': coreneuron.gpu
    }}

    # Save spike times if available
    if spike_times and len(spike_times) > 0:
        spike_data = []
        for i, spike_vec in enumerate(spike_times):
            if len(spike_vec) > 0:
                spike_data.append({{
                    'population': i,
                    'times': list(spike_vec)
                }})
        results['spike_data'] = spike_data

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Save results
    with open(results_dir / 'simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save voltage traces if available
    if voltage_recordings:
        voltage_data = {{
            'time': list(h.Vector().record(h._ref_t)),
            'voltages': [list(vec) for vec in voltage_recordings]
        }}
        with open(results_dir / 'voltage_traces.json', 'w') as f:
            json.dump(voltage_data, f, indent=2)

    print(f'Results saved to: {{results_dir}}')
    print('CoreNEURON simulation completed successfully!')

    return results

if __name__ == '__main__':
    try:
        results = run_simulation()
        print('\\nSimulation Summary:')
        print(f'- Execution time: {{:.2f}}s', results['execution_time_s'])
        print(f'- Populations simulated: {{}}', results['populations'])
        print(f'- Connections processed: {{}}', results['connections'])
        print(f'- CoreNEURON enabled: {{}}', results['coreneuron_enabled'])
    except Exception as e:
        print(f'Simulation failed: {{e}}')
        sys.exit(1)
",
        g.name,
        chrono::Utc::now().to_rfc3339(),
        g.name
    );

    fs::write(out_dir.join("run_simulation.py"), script)?;
    Ok(())
}

/// Run CoreNEURON simulation if available
pub fn run_simulation(out_dir: &Path) -> Result<serde_json::Value> {
    let script_path = out_dir.join("run_simulation.py");

    if !script_path.exists() {
        return Err(anyhow::anyhow!("Simulation script not found: {:?}", script_path));
    }

    println!("Running CoreNEURON simulation...");

    let output = Command::new("python3")
        .arg(&script_path)
        .current_dir(out_dir)
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to run simulation: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("Simulation failed: {}", stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);

    // Try to read results
    let results_path = out_dir.join("results").join("simulation_results.json");
    if results_path.exists() {
        let results_content = fs::read_to_string(results_path)?;
        let results: serde_json::Value = serde_json::from_str(&results_content)?;
        Ok(results)
    } else {
        Ok(serde_json::json!({
            "status": "completed",
            "stdout": stdout.to_string(),
            "stderr": String::from_utf8_lossy(&output.stderr).to_string()
        }))
    }
}

pub fn stub() -> &'static str { "ok" }
