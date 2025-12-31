use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use nc_nir as nir;
#[cfg(feature = "telemetry")]
use nc_telemetry as telemetry;

/// Emit Arbor-compatible artifacts and optionally run simulation
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
            let labels = telemetry::labels::simulator(&g.name, "arbor");
            Some(a.start_timer("sim.emit_ms", labels))
        } else {
            None
        }
    };

    // Generate Arbor Python model file
    generate_arbor_model(g, out_dir)?;

    // Generate simulation script
    generate_arbor_script(g, out_dir)?;

    // Generate model summary
    let summary = serde_json::json!({
        "simulator": "arbor",
        "name": g.name,
        "populations": g.populations.len(),
        "connections": g.connections.len(),
        "probes": g.probes.len(),
        "generated_files": ["model.py", "run_simulation.py", "model_summary.json"]
    });
    fs::write(out_dir.join("model_summary.json"), serde_json::to_string_pretty(&summary)?)?;

    // Generate RUN instructions
    let run_instructions = format!(
        "Arbor Simulation Instructions\n\
        ============================\n\
        Model: {}\n\
        Generated: {}\n\
        \nTo run the simulation:\n\
        1. Ensure Arbor is installed: pip install arbor\n\
        2. Run: python run_simulation.py\n\
        3. Results will be saved to: results/\n\
        \nNote: Arbor provides high-performance multi-compartment neuron simulations.\n",
        g.name,
        chrono::Utc::now().to_rfc3339()
    );
    fs::write(out_dir.join("RUN.txt"), run_instructions)?;

    #[cfg(feature = "telemetry")]
    if let Some(a) = &app {
        let l = telemetry::labels::simulator(&g.name, "arbor");
        let _ = a.counter("graph.populations", g.populations.len() as f64, l.clone());
        let _ = a.counter("graph.connections", g.connections.len() as f64, l.clone());
        let _ = a.counter("graph.probes", g.probes.len() as f64, l.clone());
        let _ = a.counter("artifacts.generated", 4.0, l);
    }

    Ok(out_dir.to_path_buf())
}

/// Generate Arbor-compatible Python model
fn generate_arbor_model(g: &nir::Graph, out_dir: &Path) -> Result<()> {
    let mut model_py = String::new();
    model_py.push_str(&format!(
        "# Arbor Model: {}\n\
        # Generated: {}\n\
        # Populations: {}\n\
        # Connections: {}\n\
        \n",
        g.name,
        chrono::Utc::now().to_rfc3339(),
        g.populations.len(),
        g.connections.len()
    ));

    model_py.push_str("import arbor\n\
import numpy as np\n\
import pandas as pd\n\
from pathlib import Path\n\
\n\
# Create Arbor model\n\
model = arbor.single_cell_model()\n\
\n");

    // Define morphologies and mechanisms for populations
    for (i, pop) in g.populations.iter().enumerate() {
        model_py.push_str(&format!(
            "# Population {}: {}\n\
            # Create morphology (single compartment)\n\
            morph_{} = arbor.segment_tree()\n\
            morph_{}.append(arbor.mnpos, arbor.mpoint(-6, 0, 0, 6), arbor.mpoint(6, 0, 0, 6), tag=1)\n\
            \n\
            # Create label dictionary for regions\n\
            labels_{} = arbor.label_dict({{'soma': '(tag 1)', 'center': '(location 0 0.5)'}})\n\
            \n\
            # Define ion channels (Hodgkin-Huxley)\n\
            hh_{} = arbor.mechanism('hh')\n\
            hh_{}['gnabar'] = 0.12  # Sodium conductance (S/cm2)\n\
            hh_{}['gkbar'] = 0.036  # Potassium conductance (S/cm2)\n\
            hh_{}['gl'] = 0.0003    # Leak conductance (S/cm2)\n\
            hh_{}['el'] = -54.3     # Leak reversal potential (mV)\n\
            \n\
            # Create decor (cable properties and mechanisms)\n\
            decor_{} = arbor.decor()\n\
            decor_{}.paint('soma', arbor.density(hh_{}))\n\
            decor_{}.paint('soma', arbor.density(arbor.mechanism('pas', {{'g': 0.001, 'e': -65}})))\n\
            \n",
            i, pop.name, i, i, i, i, i, i, i, i, i, i, i, i
        ));
    }

    // Create cells
    model_py.push_str("// Create cells\n\
cells = []\n\
cell_labels = []\n\
\n");

    for (i, pop) in g.populations.iter().enumerate() {
        model_py.push_str(&format!(
            "# Create cells for population {}\n\
            pop_{}_cells = []\n\
            for j in range({}):\n\
                cell = arbor.cable_cell(morph_{}, labels_{}, decor_{})\n\
                pop_{}_cells.append(cell)\n\
                cells.append(cell)\n\
                cell_labels.append('pop_{}_{}')\n\
            \n",
            i, i, pop.size, i, i, i, i, i, i
        ));
    }

    // Setup synapses and connections
    model_py.push_str("// Setup synapses and connections\n\
synapses = []\n\
connections = []\n\
\n");

    for (i, conn) in g.connections.iter().enumerate() {
        model_py.push_str(&format!(
            "# Connection {}: {} -> {}\n\
            # Create exponential synapse\n\
            syn_{} = arbor.mechanism('exp2syn', {{\n\
                'tau1': 0.1,     # Rise time constant (ms)\n\
                'tau2': 2.0,     # Decay time constant (ms)\n\
                'e': 0.0         # Reversal potential (mV)\n\
            }})\n\
            synapses.append(syn_{})\n\
            \n\
            # Find target cell index for population {}\n\
            # Note: Simplified mapping - assumes single cell per population for demo\n\
            target_idx = {}\n\
            target_cell = cells[target_idx]\n\
            \n\
            # Add synapse to target cell\n\
            syn_loc = arbor.location(0, 0.5)  # Middle of soma\n\
            target_cell.place(syn_loc, syn_{})\n\
            \n\
            # Create spike source for presynaptic cell\n\
            source_idx = {}\n\
            source_cell = cells[source_idx]\n\
            \n\
            # Create connection (spike source -> synapse)\n\
            conn_{} = arbor.connection(\n\
                arbor.cell_member(source_cell, 0),  # spike detector on source\n\
                arbor.cell_member(target_cell, syn_loc, 'exp2syn'),  # synapse on target\n\
                syn_{}['weight'] = {:.6},  # connection weight\n\
                syn_{}['delay'] = {:.3}    # synaptic delay (ms)\n\
            )\n\
            connections.append(conn_{})\n\
            \n",
            i, conn.pre, conn.post, i, i, i, conn.post, i, conn.pre, i,
            conn.weight, conn.weight, i, i, i
        ));
    }

    // Setup spike detectors
    model_py.push_str("// Setup spike detectors and probes\n\
detectors = []\n\
probes = []\n\
\n");

    for (i, pop) in g.populations.iter().enumerate() {
        let cell_start = g.populations.iter().take(i).map(|p| p.size as usize).sum::<usize>();
        model_py.push_str(&format!(
            "# Spike detectors for population {}\n\
            for j in range({}):\n\
                cell_idx = {} + j\n\
                detector = arbor.spike_detector(arbor.location(0, 0.5))\n\
                cells[cell_idx].place(arbor.location(0, 0.5), detector)\n\
                detectors.append(detector)\n\
                \n\
                # Voltage probe\n\
                probe = arbor.cable_probe_membrane_voltage(arbor.location(0, 0.5))\n\
                cells[cell_idx].place(arbor.location(0, 0.5), probe)\n\
                probes.append(probe)\n\
            \n",
            i, pop.size, cell_start
        ));
    }

    model_py.push_str("print('Arbor model loaded successfully')\n\
print(f'Cells created: {len(cells)}')\n\
print(f'Connections established: {len(connections)}')\n\
print(f'Synapses placed: {len(synapses)}')\n");

    fs::write(out_dir.join("model.py"), model_py)?;
    Ok(())
}

/// Generate simulation execution script
fn generate_arbor_script(g: &nir::Graph, out_dir: &Path) -> Result<()> {
    let script = format!(
        "#!/usr/bin/env python3
# Arbor Simulation Script: {}
# Generated: {}

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

# Import the generated model
exec(open('model.py').read())

def run_simulation():
    \"\"\"Run Arbor simulation and collect results\"\"\"

    print('Starting Arbor simulation...')
    start_time = time.time()

    # Simulation parameters
    tstop = 1000.0  # simulation time (ms)
    dt = 0.025      # time step (ms)

    # Create simulation recipe
    class Recipe(arbor.recipe):
        def __init__(self, cells, connections):
            arbor.recipe.__init__(self)
            self.cells = cells
            self.connections = connections
            self.probes = []
            self.detectors = []

            # Register probes and detectors
            for i, cell in enumerate(cells):
                # Voltage probe
                probe = arbor.cable_probe_membrane_voltage(arbor.location(0, 0.5))
                cell.place(arbor.location(0, 0.5), probe)
                self.probes.append((i, 0))  # gid, probe_id

                # Spike detector
                detector = arbor.spike_detector(arbor.location(0, 0.5))
                cell.place(arbor.location(0, 0.5), detector)
                self.detectors.append((i, 0))  # gid, detector_id

        def num_cells(self):
            return len(self.cells)

        def cell_kind(self, gid):
            return arbor.cell_kind.cable

        def cell_description(self, gid):
            return self.cells[gid]

        def connections_on(self, gid):
            return [c for c in self.connections if c.dest.gid == gid]

        def probes(self, gid):
            return [self.probes[gid]]

        def spike_detectors(self, gid):
            return [self.detectors[gid]]

    # Create recipe and simulation
    recipe = Recipe(cells, connections)
    context = arbor.context()
    domains = arbor.partition_load_balance(recipe, context)
    simulation = arbor.simulation(recipe, context, domains)

    # Set up spike recording
    simulation.record(arbor.spike_recording.all)

    # Set up voltage recording
    handles = []
    for gid in range(len(cells)):
        handles.append(simulation.sample((gid, 0), arbor.regular_schedule(dt)))

    print('Running simulation...')
    simulation.run(tstop, dt)

    end_time = time.time()
    sim_time = end_time - start_time

    print(f'Simulation completed in {{:.2f}} seconds', sim_time)

    # Collect results
    results = {{
        'simulator': 'arbor',
        'model': '{}',
        'simulation_time_ms': tstop,
        'time_step_ms': dt,
        'execution_time_s': sim_time,
        'cells': len(cells),
        'connections': len(connections),
        'synapses': len(synapses),
        'arbor_version': '4.0+'
    }}

    # Collect spike data
    spike_data = []
    spike_times = simulation.spikes()
    if spike_times:
        for spike in spike_times:
            spike_data.append({{
                'time': float(spike.time),
                'gid': int(spike.source.gid)
            }})
    results['spike_data'] = spike_data

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Save results
    with open(results_dir / 'simulation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save voltage traces
    voltage_data = {{
        'time': np.arange(0, tstop, dt).tolist(),
        'voltages': []
    }}

    for handle in handles:
        samples, _ = simulation.samples(handle)
        if samples:
            voltage_data['voltages'].append([float(s) for s in samples])

    with open(results_dir / 'voltage_traces.json', 'w') as f:
        json.dump(voltage_data, f, indent=2)

    print(f'Results saved to: {{results_dir}}')
    print('Arbor simulation completed successfully!')

    return results

if __name__ == '__main__':
    try:
        results = run_simulation()
        print('\\nSimulation Summary:')
        print(f'- Execution time: {{:.2f}}s', results['execution_time_s'])
        print(f'- Cells simulated: {{}}', results['cells'])
        print(f'- Connections processed: {{}}', results['connections'])
        print(f'- Synapses active: {{}}', results['synapses'])
    except Exception as e:
        print(f'Simulation failed: {{e}}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
",
        g.name,
        chrono::Utc::now().to_rfc3339(),
        g.name
    );

    fs::write(out_dir.join("run_simulation.py"), script)?;
    Ok(())
}

/// Run Arbor simulation if available
pub fn run_simulation(out_dir: &Path) -> Result<serde_json::Value> {
    let script_path = out_dir.join("run_simulation.py");

    if !script_path.exists() {
        return Err(anyhow::anyhow!("Simulation script not found: {:?}", script_path));
    }

    println!("Running Arbor simulation...");

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
