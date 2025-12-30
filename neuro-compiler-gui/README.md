# Neuro-Compiler GUI

A modern, user-friendly graphical desktop application for the neuro-compiler toolchain, built with Python and PyQt5.

## Features

- **Complete CLI Integration**: Access all neuro-compiler CLI functionalities through an intuitive GUI
- **Tabbed Interface**: Organized tabs for each command (List Targets, Import, Lower, Compile, Simulate, etc.)
- **Real-time Updates**: Asynchronous execution with progress indicators and live output
- **Input Validation**: Built-in validation for all inputs with user-friendly error messages
- **Security**: Data encryption for sensitive information and input sanitization
- **Accessibility**: Keyboard shortcuts, screen reader support, and proper ARIA labels
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Offline Capabilities**: Core functionality works without internet connection
- **Professional Theme**: Clean, modern interface with customizable styling

## Requirements

- Python 3.7+
- PyQt5
- cryptography

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure neuro-compiler CLI is available in PATH or set the path in the application settings.

## Usage

Run the application:
```bash
python main.py
```

### Available Tabs

- **List Targets**: Display all available compilation targets
- **Import**: Import models from various frameworks (PyNN, Nengo, etc.)
- **Lower**: Run optimization passes on NIR graphs
- **Compile**: Compile models to specific hardware targets
- **Simulate**: Run simulations with various simulators
- **Profile**: Analyze performance metrics from runs
- **Package**: Create deployment packages
- **Deploy**: Deploy to hardware or cloud targets
- **Export MLIR**: Export NIR graphs to MLIR format
- **Run**: Execute NIR+EIR plans on HAL backends

### Keyboard Shortcuts

- `Ctrl+Enter`: Execute current tab's command
- `Ctrl+T`: Switch to next tab
- `Ctrl+Shift+T`: Switch to previous tab
- `Ctrl+Q`: Quit application

## Architecture

### System Overview

The Neuro-Compiler GUI follows a modular MVC (Model-View-Controller) architecture built on PyQt5, providing a modern graphical interface to the neuro-compiler CLI toolchain. The application integrates all CLI functionalities through a tabbed interface while maintaining real-time feedback and robust error handling.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MainWindow    │    │   BaseTab       │    │  CLIExecutor    │
│   (Controller)  │◄──►│   (View)        │◄──►│   (Model)       │
│                 │    │                 │    │                 │
│ • Tab Management│    │ • UI Components │    │ • Command Exec  │
│ • Event Handling│    │ • Input Forms   │    │ • Validation     │
│ • Status Updates│    │ • Output Display│    │ • Encryption     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Classes

#### Application Layer
- **`NeuroCompilerGUI`**: Main application class extending QApplication
  - Handles application lifecycle and global settings
  - Manages high DPI scaling and platform-specific configurations
  - Provides application-wide theme and font management

- **`MainWindow`**: Central window with menu bar, status bar, and tab widget
  - Manages tab navigation and keyboard shortcuts
  - Provides status bar updates and progress tracking
  - Handles window geometry and accessibility settings

#### Tab System
- **`BaseTab`**: Abstract base class for all functionality tabs
  - Defines common UI patterns and behaviors
  - Provides file/directory selection utilities
  - Implements background thread management
  - Handles output display and error reporting

- **Concrete Tabs**: Specialized implementations for each CLI command
  - `ListTargetsTab`: Target enumeration interface
  - `ImportTab`: Model import functionality
  - `LowerTab`: Pass pipeline execution
  - `CompileTab`: Hardware compilation interface
  - `SimulateTab`: Simulator execution controls
  - Additional tabs for profile, package, deploy, export, and run commands

#### Backend Layer
- **`CLIExecutor`**: Handles subprocess calls to neuro-compiler CLI
  - Automatic CLI discovery (PATH or cargo run)
  - Command argument validation and sanitization
  - Secure data encryption for sensitive inputs
  - Cross-platform path resolution

- **`CLIWorker`**: QThread subclass for background command execution
  - Asynchronous command processing
  - Real-time output streaming
  - Timeout handling and cancellation support
  - Signal-based progress reporting

### MVC Pattern Implementation

- **Model Layer**:
  - `CLIExecutor`: Core business logic for CLI interactions
  - `CLIWorker`: Background processing and state management
  - Data validation and encryption services
  - Configuration persistence and retrieval

- **View Layer**:
  - PyQt5 widgets for all UI components
  - Custom form layouts for each command type
  - Real-time output display with syntax highlighting
  - Progress indicators and status visualizations

- **Controller Layer**:
  - Event handling and user interaction management
  - Input validation and error handling
  - Thread coordination and lifecycle management
  - State synchronization between model and view

### Security Architecture

#### Input Validation
- **File Path Validation**: Ensures files exist and are accessible
- **Directory Validation**: Verifies directory permissions and structure
- **Command Argument Sanitization**: Prevents injection attacks
- **Pipeline Validation**: Verifies pass names against known valid passes

#### Data Encryption
- **Fernet AES Encryption**: Symmetric encryption for sensitive data
- **Secure Key Storage**: Keys stored in user home with restrictive permissions (0o600)
- **Automatic Key Generation**: First-run key creation with fallback handling
- **Memory-Safe Operations**: No plaintext sensitive data in memory

#### Platform Security
- **Permission-Aware Operations**: Respects file system permissions
- **Environment Isolation**: Clean environment for subprocess execution
- **Path Resolution**: Secure path handling across platforms

### Threading Architecture

The application uses a sophisticated threading model to maintain UI responsiveness:

#### Background Workers
- **`CLIWorker`**: Primary command execution thread
  - Emits progress, result, and error signals
  - Handles subprocess lifecycle management
  - Supports command cancellation and timeout

- **`ValidationWorker`**: Input validation thread
  - Asynchronous validation to prevent UI blocking
  - Caches validation results for performance
  - Provides detailed validation feedback

#### Thread Coordination
- **Signal-Slot Mechanism**: Qt's signal-slot system for thread communication
- **Worker Lifecycle Management**: Proper cleanup and resource management
- **UI Thread Protection**: All UI updates occur on main thread
- **Exception Handling**: Thread-safe error propagation to UI

### Configuration Management

#### Settings Storage
- **Platform-Specific Locations**:
  - Windows: `%APPDATA%\neuro-compiler-gui\`
  - macOS: `~/Library/Application Support/neuro-compiler-gui/`
  - Linux: `~/.config/neuro-compiler-gui/`

#### Persistent Data
- **Encryption Keys**: Securely stored for data protection
- **User Preferences**: Theme, font, and layout settings
- **Recent Files**: MRU lists for file inputs
- **Command History**: Cached results and configurations

## Cross-platform Compatibility

The application uses PyQt5 which provides native look and feel on all platforms:
- **Windows**: Uses Windows native theme
- **macOS**: Integrates with macOS design language
- **Linux**: Adapts to system GTK theme

High DPI scaling is enabled for all platforms.

## Accessibility

- All widgets have proper accessible names and descriptions
- Keyboard navigation support
- Screen reader compatibility
- High contrast theme support

## Configuration

Settings are stored in standard platform locations:
- **Windows**: `%APPDATA%\neuro-compiler-gui\`
- **macOS**: `~/Library/Application Support/neuro-compiler-gui/`
- **Linux**: `~/.config/neuro-compiler-gui/`

## Development

### Adding New Tabs

1. Create a new tab class inheriting from `BaseTab`
2. Implement required methods: `create_input_section()`, `get_command_args()`, etc.
3. Add the tab to `MainWindow.init_tabs()`
4. Import the tab in `main.py`

### Extending CLI Integration

Modify `CLIExecutor` to add new validation rules or command handling.

## API Documentation

### CLIExecutor Class

The core class for neuro-compiler CLI interaction.

#### Constructor
```python
CLIExecutor(neuro_compiler_path: Optional[str] = None)
```

**Parameters:**
- `neuro_compiler_path`: Path to neuro-compiler CLI executable. Auto-discovered if None.

#### Methods

##### execute_command(command: str, args: List[str] = None, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]

Execute a neuro-compiler CLI command.

**Parameters:**
- `command`: CLI subcommand (e.g., 'list-targets')
- `args`: List of command arguments
- `cwd`: Working directory for execution
- `env`: Additional environment variables

**Returns:** Dictionary with 'success', 'stdout', 'stderr', 'returncode' keys.

**Example:**
```python
result = cli_executor.execute_command('list-targets')
if result['success']:
    print(result['stdout'])
```

##### validate_input(input_type: str, value: str) -> bool

Validate input based on type.

**Parameters:**
- `input_type`: Type of input ('file_path', 'target', 'simulator', 'pipeline', 'dump_format')
- `value`: Value to validate

**Returns:** True if valid, False otherwise.

### BaseTab Class

Abstract base class for all GUI tabs.

#### Key Methods (Override in Subclasses)

##### create_input_section(layout: QVBoxLayout) -> None

Create input controls for the tab.

##### create_output_section(layout: QVBoxLayout) -> None

Create output display section.

##### validate_inputs() -> bool

Validate all input fields.

##### get_command_args() -> List[str]

Return command line arguments for CLI execution.

##### get_command_name() -> str

Return the CLI command name.

#### Utility Methods

##### add_file_selector(layout, label_text: str, button_text: str = "Browse...", file_filter: str = "All Files (*)") -> Tuple[QLineEdit, QPushButton]

Add a file selection widget.

##### add_directory_selector(layout, label_text: str, button_text: str = "Browse...") -> Tuple[QLineEdit, QPushButton]

Add a directory selection widget.

##### add_text_input(layout, label_text: str, placeholder: str = "") -> QLineEdit

Add a text input field.

##### add_combo_box(layout, label_text: str, items: List[str]) -> QComboBox

Add a combo box with predefined items.

## Tab Functionalities

### List Targets Tab

**CLI Command:** `list-targets`

Lists all available compilation targets defined in the neuro-compiler project.

**Inputs:** None

**Outputs:** JSON/YAML list of target configurations

### Import Tab

**CLI Command:** `import --framework <framework> --input <input_file> --output <output_file>`

Imports neural network models from various frameworks.

**Inputs:**
- Framework: PyNN, Nengo, Brian2, etc.
- Input file: Path to model file
- Output file: Path for NIR output

### Lower Tab

**CLI Command:** `lower --input <input> --pipeline <pipeline> --dump-dir <dir> --dump-format <format>`

Runs optimization passes on NIR graphs.

**Inputs:**
- Input file: NIR graph file (.json/.yaml)
- Pipeline: Comma-separated pass names
- Dump directory: Output directory for intermediate results
- Dump format: json, yaml, bin

**Valid Passes:**
- `validate`, `quantize4`, `quantize8`, `quantize16`
- `partition`, `placement`, `routing`
- `timing`, `resource-check`
- Target-specific: `tn-*`, `sn-*`, `loihi-*`, `akida-*`

### Compile Tab

**CLI Command:** `compile --input <input> --target <target> --output <output>`

Compiles NIR graphs to hardware-specific code.

**Inputs:**
- Input file: NIR graph file
- Target: Hardware target (e.g., riscv64gcv_linux)
- Output directory: Compilation output location

### Simulate Tab

**CLI Command:** `simulate --simulator <simulator> --input <input>`

Runs simulations using various neuromorphic simulators.

**Inputs:**
- Simulator: neuron, coreneuron, arbor, hw
- Input file: NIR graph or compiled model

**Features Required:**
- `--features sim-neuron` for NEURON simulator
- `--features sim-arbor` for Arbor simulator
- `--features sim-hw-specific` for hardware simulators

### Additional Tabs

- **Profile Tab:** Performance analysis and metrics
- **Package Tab:** Create deployment packages
- **Deploy Tab:** Deploy to hardware/cloud targets
- **Export MLIR Tab:** Convert NIR to MLIR format
- **Run Tab:** Execute NIR+EIR plans on HAL backends

## Examples and Use Cases

### Basic Workflow Example

1. **Import a Model:**
   ```
   Framework: PyNN
   Input: model.py
   Output: model.nir.json
   ```

2. **Lower with Optimization:**
   ```
   Input: model.nir.json
   Pipeline: validate,quantize8,partition,placement,routing
   Dump Dir: ./dumps
   Dump Format: json
   ```

3. **Compile to Target:**
   ```
   Input: model.nir.json
   Target: riscv64gcv_linux
   Output: ./compiled
   ```

4. **Simulate:**
   ```
   Simulator: neuron
   Input: compiled/model.neuron
   ```

### Advanced Pipeline Example

For a complex neuromorphic application:

```bash
# Import from Nengo
# Lower with full optimization pipeline
# Compile for Loihi hardware
# Deploy to cloud infrastructure
```

### Batch Processing

Use the GUI to process multiple models:

1. Set up templates for common configurations
2. Use file selection for batch input
3. Monitor progress across multiple operations
4. Review aggregated results

## Extensibility and Plugin System

### Adding New Tabs

1. **Create Tab Class:**
   ```python
   from tabs.base_tab import BaseTab

   class MyNewTab(BaseTab):
       def __init__(self):
           super().__init__("My New Feature")

       def create_input_section(self, layout):
           # Add custom input widgets
           self.add_text_input(layout, "Parameter:", "default_value")

       def get_command_args(self):
           # Return CLI arguments
           return ["--my-param", self.param_input.text()]
   ```

2. **Register Tab:**
   ```python
   # In main.py MainWindow.init_tabs()
   self.my_tab = MyNewTab()
   self.tab_widget.addTab(self.my_tab, "My Feature")
   ```

3. **Import Tab:**
   ```python
   # In main.py imports
   from tabs.my_new_tab import MyNewTab
   ```

### Extending CLIExecutor

Add new validation rules:

```python
def validate_my_input(self, input_type: str, value: str) -> bool:
    if input_type == 'my_custom_type':
        # Custom validation logic
        return self._validate_custom_format(value)
    return super().validate_input(input_type, value)
```

### Custom Themes

Override `apply_theme()` in MainWindow:

```python
def apply_theme(self):
    palette = QPalette()
    # Custom color scheme
    palette.setColor(QPalette.Window, QColor("#2d2d2d"))
    # ... more colors
    self.app.setPalette(palette)
```

## Deployment and Packaging

### Standalone Application

Create executable bundles:

**PyInstaller (Cross-platform):**
```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```

**Windows:**
```bash
pyinstaller --onefile --windowed --icon=icon.ico main.py
```

**macOS:**
```bash
pyinstaller --onefile --windowed --icon=icon.icns main.py
```

### System Installation

**Linux (Ubuntu/Debian):**
```bash
# Create package structure
mkdir -p neuro-compiler-gui/usr/bin
mkdir -p neuro-compiler-gui/usr/share/applications

# Copy files
cp main.py neuro-compiler-gui/usr/bin/
cp neuro-compiler.desktop neuro-compiler-gui/usr/share/applications/

# Build deb package
dpkg-deb --build neuro-compiler-gui
```

**macOS Application Bundle:**
```bash
# Use py2app
pip install py2app
python setup.py py2app
```

### Docker Container

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]
```

### CI/CD Pipeline

Example GitHub Actions workflow:

```yaml
name: Build and Release
on: [push, pull_request]

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Build executable
      run: |
        pip install pyinstaller
        pyinstaller --onefile main.py

    - name: Upload artifact
      uses: actions/upload-artifact@v2
      with:
        name: neuro-compiler-gui-${{ matrix.os }}
        path: dist/
```

## Troubleshooting

### GUI Not Launching

**Symptom:** Application starts but no window appears

**Solutions:**
1. Check display environment:
   ```bash
   echo $DISPLAY  # Linux
   # Ensure X11 forwarding for SSH
   ```

2. Verify PyQt5 installation:
   ```bash
   python -c "import PyQt5.QtWidgets; print('PyQt5 OK')"
   ```

3. Check for Qt platform plugins:
   ```bash
   # Linux
   apt-get install libxcb-xinerama0 libxcb-icccm4 libxcb-image0
   ```

### CLI Not Found

**Error:** Command execution fails with "CLI not found"

**Solutions:**
1. **Manual Path Setting:**
   ```python
   # In main.py or CLIExecutor init
   cli_executor = CLIExecutor("/path/to/neuro-compiler")
   ```

2. **Build CLI:**
   ```bash
   cargo build --release -p neuro-compiler-cli
   export PATH=$PATH:$(pwd)/target/release
   ```

3. **Verify Features:**
   ```bash
   # For simulation features
   cargo build --release --features sim-neuron -p neuro-compiler-cli
   ```

### Import Errors

**Common Issues:**

1. **Missing PyQt5:**
   ```bash
   pip install PyQt5
   # On Linux:
   apt-get install python3-pyqt5
   ```

2. **Cryptography Issues:**
   ```bash
   pip install cryptography
   # May require rust compiler for some platforms
   ```

3. **Platform-specific Issues:**
   - **macOS:** Install Qt via Homebrew if needed
   - **Windows:** Ensure Visual C++ Redistributables
   - **Linux:** Install system Qt5 development packages

### Permission Errors

**Encryption key storage issues:**

1. **Fix Permissions:**
   ```bash
   chmod 600 ~/.neuro_compiler_gui_key
   ```

2. **Change Key Location:**
   ```python
   # In CLIExecutor
   key_file = os.path.join(os.path.expanduser("~"), "Desktop", "gui_key")
   ```

### Performance Issues

**Application running slowly:**

1. **Reduce Threading:**
   ```python
   # Disable background validation
   # Modify BaseTab to run validation synchronously
   ```

2. **Memory Optimization:**
   - Clear large output buffers periodically
   - Limit history size in combo boxes

### High DPI Issues

**UI scaling problems:**

1. **Force Scaling:**
   ```python
   # In main.py NeuroCompilerGUI.__init__
   QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
   QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
   ```

2. **Qt Version Check:**
   ```bash
   python -c "import PyQt5.QtCore; print(PyQt5.QtCore.PYQT_VERSION_STR)"
   # Ensure Qt 5.14+ for best DPI support
   ```

### Network Issues

**Offline functionality not working:**

1. **Check Network Dependencies:**
   - CLI commands may require network for some operations
   - Ensure offline mode respects `--offline` flags where available

2. **Proxy Configuration:**
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

## Contributing

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function parameters and return values
- Document all public methods with docstrings
- Use meaningful variable and method names

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run with coverage
pip install pytest-cov
python -m pytest --cov=neuro_compiler_gui tests/
```

### Development Setup

1. **Clone Repository:**
   ```bash
   git clone https://github.com/neuro-compiler/gui.git
   cd neuro-compiler-gui
   ```

2. **Setup Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Development Dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Run in Development Mode:**
   ```bash
   python main.py --debug
   ```

## License

This project follows the same license as the neuro-compiler project.

## Changelog

### Version 1.0.0
- Initial release with full CLI integration
- MVC architecture implementation
- Cross-platform support (Windows, macOS, Linux)
- Security features (encryption, validation)
- Accessibility compliance
- Comprehensive documentation

## License

This project follows the same license as the neuro-compiler project.