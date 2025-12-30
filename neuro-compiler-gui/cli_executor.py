#!/usr/bin/env python3
"""
CLI Executor for Neuro-Compiler

Handles subprocess calls to the neuro-compiler CLI.
"""

import subprocess
import sys
import os
from typing import List, Optional, Dict, Any
from cryptography.fernet import Fernet
import base64


class CLIExecutor:
    """Handles execution of neuro-compiler CLI commands."""

    def __init__(self, neuro_compiler_path: Optional[str] = None):
        """
        Initialize CLI executor.

        Args:
            neuro_compiler_path: Path to the neuro-compiler CLI executable.
                                If None, assumes it's in PATH or uses cargo run.
        """
        self.neuro_compiler_path = neuro_compiler_path or self._find_neuro_compiler()
        self.encryption_key = self._load_or_generate_key()

    def _find_neuro_compiler(self) -> str:
        """Find the neuro-compiler CLI executable."""
        # First try if it's in PATH
        try:
            result = subprocess.run(['which', 'neuro-compiler'],
                                  capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            pass

        # Try cargo run if we're in the project directory
        project_root = self._find_project_root()
        if project_root:
            return f"cd {project_root} && cargo run -p neuro-compiler-cli --"

        # Fallback to assuming it's in PATH
        return "neuro-compiler"

    def _find_project_root(self) -> Optional[str]:
        """Find the neuro-compiler project root directory."""
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):
            if os.path.exists(os.path.join(current_dir, 'Cargo.toml')) and \
               os.path.exists(os.path.join(current_dir, 'crates', 'cli')):
                return current_dir
            current_dir = os.path.dirname(current_dir)
        return None

    def _load_or_generate_key(self) -> bytes:
        """Load or generate encryption key for sensitive data."""
        key_file = os.path.expanduser("~/.neuro_compiler_gui_key")
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            return key

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        f = Fernet(self.encryption_key)
        return f.encrypt(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        f = Fernet(self.encryption_key)
        return f.decrypt(encrypted_data.encode()).decode()

    def execute_command(self, command: str, args: List[str] = None,
                       cwd: Optional[str] = None,
                       env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Execute a neuro-compiler CLI command.

        Args:
            command: The CLI subcommand (e.g., 'list-targets')
            args: List of arguments for the command
            cwd: Working directory
            env: Environment variables

        Returns:
            Dict with 'success', 'stdout', 'stderr', 'returncode'
        """
        if args is None:
            args = []

        # Build the full command
        if self.neuro_compiler_path.endswith('--'):
            # Using cargo run
            full_command = f"{self.neuro_compiler_path} {command} {' '.join(args)}"
        else:
            full_command = [self.neuro_compiler_path, command] + args

        try:
            # Set up environment
            merged_env = os.environ.copy()
            if env:
                merged_env.update(env)

            # Execute the command
            if isinstance(full_command, str):
                result = subprocess.run(full_command, shell=True, capture_output=True,
                                      text=True, cwd=cwd, env=merged_env)
            else:
                result = subprocess.run(full_command, capture_output=True,
                                      text=True, cwd=cwd, env=merged_env)

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Command timed out',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }

    def validate_input(self, input_type: str, value: str) -> bool:
        """
        Validate input based on type.

        Args:
            input_type: Type of input ('file_path', 'target', 'simulator', etc.)
            value: The value to validate

        Returns:
            True if valid, False otherwise
        """
        if not value or not value.strip():
            return False

        if input_type == 'file_path':
            return os.path.exists(value) and os.path.isfile(value)
        elif input_type == 'directory':
            return os.path.exists(value) and os.path.isdir(value)
        elif input_type == 'target':
            # Basic validation - could be extended with known targets
            return bool(value.strip())
        elif input_type == 'simulator':
            valid_simulators = ['neuron', 'coreneuron', 'arbor', 'hw']
            return value in valid_simulators
        elif input_type == 'pipeline':
            # Validate comma-separated pass names
            passes = [p.strip() for p in value.split(',') if p.strip()]
            valid_passes = [
                'noop', 'validate', 'quantize4', 'quantize8', 'quantize16',
                'partition', 'placement', 'routing', 'timing', 'resource-check',
                'tn-core-mapping', 'tn-weight-programming', 'tn-crossbar-config',
                'sn-core-allocation', 'sn-aer-routing', 'sn-synapse-programming',
                'loihi-core-mapping', 'loihi-synapse-programming', 'loihi-learning-rule',
                'akida-layer-mapping', 'akida-weight-programming', 'akida-event-routing'
            ]
            return all(p in valid_passes for p in passes)
        elif input_type == 'dump_format':
            valid_formats = ['json', 'yaml', 'bin']
            formats = [f.strip() for f in value.split(',') if f.strip()]
            return all(f in valid_formats for f in formats)
        else:
            return bool(value.strip())


# Global instance
cli_executor = CLIExecutor()