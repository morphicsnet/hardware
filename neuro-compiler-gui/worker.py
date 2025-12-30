#!/usr/bin/env python3
"""
Worker Thread for Neuro-Compiler GUI

Handles background execution of CLI commands using QThread.
"""

from PyQt5.QtCore import QThread, pyqtSignal
from typing import Dict, Any, List, Optional
from cli_executor import cli_executor


class CLIWorker(QThread):
    """Worker thread for executing CLI commands."""

    # Signals
    finished = pyqtSignal(dict)  # Emitted when command completes
    progress = pyqtSignal(str)   # Emitted for progress updates
    error = pyqtSignal(str)      # Emitted on errors

    def __init__(self, command: str, args: List[str] = None,
                 cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None):
        super().__init__()
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.env = env

    def run(self):
        """Execute the CLI command in background."""
        try:
            self.progress.emit(f"Executing: {self.command} {' '.join(self.args)}")

            result = cli_executor.execute_command(
                self.command, self.args, self.cwd, self.env
            )

            if result['success']:
                self.progress.emit("Command completed successfully")
            else:
                self.progress.emit(f"Command failed with return code {result['returncode']}")

            self.finished.emit(result)

        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            self.error.emit(error_msg)
            self.finished.emit({
                'success': False,
                'stdout': '',
                'stderr': error_msg,
                'returncode': -1
            })


class ValidationWorker(QThread):
    """Worker thread for input validation."""

    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, input_type: str, value: str):
        super().__init__()
        self.input_type = input_type
        self.value = value

    def run(self):
        """Validate input."""
        try:
            is_valid = cli_executor.validate_input(self.input_type, self.value)
            message = "Valid" if is_valid else "Invalid input"
            self.finished.emit(is_valid, message)
        except Exception as e:
            self.finished.emit(False, f"Validation error: {str(e)}")


class RealTimeUpdateWorker(QThread):
    """Worker thread for real-time status updates."""

    update = pyqtSignal(str)  # Emitted with status message

    def __init__(self, update_interval: int = 1000):
        super().__init__()
        self.update_interval = update_interval
        self.running = True

    def run(self):
        """Continuously update status."""
        import time

        while self.running:
            # Example: Check if neuro-compiler is available
            try:
                result = cli_executor.execute_command('list-targets')
                if result['success']:
                    self.update.emit("Neuro-compiler CLI available")
                else:
                    self.update.emit("Neuro-compiler CLI not available")
            except:
                self.update.emit("Checking CLI availability...")

            self.msleep(self.update_interval)

    def stop(self):
        """Stop the update loop."""
        self.running = False