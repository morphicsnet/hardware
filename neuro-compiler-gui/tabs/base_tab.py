#!/usr/bin/env python3
"""
Base Tab Class for Neuro-Compiler GUI

Provides common functionality for all GUI tabs.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QTextEdit, QProgressBar, QGroupBox, QLineEdit, QComboBox, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QFont, QKeySequence
from worker import CLIWorker, ValidationWorker
from cli_executor import cli_executor
import os


class BaseTab(QWidget):
    """Base class for all neuro-compiler GUI tabs."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.worker = None

        # Set accessibility
        self.setAccessibleName(f"{title} Tab")
        self.setAccessibleDescription(f"Interface for {title.lower()} operations")

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)

        # Input section
        self.create_input_section(layout)

        # Execute button
        self.execute_button = QPushButton(f"Execute {self.title}")
        self.execute_button.setFont(QFont("Arial", 12))
        self.execute_button.setMinimumHeight(35)
        self.execute_button.clicked.connect(self.execute_command)
        layout.addWidget(self.execute_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Output section
        self.create_output_section(layout)

        # Keyboard shortcuts
        self.setup_shortcuts()

    def create_input_section(self, layout):
        """Create input controls. Override in subclasses."""
        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout(input_group)
        input_layout.addWidget(QLabel("Override in subclass"))
        layout.addWidget(input_group)

    def create_output_section(self, layout):
        """Create output display. Override in subclasses."""
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        self.output_text.setAccessibleName("Command output")
        self.output_text.setAccessibleDescription("Displays the results of the executed command")

        output_layout.addWidget(self.output_text)
        layout.addWidget(output_group)

    def setup_connections(self):
        """Set up signal connections."""
        pass

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # Ctrl+Enter to execute
        self.execute_shortcut = QKeySequence(Qt.CTRL + Qt.Key_Return)
        # Note: Shortcuts are handled at main window level

    def execute_command(self):
        """Execute the command. Override in subclasses."""
        pass

    def validate_inputs(self) -> bool:
        """Validate input fields. Override in subclasses."""
        return True

    def get_command_args(self) -> list:
        """Get command line arguments. Override in subclasses."""
        return []

    def show_progress(self, message: str):
        """Show progress message."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setFormat(message)
        self.output_text.append(f"[PROGRESS] {message}")
        # Scroll to bottom
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)

    def hide_progress(self):
        """Hide progress bar."""
        self.progress_bar.setVisible(False)

    def show_result(self, result: dict):
        """Display command result."""
        self.hide_progress()

        if result['success']:
            self.output_text.append("[SUCCESS] Command completed")
            if result['stdout']:
                self.output_text.append("Output:")
                self.output_text.append(result['stdout'])
        else:
            self.output_text.append("[ERROR] Command failed")
            if result['stderr']:
                self.output_text.append("Error:")
                self.output_text.append(result['stderr'])
            else:
                self.output_text.append("No error details available")

        # Scroll to bottom
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)

    def show_error(self, error_msg: str):
        """Display error message."""
        self.hide_progress()
        self.output_text.append(f"[ERROR] {error_msg}")
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)

    def clear_output(self):
        """Clear the output text."""
        self.output_text.clear()

    def set_worker_callbacks(self, worker: CLIWorker):
        """Set up worker signal connections."""
        worker.progress.connect(self.show_progress)
        worker.finished.connect(self.show_result)
        worker.error.connect(self.show_error)

    def run_command(self, command: str, args: list = None):
        """Run a CLI command in background thread."""
        if self.worker and self.worker.isRunning():
            self.show_error("A command is already running. Please wait.")
            return

        self.worker = CLIWorker(command, args or [])
        self.set_worker_callbacks(self.worker)
        self.worker.start()

    @pyqtSlot()
    def on_execute_button_clicked(self):
        """Handle execute button click."""
        if not self.validate_inputs():
            self.show_error("Invalid input parameters. Please check your inputs.")
            return

        self.clear_output()
        args = self.get_command_args()
        if args is not None:
            self.run_command(self.get_command_name(), args)

    def get_command_name(self) -> str:
        """Get the CLI command name. Override in subclasses."""
        return self.title.lower().replace(" ", "-")

    def add_file_selector(self, layout, label_text: str, button_text: str = "Browse...",
                         file_filter: str = "All Files (*)") -> tuple:
        """Add a file selector widget."""
        h_layout = QHBoxLayout()

        label = QLabel(label_text)
        h_layout.addWidget(label)

        line_edit = QLineEdit()
        line_edit.setAccessibleName(f"{label_text} path")
        h_layout.addWidget(line_edit)

        button = QPushButton(button_text)
        button.clicked.connect(lambda: self.browse_file(line_edit, file_filter))
        h_layout.addWidget(button)

        layout.addLayout(h_layout)
        return line_edit, button

    def add_directory_selector(self, layout, label_text: str, button_text: str = "Browse...") -> tuple:
        """Add a directory selector widget."""
        h_layout = QHBoxLayout()

        label = QLabel(label_text)
        h_layout.addWidget(label)

        line_edit = QLineEdit()
        line_edit.setAccessibleName(f"{label_text} directory")
        h_layout.addWidget(line_edit)

        button = QPushButton(button_text)
        button.clicked.connect(lambda: self.browse_directory(line_edit))
        h_layout.addWidget(button)

        layout.addLayout(h_layout)
        return line_edit, button

    def browse_file(self, line_edit, file_filter: str = "All Files (*)"):
        """Browse for a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter)
        if file_path:
            line_edit.setText(file_path)

    def browse_directory(self, line_edit):
        """Browse for a directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            line_edit.setText(dir_path)

    def add_text_input(self, layout, label_text: str, placeholder: str = "") -> 'QLineEdit':
        """Add a text input field."""
        h_layout = QHBoxLayout()

        label = QLabel(label_text)
        h_layout.addWidget(label)

        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setAccessibleName(label_text)
        h_layout.addWidget(line_edit)

        layout.addLayout(h_layout)
        return line_edit

    def add_combo_box(self, layout, label_text: str, items: list) -> 'QComboBox':
        """Add a combo box."""
        h_layout = QHBoxLayout()

        label = QLabel(label_text)
        h_layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(items)
        combo.setAccessibleName(label_text)
        h_layout.addWidget(combo)

        layout.addLayout(h_layout)
        return combo