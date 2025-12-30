#!/usr/bin/env python3
"""
List Targets Tab for Neuro-Compiler GUI

Displays available compilation targets.
"""

from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGroupBox
from .base_tab import BaseTab


class ListTargetsTab(BaseTab):
    """Tab for listing available compilation targets."""

    def __init__(self, parent=None):
        super().__init__("List Targets", parent)

    def create_input_section(self, layout):
        """No input parameters needed for list-targets."""
        info_label = QLabel("This command lists all available compilation targets from the targets/ directory.\nNo input parameters required.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

    def get_command_name(self) -> str:
        """Return the CLI command name."""
        return "list-targets"

    def get_command_args(self) -> list:
        """No arguments needed for list-targets."""
        return []

    def validate_inputs(self) -> bool:
        """Always valid since no inputs."""
        return True