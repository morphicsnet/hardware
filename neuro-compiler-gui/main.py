#!/usr/bin/env python3
"""
Neuro-Compiler GUI Application

A modern PyQt5-based desktop application for the neuro-compiler toolchain.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QStatusBar, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor

from tabs.list_targets_tab import ListTargetsTab
from tabs.import_tab import ImportTab
from tabs.lower_tab import LowerTab
from tabs.compile_tab import CompileTab
from tabs.simulate_tab import SimulateTab
from tabs.profile_tab import ProfileTab
from tabs.package_tab import PackageTab
from tabs.deploy_tab import DeployTab
from tabs.export_mlir_tab import ExportMlirTab
from tabs.run_tab import RunTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro-Compiler GUI")
        self.setGeometry(100, 100, 1200, 800)

        # Set up status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Initialize tabs
        self.init_tabs()

        # Apply theme
        self.apply_theme()

        # Set accessibility properties
        self.setAccessibleName("Neuro Compiler Main Window")
        self.setAccessibleDescription("Main window for neuro-compiler GUI application")

    def init_tabs(self):
        """Initialize all functionality tabs."""
        # List Targets Tab
        self.list_targets_tab = ListTargetsTab()
        self.tab_widget.addTab(self.list_targets_tab, "List Targets")

        # Import Tab
        self.import_tab = ImportTab()
        self.tab_widget.addTab(self.import_tab, "Import")

        # Lower Tab
        self.lower_tab = LowerTab()
        self.tab_widget.addTab(self.lower_tab, "Lower")

        # Compile Tab
        self.compile_tab = CompileTab()
        self.tab_widget.addTab(self.compile_tab, "Compile")

        # Simulate Tab
        self.simulate_tab = SimulateTab()
        self.tab_widget.addTab(self.simulate_tab, "Simulate")

        # Profile Tab
        self.profile_tab = ProfileTab()
        self.tab_widget.addTab(self.profile_tab, "Profile")

        # Package Tab
        self.package_tab = PackageTab()
        self.tab_widget.addTab(self.package_tab, "Package")

        # Deploy Tab
        self.deploy_tab = DeployTab()
        self.tab_widget.addTab(self.deploy_tab, "Deploy")

        # Export MLIR Tab
        self.export_mlir_tab = ExportMlirTab()
        self.tab_widget.addTab(self.export_mlir_tab, "Export MLIR")

        # Run Tab
        self.run_tab = RunTab()
        self.tab_widget.addTab(self.run_tab, "Run")

    def apply_theme(self):
        """Apply clean, professional theme."""
        app = QApplication.instance()
        palette = QPalette()

        # Set colors for a clean, professional look
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(248, 248, 248))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

        app.setPalette(palette)

        # Set font
        if sys.platform == "win32":
            font = QFont("Segoe UI", 10)
        elif sys.platform == "darwin":  # macOS
            font = QFont("Helvetica Neue", 10)
        else:  # Linux
            font = QFont("DejaVu Sans", 10)
        app.setFont(font)

    def show_error_dialog(self, title, message):
        """Show error dialog."""
        QMessageBox.critical(self, title, message)


class NeuroCompilerGUI(QApplication):
    def __init__(self, argv):
        # Set high DPI scaling before creating QApplication
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        super().__init__(argv)
        self.setApplicationName("Neuro-Compiler GUI")
        self.setApplicationVersion("1.0.0")
        self.setOrganizationName("Neuro-Compiler Project")

        self.main_window = MainWindow()
        self.main_window.show()


def main():
    print("Starting Neuro-Compiler GUI...")
    app = NeuroCompilerGUI(sys.argv)
    print("GUI application created and window shown. Starting event loop...")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())