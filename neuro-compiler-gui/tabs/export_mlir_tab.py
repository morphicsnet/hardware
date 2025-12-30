from .base_tab import BaseTab

class ExportMlirTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__("Export MLIR", parent)