from .base_tab import BaseTab

class PackageTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__("Package", parent)