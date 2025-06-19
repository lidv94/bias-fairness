import pandas as pd
import numpy as np
from bias_fairness.utils import timer

# --------------------------------------
# Base class to validate inputs
# --------------------------------------
class BaseValidator:
    def __init__(self, df: pd.DataFrame, target: str, protected: str):
        self.df = df.copy()
        self.target = target
        self.protected = protected
        self._validate_inputs()

    def _validate_inputs(self):
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in DataFrame")
        if self.protected not in self.df.columns:
            raise ValueError(f"Protected column '{self.protected}' not found in DataFrame")

