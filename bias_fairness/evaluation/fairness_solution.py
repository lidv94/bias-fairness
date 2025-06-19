import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from bias_fairness.utils import timer
from bias_fairness.evaluation.base import BaseValidator

# --------------------------------------
# Class to suggest and visualize fairness solutions
# --------------------------------------
class FairnessSolution(BaseValidator):
    pass
    # def reweight_data(self):
    #     counts = self.df[self.protected].value_counts()
    #     weights = self.df[self.protected].apply(lambda x: 1.0 / counts[x])
    #     self.df['sample_weight'] = weights
    #     print("Sample weights applied for reweighting.")
    #     return self.df

    # def check_calibration(self, y_prob):
    #     prob_true, prob_pred = calibration_curve(self.df[self.target], y_prob, n_bins=10)
    #     plt.plot(prob_pred, prob_true, marker='o')
    #     plt.plot([0, 1], [0, 1], linestyle='--')
    #     plt.title("Calibration Curve")
    #     plt.xlabel("Predicted probability")
    #     plt.ylabel("True probability")
    #     plt.grid(True)
    #     plt.show()

    # def simulate_counterfactual(self, feature: str, value):
    #     modified = self.df.copy()
    #     modified[feature] = value
    #     print(f"Simulated counterfactual by setting {feature} = {value}")
    #     return modified[[feature, self.protected, self.target]].head()

    # def visualize_group_accuracy(self, y_pred):
    #     df_copy = self.df.copy()
    #     df_copy['pred'] = y_pred
    #     df_copy['correct'] = (df_copy['pred'] == df_copy[self.target]).astype(int)
    #     acc = df_copy.groupby(self.protected)['correct'].mean()
    #     acc.plot(kind='bar')
    #     plt.title("Accuracy by Protected Group")
    #     plt.ylabel("Accuracy")
    #     plt.show()
    #     return acc