import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import calibration_curve
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference, selection_rate
from bias_fairness.utils import timer
from bias_fairness.evaluation.base import BaseValidator

# --------------------------------------
# Class to detect different bias metrics
# --------------------------------------
class DetectBias(BaseValidator):
    pass
    # def demographic_parity(self):
    #     result = demographic_parity_difference(self.df[self.target], self.df[self.target], sensitive_features=self.df[self.protected])
    #     print(f"Demographic Parity Difference: {result:.4f}")
    #     return result

    # def equalized_odds(self, y_pred):
    #     result = equalized_odds_difference(self.df[self.target], y_pred, sensitive_features=self.df[self.protected])
    #     print(f"Equalized Odds Difference: {result:.4f}")
    #     return result

    # def equal_opportunity(self, y_pred):
    #     cmf = MetricFrame(metrics="true_positive_rate", y_true=self.df[self.target], y_pred=y_pred, sensitive_features=self.df[self.protected])
    #     print("Equal Opportunity (True Positive Rate by group):")
    #     print(cmf.by_group)
    #     return cmf.by_group

    # def confusion_matrix_plot(self, y_pred):
    #     cm = confusion_matrix(self.df[self.target], y_pred)
    #     ConfusionMatrixDisplay(cm).plot()
    #     plt.title("Confusion Matrix")
    #     plt.show()



