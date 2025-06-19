import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bias_fairness.utils import timer
from bias_fairness.evaluation.base import BaseValidator
from scipy.stats import chi2_contingency

# --------------------------------------
# Class to prepare and analyze data
# --------------------------------------
class AnalyzeData(BaseValidator):
    def prevalence_plot(self):
        '''
        check for data imbalance across protected groups
        '''
        prevalence = self.df[self.protected].value_counts(normalize=True)
        sns.barplot(x=prevalence.index, y=prevalence.values)
        plt.title("Prevalence by Protected Class")
        plt.ylabel("Proportion")
        plt.show()

    def predicted_prevalence(self, y_pred):
        df = self.df.copy()
        df['y_pred'] = y_pred
        pred_prev = df.groupby(self.protected)['y_pred'].mean()
        
        sns.barplot(x=pred_prev.index, y=pred_prev.values)
        plt.title("Predicted Prevalence by Protected Group")
        plt.ylabel("P(Ŷ = 1 | group)")
        plt.show()
        return pred_prev  
        
    def true_prevalence(self, y_true):
        df = self.df.copy()
        df['y_true'] = y_true
        true_prev = df.groupby(self.protected)['y_true'].mean()
    
        sns.barplot(x=true_prev.index, y=true_prev.values)
        plt.title("True Outcome Prevalence by Protected Group")
        plt.ylabel("P(Y = 1 | group)")
        plt.show()
        return true_prev
        
    def disparity_summary(self, y_true, y_pred):
        df = self.df.copy()
        df['y_true'] = y_true
        df['y_pred'] = y_pred
    
        summary = df.groupby(self.protected).agg(
            prevalence=('y_true', 'mean'),
            predicted_prevalence=('y_pred', 'mean'),
            count=('y_true', 'count')
        )
        summary['disparity'] = summary['predicted_prevalence'] / summary['predicted_prevalence'].min()
    
        print("Disparity Summary Table:")
        print(summary)
        return summary


    def detect_proxy_variables(self, threshold=0.8):
        correlations = self.df.corr(numeric_only=True)[self.protected].drop(self.protected, errors='ignore')
        proxies = correlations[correlations.abs() > threshold].sort_values(ascending=False)
        print("Possible Proxy Variables:")
        print(proxies)
        return proxies

# def cramers_v(x, y):
#     """Calculate Cramér's V statistic for categorical-categorical association."""
#     confusion_matrix = pd.crosstab(x, y)
#     chi2, p, dof, expected = chi2_contingency(confusion_matrix)

#     n = confusion_matrix.sum().sum()
#     phi2 = chi2 / n
#     r, k = confusion_matrix.shape
#     phi2_corrected = max(0, phi2 - ((k-1)*(r-1))/(n-1))  # Bias correction
#     r_corrected = r - ((r-1)**2)/(n-1)
#     k_corrected = k - ((k-1)**2)/(n-1)

#     return np.sqrt(phi2_corrected / min((k_corrected-1), (r_corrected-1)))