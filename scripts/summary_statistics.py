import pandas as pd

class SummaryStatistics:
    """
    A class for calculating summary statistics for numeric columns in a DataFrame.
    """

    def __init__(self, data):
        """
        Initializes the SummaryStatistics class with a DataFrame.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        self.data = data

    def calculate_summary_statistics(self):
        """
        Calculates and returns summary statistics for all numeric columns in the dataset.

        Returns:
        - pd.DataFrame: A DataFrame containing the mean, median, standard deviation,
          min, max, and percentiles for each numeric column.
        """
        numeric_data = self.data.select_dtypes(include='number')  # Select only numeric columns
        summary_stats = numeric_data.describe().T  # Transpose for readability
        summary_stats['median'] = numeric_data.median()  # Add median
        return summary_stats
