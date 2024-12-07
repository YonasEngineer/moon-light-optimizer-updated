import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class TimeSeriesAnalysis:
    def __init__(self, dataframe):
        """
        Initialize the TimeSeriesAnalysis class with a dataframe.
        """
        self.dataframe = dataframe

    def plot_time_series(self, columns):
        """
        Plot time series for the specified columns using seaborn (line charts).
        
        Args:
            columns (list): List of column names to plot.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        
        for column in columns:
            if column in self.dataframe:
                sns.lineplot(
                    x=self.dataframe.index, 
                    y=self.dataframe[column], 
                    label=column
                )
            else:
                print(f"Column '{column}' not found in the dataframe.")
        
        plt.title("Time Series Data", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Values", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_monthly_patterns(self, columns):
        """
        Plot monthly trends for specified columns using bar charts.
        
        Args:
            columns (list): List of columns to plot.
        """
        self.dataframe['month'] = self.dataframe.index.month
        plt.figure(figsize=(14, 8))

        for column in columns:
            if column in self.dataframe:
                monthly_avg = self.dataframe.groupby('month')[column].mean()
                sns.barplot(x=monthly_avg.index, y=monthly_avg.values, label=column)
            else:
                print(f"Column '{column}' not found in the dataframe.")

        plt.title("Monthly Patterns", fontsize=16)
        plt.xlabel("Month", fontsize=14)
        plt.ylabel("Average Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_daily_trends(self, columns):
        """
        Plot daily trends for specified columns, aggregated by hour.
        
        Args:
            columns (list): List of columns to plot.
        """
        self.dataframe['hour'] = self.dataframe.index.hour
        plt.figure(figsize=(14, 8))

        for column in columns:
            if column in self.dataframe:
                hourly_avg = self.dataframe.groupby('hour')[column].mean()
                sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, label=column)
            else:
                print(f"Column '{column}' not found in the dataframe.")

        plt.title("Daily Trends (Hourly)", fontsize=16)
        plt.xlabel("Hour of the Day", fontsize=14)
        plt.ylabel("Average Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_anomalies(self, columns, threshold=2):
        """
        Identify and plot anomalies where values exceed a given threshold.

        Args:
            columns (list): List of columns to check for anomalies.
            threshold (float): Threshold for anomaly detection.
        """
        plt.figure(figsize=(14, 8))

        for column in columns:
            if column in self.dataframe:
                anomalies = self.dataframe[self.dataframe[column] > threshold]
                sns.scatterplot(x=anomalies.index, y=anomalies[column], label=f'Anomalies in {column}')
            else:
                print(f"Column '{column}' not found in the dataframe.")

        plt.title("Anomalies Detection", fontsize=16)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Anomalous Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()


