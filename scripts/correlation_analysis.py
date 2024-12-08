import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationAnalysis:
    """
    A class for creating various visualizations, such as correlation matrices,
    pair plots, and scatter plots for analyzing wind and solar data.
    """

    def __init__(self, data):
        """
        Initializes the Visualization class with a DataFrame.
        
        Parameters:
        - data (pd.DataFrame): The DataFrame containing the data to analyze.
        """
        self.data = data

    def plot_correlation_matrix(self, columns):
        """
        Plots a correlation matrix as a heatmap for the given columns.
        
        Parameters:
        - columns (list of str): List of column names to include in the correlation matrix.
        """
        corr_matrix = self.data[columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix", fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_pairplot(self, columns):
        """
        Plots a pairplot (scatter matrix) for the given columns.
        
        Parameters:
        - columns (list of str): List of column names to include in the pair plot.
        """
        sns.pairplot(self.data[columns], diag_kind="kde", corner=True)
        plt.suptitle("Pair Plot for Selected Variables", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_wind_vs_solar(self, wind_columns, solar_columns):
        """
        Plots scatter plots of wind conditions (e.g., WS, WSgust, WD)
        against solar irradiance (e.g., GHI, DNI, DHI).
        
        Parameters:
        - wind_columns (list of str): List of wind-related column names.
        - solar_columns (list of str): List of solar-related column names.
        """
        plt.figure(figsize=(14, 10))
        for wind in wind_columns:
            for solar in solar_columns:
                plt.scatter(self.data[wind], self.data[solar], alpha=0.5, label=f'{wind} vs {solar}')
                plt.xlabel(wind)
                plt.ylabel(solar)
                plt.title(f'Scatter Plot: {wind} vs {solar}')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()


