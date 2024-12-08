import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from windrose import WindroseAxes
import matplotlib.cm as cm

class WeatherAnalysis:
    """
    A class for performing wind analysis using wind roses and radial plots.
    """

    def __init__(self, data):
        """
        Initializes the WindAnalysis class with a DataFrame.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing wind speed ('WS') and wind direction ('WD') columns.
        """
        self.data = data

    def plot_wind_rose(self, wind_speed_col='WS', wind_dir_col='WD', bins=None, cmap='coolwarm'):
        """
        Plots a wind rose to show the distribution of wind speed and direction.

        Parameters:
        - wind_speed_col (str): The column name for wind speed.
        - wind_dir_col (str): The column name for wind direction.
        - bins (list): Custom bins for wind speed intervals (optional).
        - cmap (str): Colormap for the plot (default: 'coolwarm').
        """
        # Default bins if not provided
        if bins is None:
            bins = [0, 1, 2, 3, 5, 10, 15, 20]

        # Drop NaN values
        wind_data = self.data[[wind_speed_col, wind_dir_col]].dropna()

        # Ensure cmap is a colormap object
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)

        # Create a wind rose plot
        ax = WindroseAxes.from_ax()
        ax.bar(
            wind_data[wind_dir_col], 
            wind_data[wind_speed_col],
            normed=True, 
            bins=bins, 
            cmap=cmap
        )
        ax.set_legend()
        plt.title("Wind Rose", fontsize=16)
        plt.show()

    def plot_radial_bar(self, wind_speed_col='WS', wind_dir_col='WD', num_bins=8):
        """
        Plots a radial bar chart showing the average wind speed per direction sector.

        Parameters:
        - wind_speed_col (str): The column name for wind speed.
        - wind_dir_col (str): The column name for wind direction.
        - num_bins (int): The number of bins (sectors) to divide wind directions into.
        """
        # Drop NaN values
        wind_data = self.data[[wind_speed_col, wind_dir_col]].dropna()

        # Bin wind direction into sectors
        bin_edges = np.linspace(0, 360, num_bins + 1)
        wind_data['sector'] = pd.cut(
            wind_data[wind_dir_col], 
            bins=bin_edges, 
            labels=[f"{int(bin_edges[i])}-{int(bin_edges[i + 1])}" for i in range(num_bins)],
            right=False
        )

        # Calculate average wind speed for each sector
        sector_avg = wind_data.groupby('sector')[wind_speed_col].mean()

        # Plot radial bar chart
        angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
        values = sector_avg.values

        # Close the plot
        angles = np.concatenate((angles, [angles[0]]))
        values = np.concatenate((values, [values[0]]))

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
        ax.fill(angles, values, color='blue', alpha=0.5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(sector_avg.index, fontsize=10)
        ax.set_title("Radial Bar Plot of Wind Speed by Direction", fontsize=16)
        plt.show()

    def analyze_temperature(self, temp_col='Temperature', rh_col='RH', solar_rad_col='SolarRadiation'):
        # Drop NaN values for the analysis
        temp_data = self.data[[temp_col, rh_col, solar_rad_col]].dropna()

        # Scatter plot with color representing Solar Radiation
        scatter = plt.scatter(temp_data[rh_col], temp_data[temp_col], c=temp_data[solar_rad_col], cmap='viridis')
        
        # Add labels and colorbar
        plt.xlabel("Relative Humidity (%)")
        plt.ylabel("Temperature (°C)")
        plt.colorbar(scatter, label="Solar Radiation (GHI)")  # Colorbar based on Solar Radiation

        plt.show()
        
        # Calculate and print correlation matrix
        corr_matrix = temp_data.corr()
        print("Correlation Matrix:")
        print(corr_matrix)
    def plot_histograms(self, columns):
        import seaborn as sns

        """
        This function will plot histograms for the given columns.
        
        :param columns: List of columns to plot histograms for.
        """
        # Set up the matplotlib figure
        plt.figure(figsize=(12, 8))
        
        for i, col in enumerate(columns, 1):
            plt.subplot(2, 3, i)  # 2 rows, 3 columns of subplots
            sns.histplot(self.data[col], kde=True, bins=30, color='skyblue', edgecolor='black')
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel('Frequency')

        plt.tight_layout()  # Adjust subplots to fit into figure area.
        plt.show()
    def calculate_z_scores(self, columns, threshold=3):
        """
        Calculate Z-scores for the specified columns and flag data points
        that deviate significantly from the mean.
        
        :param columns: List of columns to calculate Z-scores for.
        :param threshold: Z-score threshold for flagging significant deviations.
                          Default is 3 (corresponds to approximately 99.7% of data in a normal distribution).
        :return: A dictionary with column names as keys and flagged rows as values.
        """
        flagged_data = {}
        
        for col in columns:
            if col not in self.data.columns:
                print(f"Warning: {col} is not in the dataset.")
                continue
            
            # Calculate Z-scores
            mean = self.data[col].mean()
            std = self.data[col].std()
            self.data[f"{col}_zscore"] = (self.data[col] - mean) / std
            
            # Flag rows exceeding the threshold
            flagged = self.data[np.abs(self.data[f"{col}_zscore"]) > threshold]
            flagged_data[col] = flagged
            
            print(f"Processed Z-scores for column '{col}': Found {len(flagged)} flagged points.")
        
        return flagged_data