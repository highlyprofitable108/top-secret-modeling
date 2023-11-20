import os
import mpld3
import logging
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta, date as date_type

matplotlib.use('Agg')


class SimVisualization:
    def __init__(self, template_dir=None, target_variable="odds.spread_close", static_dir="./flask_app/static"):
        """
        Initializes the simulation visualization class.

        Args:
            template_dir (str, optional): Directory for storing templates.
            target_variable (str): The target variable for the simulation.
            static_dir (str): Directory for storing static files.
        """
        self.template_dir = template_dir
        self.static_dir = static_dir
        self.TARGET_VARIABLE = target_variable

    def compute_confidence_interval(self, data, confidence=0.95):
        """
        Computes the confidence interval for a given dataset.

        Args:
            data (list or np.array): Dataset to compute the confidence interval for.
            confidence (float): Confidence level for the interval.

        Returns:
            tuple: Lower and upper bounds of the confidence interval.
        """
        logging.info("Starting confidence interval computation.")

        a = 1.0 * np.array(data)
        n = len(a)
        mean, se = np.mean(a), np.std(a) / np.sqrt(n)  # Standard error calculation
        h = se * norm.ppf((1 + confidence) / 2.)  # Margin of error

        confidence_interval = (mean - h, mean + h)
        logging.info(f"Computed confidence interval: {confidence_interval}")

        return confidence_interval

    def filter_simulation_results(self, simulation_results):
        """
        Filters simulation results to remove outliers and NaN values.

        Args:
            simulation_results (list or np.array): Simulation results to be filtered.

        Returns:
            list: Filtered simulation results.
        """
        logging.info("Starting filtering of simulation results.")

        # Handling empty or invalid simulation results
        if simulation_results is None or (isinstance(simulation_results, np.ndarray) and simulation_results.size <= 1):
            logging.info(f"Simulation results are empty or contain only a single value: {simulation_results}")
            return []

        # Replace NaN values with zero
        simulation_results = np.nan_to_num(simulation_results)

        if simulation_results.size == 0:
            logging.info("Simulation results are empty.")
            return []

        # Define bounds for filtering
        LOWER_PERCENTILE = 0.0
        UPPER_PERCENTILE = 1

        # Calculate bounds and filter
        lower_bound = np.percentile(simulation_results, LOWER_PERCENTILE * 100)
        upper_bound = np.percentile(simulation_results, UPPER_PERCENTILE * 100)
        filtered_results = [result for result in simulation_results if lower_bound <= result <= upper_bound]

        return filtered_results

    def analyze_simulation_results(self, simulation_results):
        """
        Analyzes simulation results to compute the range of outcomes, standard deviation, and confidence interval.

        Args:
            simulation_results (list or np.array): Simulation results to be analyzed.

        Returns:
            tuple: Range of outcomes, standard deviation, and confidence interval.
        """
        logging.info("Starting analysis of simulation results.")

        filtered_results = self.filter_simulation_results(simulation_results)
        if not filtered_results:
            logging.info("No data available after filtering simulation results.")
            return None, None, None

        # Compute range, standard deviation, and confidence interval
        range_of_outcomes = (min(filtered_results), max(filtered_results))
        std_deviation = np.std(filtered_results)
        confidence_interval = self.compute_confidence_interval(filtered_results)

        logging.info(f"Range of outcomes: {range_of_outcomes}")
        logging.info(f"Standard deviation: {std_deviation}")
        logging.info(f"Confidence interval: {confidence_interval}")

        return range_of_outcomes, std_deviation, confidence_interval

    def format_records(self, records):
        """
        Formats betting records into a human-readable DataFrame.

        Args:
            records (pd.Series): Series containing win, loss, and push counts.

        Returns:
            pd.DataFrame: Formatted DataFrame with win percentage and counts.
        """
        # Reindex and calculate win percentage
        records = records.reindex(['win', 'loss', 'push'], fill_value=0)
        win_pct = records['win'] / (records['win'] + records['loss']) if (records['win'] + records['loss']) > 0 else 0
        win_pct = "{:.2f}%".format(win_pct * 100)  # Format as percentage

        # Create DataFrame for display
        records_df = records.to_frame().reset_index()
        records_df.columns = ['Bet Outcome', 'Count']
        win_pct_row = pd.DataFrame([['win pct', win_pct]], columns=['Bet Outcome', 'Count'])

        return pd.concat([records_df, win_pct_row], ignore_index=True)

    def prepare_data(self, historical_df, get_current):
        """
        Prepares data frames for visualization based on historical data.

        Args:
            historical_df (pd.DataFrame): DataFrame containing historical data.
            get_current (bool): Flag to determine the type of data preparation.

        Returns:
            tuple: Historical DataFrame and an empty results DataFrame.
        """
        historical_df = historical_df.reset_index(drop=True)

        # Define different column sets based on the 'get_current' flag
        if get_current is True:
            results_df = pd.DataFrame(columns=[
                'Date', 'Home Team', 'Vegas Odds', 'Modeling Odds', 'Away Team', 'Recommended Bet', 'Expected Value (%)'
            ])
        else:
            results_df = pd.DataFrame(columns=[
                'Date', 'Home Team', 'Vegas Odds', 'Modeling Odds', 'Away Team', 'Recommended Bet',
                'Home Points', 'Away Points', 'Result with Spread', 'Actual Covered', 'Bet Outcome', 'Actual Value ($)'
            ])
        return historical_df, results_df

    def round_to_nearest_half(self, value):
        """
        Rounds a value to the nearest half.

        Args:
            value (float): Value to round.

        Returns:
            float: Value rounded to the nearest half.
        """
        rounded = round(value * 2) / 2
        result = rounded if value % 0.5 > 0.25 else round(value)
        return result

    def create_custom_percentiles(self):
        """
        Creates a custom distribution of percentiles with more focus around the center.

        Returns:
            np.array: Array of percentiles.
        """
        logging.info("Creating custom percentiles.")
        close_to_center = np.linspace(30, 70, num=10)
        extreme_lower = np.linspace(0, 30, num=5)
        extreme_upper = np.linspace(70, 100, num=5)
        all_percentiles = np.sort(np.concatenate([extreme_lower, close_to_center, extreme_upper]))
        logging.info(f"Custom percentiles created: {all_percentiles}")
        return all_percentiles

    def estimate_win_probability_based_on_spread_diff(self, predicted_spread, vegas_line, df):
        """
        Estimates the win probability based on the difference between the predicted spread and Vegas line.

        Args:
            predicted_spread (float): The predicted spread.
            vegas_line (float): The Vegas line spread.
            df (pd.DataFrame): DataFrame containing historical data.

        Returns:
            float: Estimated win probability.
        """
        logging.info(f"Estimating win probability for predicted spread: {predicted_spread}, Vegas line: {vegas_line}")

        df['Spread Difference'] = df['Modeling Odds'] - df['Vegas Odds']
        df['Spread Difference Rounded'] = df['Spread Difference'].apply(lambda x: self.round_to_nearest_half(x) if not pd.isna(x) else x)
        predicted_spread_rounded = self.round_to_nearest_half(predicted_spread)
        vegas_line_rounded = self.round_to_nearest_half(vegas_line)
        spread_diff = predicted_spread_rounded - vegas_line_rounded
        logging.info(f"Calculated spread difference: {spread_diff}")

        custom_percentiles = self.create_custom_percentiles()
        quantiles = np.percentile(df['Spread Difference Rounded'].dropna(), custom_percentiles)  # Drop NaN before calculating percentiles
        unique_quantiles = np.unique(quantiles)
        extended_bins = np.concatenate(([-np.inf], unique_quantiles, [np.inf]))
        labels = [f"{extended_bins[i]} to {extended_bins[i+1]}" for i in range(len(extended_bins) - 1)]
        df['Spread Diff Bin'] = pd.cut(df['Spread Difference Rounded'], bins=extended_bins, labels=labels)
        win_rates = df.groupby('Spread Diff Bin', observed=True)['Bet Outcome'].apply(lambda x: (x == 'win').sum() / len(x) if len(x) > 0 else 0)

        # Safely determine the bin label for current spread difference
        digitized_index = np.digitize([spread_diff], extended_bins)[0] - 1
        current_bin_label = labels[digitized_index] if digitized_index < len(labels) else None
        P_win = win_rates.get(current_bin_label, 0) if current_bin_label is not None else 0

        logging.info(f"Estimated win probability: {P_win}")
        return P_win

    def calculate_ev(self, predicted_spread, vegas_line):
        """
        Calculates the Expected Value (EV) for a bet based on predicted spread and Vegas line.

        Args:
            predicted_spread (float): The predicted spread.
            vegas_line (float): The Vegas line spread.

        Returns:
            float: Calculated Expected Value.
        """
        logging.info(f"Calculating EV: Predicted Spread: {predicted_spread}, Vegas Line: {vegas_line}")

        # Load historical data
        historical_df = pd.read_csv(os.path.join(self.static_dir, 'historical.csv'))
        logging.info("Historical data loaded for EV calculation.")

        # Estimate win probability based on spread difference
        P_win = self.estimate_win_probability_based_on_spread_diff(predicted_spread, vegas_line, historical_df)
        logging.info(f"Estimated win probability: {P_win}")

        # Check for NaN in win probability
        if pd.isna(P_win):
            logging.error("NaN found in win probability calculation.")
            return None

        # Calculate loss probability and expected value
        P_loss = 1 - P_win
        expected_value = (P_win * 100) - (P_loss * 110)  # Assumes -110 odds
        logging.info(f"Calculated Expected Value: {expected_value}")

        # Check for NaN in expected value
        if pd.isna(expected_value):
            logging.error("NaN found in expected value calculation.")
            return None

        return expected_value

    def process_row(self, row, simulation_results, idx, get_current):
        """
        Process a row of data, calculate betting recommendations and outcomes, and return the results.

        Args:
            row (pd.Series): A row of data containing game information.
            simulation_results (np.ndarray): Array of simulation results.
            idx (int): Index to access the corresponding simulation result.
            get_current (bool): Whether to process for current or historical data.

        Returns:
            dict: A dictionary containing processed game information.
        """
        logging.info(f"Processing row {idx}.")

        # Extract game information from the row
        actual_home_points = row.get('summary.home.points')
        actual_away_points = row.get('summary.away.points')
        home_team = row.get('summary.home.name')
        away_team = row.get('summary.away.name')
        vegas_line = row.get("summary." + self.TARGET_VARIABLE)
        date = row.get('scheduled')
        predicted_difference = simulation_results[idx]

        logging.info(f"Extracted game info for row {idx}: Home Points: {actual_home_points}, Away Points: {actual_away_points}, Vegas Line: {vegas_line}, Date: {date}")

        # Convert 'date' to a datetime object if it's not already
        if isinstance(date, date_type) and not isinstance(date, datetime):
            date = datetime.combine(date, datetime.min.time())

        # Initialize variables
        recommended_bet = None
        actual_covered = None
        bet_outcome = None
        actual_value = None
        actual_difference = None

        # Skip processing for future games with no odds and no results
        if get_current and pd.isna(vegas_line) and pd.isna(actual_home_points) and pd.isna(actual_away_points):
            logging.info(f"Skipping row {idx} as it's a future game with no odds and no results.")
            return {}

        # Process for games with odds but no results (Future Games)
        if get_current and not pd.isna(vegas_line) and pd.isna(actual_home_points) and pd.isna(actual_away_points):
            logging.info(f"Mean: {np.mean(predicted_difference)}")
            logging.info(f"Vegas Line: {vegas_line}")

            recommendation_calc = vegas_line - np.mean(predicted_difference)
            recommended_bet = self.determine_recommended_bet(recommendation_calc)
            logging.info(f"Recommended Bet: {recommended_bet}")

            expected_value = self.calculate_ev(np.mean(predicted_difference), vegas_line)
            return {
                'Date': date,
                'Home Team': home_team,
                'Vegas Odds': vegas_line,
                'Modeling Odds': np.median(predicted_difference),
                'Away Team': away_team,
                'Recommended Bet': recommended_bet,
                'Expected Value (%)': expected_value
            }

        # Process for historical games (calculate everything except EV)
        if not get_current:
            actual_difference = (actual_home_points + vegas_line) - actual_away_points if not pd.isna(actual_home_points) and not pd.isna(vegas_line) else None
            recommendation_calc = vegas_line - np.mean(predicted_difference) if not pd.isna(vegas_line) else None
            recommended_bet = self.determine_recommended_bet(recommendation_calc) if recommendation_calc is not None else None
            actual_covered, bet_outcome, actual_value = self.determine_bet_outcome(actual_difference, recommended_bet)

            return {
                'Date': date,
                'Home Team': home_team,
                'Vegas Odds': vegas_line,
                'Modeling Odds': np.median(predicted_difference),
                'Away Team': away_team,
                'Recommended Bet': recommended_bet,
                'Home Points': actual_home_points,
                'Away Points': actual_away_points,
                'Result with Spread': actual_difference,
                'Actual Covered': actual_covered,
                'Bet Outcome': bet_outcome,
                'Actual Value ($)': actual_value
            }

        logging.warning(f"Row {idx} does not match any expected scenarios.")
        return {}

    def determine_recommended_bet(self, recommendation_calc):
        if recommendation_calc < 0:
            return "away"
        elif recommendation_calc > 0:
            return "home"
        elif recommendation_calc == 0:
            return "push"
        return None

    def determine_bet_outcome(self, actual_difference, recommended_bet):
        if actual_difference is not None:
            actual_covered = "away" if actual_difference < 0 else "home" if actual_difference > 0 else "push"
            if recommended_bet == actual_covered:
                return actual_covered, "win", 100
            elif recommended_bet == "push" or actual_covered == "push":
                return actual_covered, "push", 0
            else:
                return actual_covered, "loss", -110
        return None, None, None

    def append_results(self, results_df, processed_data):
        """
        Append processed game data to the results DataFrame.

        Args:
            results_df (pd.DataFrame): DataFrame to which data will be appended.
            processed_data (dict): Processed game data.

        Returns:
            pd.DataFrame: Updated results DataFrame.
        """
        # Convert processed_data to a DataFrame row
        new_row = pd.DataFrame([processed_data])

        # Append the new row to the results DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        return results_df

    def finalize_results(self, results_df, get_current):
        """
        Finalize the results DataFrame, create a Plotly table for visualization, and save it as an HTML file.

        Args:
            results_df (pd.DataFrame): DataFrame containing processed game data.
            get_current (bool): Whether to finalize current or historical results.

        Returns:
            pd.DataFrame: The finalized results DataFrame.
        """
        # Round all values in the DataFrame to 2 decimals
        results_df = results_df.round(2)

        # Create a Plotly table for visualization
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(results_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[results_df[col].tolist() for col in results_df.columns],
                       fill_color='lavender',
                       align='left'))
        ])

        # Update layout for a better visual appearance
        fig.update_layout(
            title='Betting Results',
            title_x=0.5,
            margin=dict(l=10, r=10, t=30, b=10)
        )

        # Convert the Plotly figure to an HTML string
        betting_results_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        if get_current is True:
            # Define the path to save the results
            betting_results_path = os.path.join(self.template_dir, 'future_betting_recommendations.html')
        else:
            betting_results_path = os.path.join(self.template_dir, 'historical_results_backtesting.html')

        # Save the combined HTML
        try:
            with open(betting_results_path, "w") as f:
                f.write(betting_results_html)
        except IOError as e:
            logging.info(f"Error writing to file: {e}")

        return results_df

    def calculate_summary_statistics(self, results_df, total_bets, correct_recommendations, get_current):
        """
        Calculate summary statistics including recommendation accuracy and total actual value.

        Args:
            results_df (pd.DataFrame): DataFrame containing processed game data.
            total_bets (int): Total number of bets placed.
            correct_recommendations (int): Total number of correct recommendations.
            get_current (bool): Whether to calculate for current or historical results.

        Returns:
            tuple: A tuple containing recommendation accuracy (float) and total actual value (float).
        """
        recommendation_accuracy = 0
        total_actual_value = 0

        if total_bets > 0:
            # Calculate the total actual value from all bets
            total_actual_value = results_df['Actual Value ($)'].sum()

            # Calculate the recommendation accuracy
            recommendation_accuracy = (correct_recommendations / total_bets) * 100

        return recommendation_accuracy, total_actual_value

    def create_summary_dashboard(self, results_df, total_bets, recommendation_accuracy, total_actual_value):
        """
        Create a summary dashboard in HTML format with key metrics.

        Args:
            results_df (pd.DataFrame): DataFrame containing processed game data.
            total_bets (int): Total number of bets placed.
            recommendation_accuracy (float): Recommendation accuracy in percentage.
            total_actual_value (float): Total actual value from all bets.

        Returns:
            None
        """
        # Calculate additional metrics
        overall_record = results_df['Bet Outcome'].value_counts()

        # Format records and calculate win percentage
        overall_record_df = self.format_records(overall_record)

        # Convert to HTML without the index
        overall_record_html = overall_record_df.to_html(index=False, classes='table')

        # Create HTML content
        html_content = f"""
            <div class="container">
                <div class="stats-card">
                    <h2>Key Metrics</h2>
                    <p>Total Bets: <strong>{total_bets}</strong></p>
                    <p>Overall Record: <strong>{overall_record_html}</strong></p>
                    <p>Overall Accuracy: <strong>{recommendation_accuracy:.2f}%</strong></p>
                    <p>Total Actual Value: <strong>${total_actual_value:.2f}</strong></p>
                </div>
            </div>
        """

        summary_dash_path = os.path.join(self.template_dir, 'summary_dash.html')

        # Write the HTML content to a file
        with open(summary_dash_path, 'w') as f:
            f.write(html_content)

    def save_results_as_csv(self, results_df):
        """
        Save the results DataFrame as a CSV file.

        Args:
            results_df (pd.DataFrame): DataFrame containing processed game data.

        Returns:
            None
        """
        # Construct the file path for the CSV file
        csv_results_path = os.path.join(self.static_dir, 'historical.csv')

        # Save the DataFrame as a CSV file
        try:
            results_df.to_csv(csv_results_path, index=False)
            logging.info(f"Results saved successfully to {csv_results_path}")
        except Exception as e:
            logging.info(f"Error saving results to CSV: {e}")

    def calculate_weekly_profit(self, df):
        """
        Calculate weekly profit from the historical data, considering weeks from Tuesday to Tuesday
        and excluding weeks from March through August.

        Args:
            df (pd.DataFrame): DataFrame containing historical game data.

        Returns:
            pd.Series: Weekly profit values.
        """
        # Convert 'Date' to datetime and exclude games from March through August
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[~df['Date'].dt.month.isin([3, 4, 5, 6, 7, 8])]

        # Adjust the date to align weeks from Tuesday to Tuesday
        df['Adjusted Week'] = df['Date'].apply(lambda x: x - timedelta(days=x.weekday()) + timedelta(days=1))

        # Group by week and calculate cumulative sum of 'Actual Value ($)'
        weekly_profit = df.groupby('Adjusted Week')['Actual Value ($)'].sum().cumsum()

        return weekly_profit

    def plot_weekly_profit_graph(self, weekly_profit):
        """
        Plot the graph for weekly profit, excluding March through August, and return as HTML string.
        """
        # Exclude months March through August
        weekly_profit_filtered = weekly_profit[~weekly_profit.index.month.isin([3, 4, 5, 6, 7, 8])]

        fig, ax = plt.subplots(figsize=(10, 6))
        weekly_profit_filtered.plot(kind='line', ax=ax)
        ax.set_title("Weekly Profit Over Time (Excluding Mar-Aug)")
        ax.set_xlabel("Week Starting Date")
        ax.set_ylabel("Cumulative Profit")
        ax.grid(True)
        return mpld3.fig_to_html(fig)

    def plot_profitable_weeks_chart(self, weekly_profit):
        """
        Create a chart for the top 5 most and least profitable weeks and return as HTML string.
        """
        # Convert index to string if it's in datetime format
        weekly_profit.index = weekly_profit.index.strftime('%Y-%m-%d')

        # Ensure the data is numeric
        weekly_profit = weekly_profit.astype(float)

        top_5_profitable = weekly_profit.nlargest(5)
        least_5_profitable = weekly_profit.nsmallest(5)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

        top_5_profitable.plot(kind='bar', ax=axes[0], color='green')
        axes[0].set_title("Top 5 Most Profitable Weeks")
        axes[0].set_xlabel("Week")
        axes[0].set_ylabel("Profit")

        least_5_profitable.plot(kind='bar', ax=axes[1], color='red')
        axes[1].set_title("Top 5 Least Profitable Weeks")
        axes[1].set_xlabel("Week")
        axes[1].set_ylabel("Profit")

        plt.tight_layout()
        return mpld3.fig_to_html(fig)

    def calculate_accuracy_metrics(self, df):
        logging.info("Starting calculation of accuracy metrics.")

        # Convert 'Date' to datetime and exclude games from March through August
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[~df['Date'].dt.month.isin([3, 4, 5, 6, 7, 8])]
        logging.info(f"DataFrame after excluding March-August: {df.head()}")

        # Adjust the date to align weeks from Tuesday to Tuesday
        df['Adjusted Week'] = df['Date'].apply(lambda x: x - timedelta(days=x.weekday()) + timedelta(days=1))
        logging.info(f"DataFrame after adjusting weeks: {df.head()}")

        # Calculate differences
        df['Diff'] = df['Vegas Odds'] - df['Modeling Odds']
        logging.info(f"DataFrame with 'Diff' column: {df.head()}")

        # Aggregate by 'Adjusted Week'
        aggregated = df.groupby('Adjusted Week').agg(
            Mean=('Diff', 'mean'),
            SquaredDiff=('Diff', lambda x: np.mean(x**2)),  # Mean of squared differences
            AbsDiff=('Diff', lambda x: np.mean(np.abs(x))),  # Mean of absolute differences
            Std=('Diff', 'std'),
            Count=('Diff', 'count')
        )
        logging.info(f"Aggregated stats: {aggregated.head()}")

        # Calculate cumulative MSE and MAE as averages
        aggregated['Cumulative MSE'] = (aggregated['SquaredDiff'] * aggregated['Count']).cumsum() / aggregated['Count'].cumsum()
        aggregated['Cumulative MAE'] = (aggregated['AbsDiff'] * aggregated['Count']).cumsum() / aggregated['Count'].cumsum()

        # Standard deviation calculation remains the same
        aggregated['Sum of Squares'] = (aggregated['Std'] ** 2) * (aggregated['Count'] - 1)
        cumulative_sum_of_squares = aggregated['Sum of Squares'].cumsum()
        cumulative_count = aggregated['Count'].cumsum()
        aggregated['Cumulative Std Dev'] = np.sqrt(cumulative_sum_of_squares / (cumulative_count - 1))

        logging.info(f"Cumulative statistics: {aggregated[['Cumulative MSE', 'Cumulative MAE', 'Cumulative Std Dev']].head()}")

        return aggregated[['Cumulative MSE', 'Cumulative MAE', 'Cumulative Std Dev']].reset_index()

    def plot_accuracy_trends(self, accuracy_df):
        logging.info("Starting plotting of accuracy trends.")

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15), sharex=True)

        # Plotting Cumulative MSE
        axes[0].plot(accuracy_df['Adjusted Week'], accuracy_df['Cumulative MSE'], color='blue')
        axes[0].set_title("Cumulative MSE Trends Over Time")
        axes[0].set_ylabel("Cumulative MSE")

        # Plotting Cumulative MAE
        axes[1].plot(accuracy_df['Adjusted Week'], accuracy_df['Cumulative MAE'], color='green')
        axes[1].set_title("Cumulative MAE Trends Over Time")
        axes[1].set_ylabel("Cumulative MAE")

        # Plotting Cumulative Standard Deviation
        axes[2].plot(accuracy_df['Adjusted Week'], accuracy_df['Cumulative Std Dev'], color='red')
        axes[2].set_title("Cumulative Standard Deviation Trends Over Time")
        axes[2].set_ylabel("Cumulative Standard Deviation")
        axes[2].set_xlabel("Adjusted Week")  # Only set x-label for the bottommost subplot

        for ax in axes:
            ax.grid(True)
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

        # Adjust subplot spacing and apply tight layout with padding
        fig.subplots_adjust(hspace=0.5)
        plt.tight_layout(pad=3.0)

        logging.info("Plotting completed.")

        return mpld3.fig_to_html(fig)

    def create_trending_dashboard(self, results_df):
        """
        Create a separate dashboard for trending data including weekly profit tracking,
        top profitable weeks, and model accuracy metrics.
        """
        # Calculate and plot weekly profit
        weekly_profit = self.calculate_weekly_profit(results_df)
        weekly_profit_graph = self.plot_weekly_profit_graph(weekly_profit)

        # Calculate and plot accuracy metrics
        accuracy_metrics_df = self.calculate_accuracy_metrics(results_df)
        accuracy_metrics_graph = self.plot_accuracy_trends(accuracy_metrics_df)

        # Combine all HTML content for the dashboard
        html_content = f"""
            <div class="container">
                <div class="chart-card">
                    <h2>Weekly Profit Tracking</h2>
                    {weekly_profit_graph}
                </div>
                <div class="chart-card">
                    <h2>Model Accuracy Over Time</h2>
                    {accuracy_metrics_graph}
                </div>
            </div>
        """

        trending_dash_path = os.path.join(self.template_dir, 'trending_dash.html')

        # Write the HTML content to a file
        with open(trending_dash_path, 'w') as f:
            f.write(html_content)

    def evaluate_and_recommend(self, simulation_results, historical_df, get_current):
        """
        Evaluate betting recommendations and create a summary dashboard.

        Args:
            simulation_results (np.ndarray): Array of simulated betting results.
            historical_df (pd.DataFrame): DataFrame containing historical game data.
            get_current (bool): Whether to evaluate current or historical results.

        Returns:
            pd.DataFrame: The finalized results DataFrame.
            float: Recommendation accuracy in percentage.
            float: Total actual value from all bets.
        """
        logging.info("Starting evaluation and recommendation process.")

        # Prepare initial data
        historical_df, results_df = self.prepare_data(historical_df, get_current)
        logging.info("Data preparation complete.")

        correct_recommendations, total_bets = 0, 0

        # Process each row in the historical data
        for idx, row in historical_df.iterrows():
            processed_data = self.process_row(row, simulation_results, idx, get_current)
            logging.info(f"Processed data for row {idx}: {processed_data}")

            results_df = self.append_results(results_df, processed_data)
            logging.info(f"Results appended for row {idx}.")

            if get_current is False:
                # Update the count of correct recommendations and total bets
                if processed_data['Bet Outcome'] == 'win':
                    correct_recommendations += 1
                if processed_data['Bet Outcome'] in ['win', 'loss']:
                    total_bets += 1

            logging.info(f"Correct recommendations: {correct_recommendations}, Total bets: {total_bets}")

        # Finalize results and calculate summary statistics
        final_df = self.finalize_results(results_df, get_current)
        recommendation_accuracy, total_actual_value = self.calculate_summary_statistics(final_df, total_bets, correct_recommendations, get_current)
        logging.info("Finalized results and calculated summary statistics.")

        # Create a summary dashboard (if applicable)
        if get_current is False:
            self.save_results_as_csv(results_df)
            self.create_summary_dashboard(final_df, total_bets, recommendation_accuracy, total_actual_value)
            logging.info("Summary dashboard created and saved.")

            # Create and save the trending dashboard
            self.create_trending_dashboard(results_df)
            logging.info("Trending dashboard created and saved.")

        logging.info("Evaluation and recommendation process completed.")
        return final_df, recommendation_accuracy, total_actual_value
