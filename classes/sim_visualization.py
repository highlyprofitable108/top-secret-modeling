import os
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, date as date_type


class SimVisualization:
    def __init__(self, template_dir=None, target_variable="odds.spread_close", static_dir="./flask_app/static"):
        self.template_dir = template_dir
        self.static_dir = static_dir
        self.TARGET_VARIABLE = target_variable

    def compute_confidence_interval(self, data, confidence=0.95):
        """Compute the confidence interval for a given dataset."""
        print("Starting confidence interval computation.")

        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), np.std(a) / np.sqrt(n)

        h = se * norm.ppf((1 + confidence) / 2.)

        confidence_interval = (m - h, m + h)
        print(f"Computed confidence interval: {confidence_interval}")

        return confidence_interval

    def filter_simulation_results(self, simulation_results):
        print("Starting filtering of simulation results.")
        print(simulation_results)

        # Check if the simulation results are None or contain only a single value
        if simulation_results is None or (isinstance(simulation_results, np.ndarray) and simulation_results.size <= 1):
            print(f"Simulation results are empty or contain only a single value: {simulation_results}")
            return []

        # Handle NaN values by replacing them with zero
        simulation_results = np.nan_to_num(simulation_results)

        # Check if the simulation results array is empty
        if simulation_results.size == 0:
            print("Simulation results are empty.")
            return []

        print(f"Simulation results after handling NaNs: {simulation_results}")

        # Constants for bounds
        LOWER_PERCENTILE = 0.0
        UPPER_PERCENTILE = 1

        # Calculate bounds
        lower_bound = np.percentile(simulation_results, LOWER_PERCENTILE * 100)
        upper_bound = np.percentile(simulation_results, UPPER_PERCENTILE * 100)

        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

        # Filter
        filtered_results = [result for result in simulation_results if lower_bound <= result <= upper_bound]
        print(f"Filtered results: {filtered_results}")

        return filtered_results

    def analyze_simulation_results(self, simulation_results):
        print("Starting analysis of simulation results.")

        filtered_results = self.filter_simulation_results(simulation_results)
        if not filtered_results:
            print("No data available after filtering simulation results.")
            return None, None, None

        range_of_outcomes = (min(filtered_results), max(filtered_results))
        std_deviation = np.std(filtered_results)
        confidence_interval = self.compute_confidence_interval(filtered_results)

        print(f"Range of outcomes: {range_of_outcomes}")
        print(f"Standard deviation: {std_deviation}")
        print(f"Confidence interval: {confidence_interval}")

        return range_of_outcomes, std_deviation, confidence_interval

    def format_records(self, records):
        records = records.reindex(['win', 'loss', 'push'], fill_value=0)
        win_pct = records['win'] / (records['win'] + records['loss']) if (records['win'] + records['loss']) > 0 else 0
        win_pct = "{:.2f}%".format(win_pct * 100)  # Format as percentage with two decimals
        records_df = records.to_frame().reset_index()
        records_df.columns = ['Bet Outcome', 'Count']  # Rename columns for clarity
        win_pct_row = pd.DataFrame([['win pct', win_pct]], columns=['Bet Outcome', 'Count'])
        return pd.concat([records_df, win_pct_row], ignore_index=True)

    def prepare_data(self, historical_df, get_current):
        historical_df = historical_df.reset_index(drop=True)

        if get_current is True:
            results_df = pd.DataFrame(columns=['Date', 'Home Team', 'Vegas Odds', 'Modeling Odds',
                                               'Away Team', 'Recommended Bet', 'Expected Value (%)'])
        else:
            results_df = pd.DataFrame(columns=['Date', 'Home Team', 'Vegas Odds', 'Modeling Odds',
                                               'Away Team', 'Recommended Bet', 'Home Points', 'Away Points',
                                               'Result with Spread', 'Actual Covered', 'Bet Outcome', 'Actual Value ($)'])
        return historical_df, results_df

    def round_to_nearest_half(self, value):
        rounded = round(value * 2) / 2
        return rounded if value % 0.5 > 0.25 else round(value)

    def create_custom_percentiles(self):
        # Non-linear distribution of percentiles
        # More percentiles around the center, fewer towards the extremes
        close_to_center = np.linspace(30, 70, num=10)  # Adjust these numbers based on your data distribution
        extreme_lower = np.linspace(0, 30, num=5)
        extreme_upper = np.linspace(70, 100, num=5)

        # Combine and sort the arrays
        all_percentiles = np.sort(np.concatenate([extreme_lower, close_to_center, extreme_upper]))

        return all_percentiles

    def estimate_win_probability_based_on_spread_diff(self, predicted_spread, vegas_line, df):
        # Calculate and add the 'Spread Difference Rounded' column to the DataFrame
        df['Spread Difference'] = df['Modeling Odds'] - df['Vegas Odds']
        df['Spread Difference Rounded'] = df['Spread Difference'].apply(self.round_to_nearest_half)

        # Round the current spreads
        predicted_spread_rounded = self.round_to_nearest_half(predicted_spread)
        vegas_line_rounded = self.round_to_nearest_half(vegas_line)

        # Calculate the current spread difference
        spread_diff = predicted_spread_rounded - vegas_line_rounded

        # Generate custom percentiles
        custom_percentiles = self.create_custom_percentiles()

        # Calculate the actual percentiles based on the data
        quantiles = np.percentile(df['Spread Difference Rounded'], custom_percentiles)

        # Remove duplicate values from quantiles and ensure unique bin edges
        unique_quantiles = np.unique(quantiles)

        # Create bins with unique edges
        extended_bins = np.concatenate(([-np.inf], unique_quantiles, [np.inf]))
        labels = [f"{extended_bins[i]} to {extended_bins[i+1]}" for i in range(len(extended_bins) - 1)]

        # Assign each spread difference to a bin
        df['Spread Diff Bin'] = pd.cut(df['Spread Difference Rounded'], bins=extended_bins, labels=labels)

        # Calculate win rates for each bin
        win_rates = df.groupby('Spread Diff Bin')['Bet Outcome'].apply(lambda x: (x == 'win').sum() / len(x))

        # Log the size of each bin
        bin_sizes = df['Spread Diff Bin'].value_counts()
        logging.info("Bin Sizes:\n%s", bin_sizes)

        # Estimate probability for current spread difference
        current_bin_label = labels[np.digitize([spread_diff], extended_bins)[0] - 1]
        P_win = win_rates.get(current_bin_label, 0)

        return P_win

    def calculate_ev(self, predicted_spread, vegas_line):
        # Load historical data
        historical_df = pd.read_csv(os.path.join(self.static_dir, 'historical.csv'))

        # Estimate the probability of winning based on spread difference
        P_win = self.estimate_win_probability_based_on_spread_diff(predicted_spread, vegas_line, historical_df)

        # The loss probability is complementary to the win probability
        P_loss = 1 - P_win

        # Calculate EV considering all bets are at -110 odds
        expected_value = (P_win * 100) - (P_loss * 110)

        return expected_value

    def process_row(self, row, simulation_results, idx, get_current):
        actual_home_points = row['summary.home.points']
        actual_away_points = row['summary.away.points']
        home_team = row['summary.home.name']
        away_team = row['summary.away.name']
        vegas_line = row["summary." + self.TARGET_VARIABLE]
        date = row['scheduled']
        predicted_difference = simulation_results[idx]

        # Convert 'date' to a datetime object if it's not already
        if isinstance(date, date_type) and not isinstance(date, datetime):
            date = datetime.combine(date, datetime.min.time())

        # Initialize the variables
        recommended_bet = None
        actual_covered = None
        bet_outcome = None
        actual_value = None
        actual_difference = None

        # Process for games with or without odds
        if pd.isna(vegas_line):
            recommended_bet = None
        else:
            actual_difference = (actual_home_points + vegas_line) - actual_away_points
            recommendation_calc = vegas_line - np.mean(predicted_difference)
            if recommendation_calc < 0:
                recommended_bet = "away"
            elif recommendation_calc > 0:
                recommended_bet = "home"
            elif recommendation_calc == 0:
                recommended_bet = "push"
            else:
                recommended_bet = None

            # Determine actual results
            if actual_difference is not None:
                if actual_difference < 0:
                    actual_covered = "away"
                elif actual_difference > 0:
                    actual_covered = "home"
                else:
                    actual_covered = "push"

                # Determine bet outcome
                if recommended_bet == actual_covered:
                    bet_outcome = "win"
                    actual_value = 100
                elif recommended_bet == "push" or actual_covered == "push":
                    bet_outcome = "push"
                    actual_value = 0
                else:
                    bet_outcome = "loss"
                    actual_value = -110

        if get_current is True:
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
        else:
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

    def append_results(self, results_df, processed_data):
        # Convert processed_data to a DataFrame row
        new_row = pd.DataFrame([processed_data])

        # Append the new row to the results DataFrame
        results_df = pd.concat([results_df, new_row], ignore_index=True)

        return results_df

    def finalize_results(self, results_df, get_current):
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
            print(f"Error writing to file: {e}")

        return results_df

    def calculate_summary_statistics(self, results_df, total_bets, correct_recommendations, get_current):
        recommendation_accuracy = 0
        total_actual_value = 0

        if total_bets > 0:
            # Calculate the total actual value from all bets
            total_actual_value = results_df['Actual Value ($)'].sum()

            # Calculate the recommendation accuracy
            recommendation_accuracy = (correct_recommendations / total_bets) * 100

        return recommendation_accuracy, total_actual_value

    def create_summary_dashboard(self, results_df, total_bets, recommendation_accuracy, total_actual_value):
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
        # Construct the file path for the CSV file
        csv_results_path = os.path.join(self.static_dir, 'historical.csv')

        # Save the DataFrame as a CSV file
        try:
            results_df.to_csv(csv_results_path, index=False)
            print(f"Results saved successfully to {csv_results_path}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")

    def evaluate_and_recommend(self, simulation_results, historical_df, get_current):
        # Prepare initial data
        historical_df, results_df = self.prepare_data(historical_df, get_current)

        correct_recommendations, total_bets = 0, 0
        # Process each row in the historical data
        for idx, row in historical_df.iterrows():
            processed_data = self.process_row(row, simulation_results, idx, get_current)

            results_df = self.append_results(results_df, processed_data)
            if get_current is False:
                # Update the count of correct recommendations and total bets
                if processed_data['Bet Outcome'] == 'win':
                    correct_recommendations += 1
                if processed_data['Bet Outcome'] in ['win', 'loss']:
                    total_bets += 1

        # Finalize results and calculate summary statistics
        final_df = self.finalize_results(results_df, get_current)
        recommendation_accuracy, total_actual_value = self.calculate_summary_statistics(final_df, total_bets, correct_recommendations, get_current)

        # Create a summary dashboard (if applicable)
        if get_current is False:
            self.save_results_as_csv(results_df)
            self.create_summary_dashboard(final_df, total_bets, recommendation_accuracy, total_actual_value)

        return final_df, recommendation_accuracy, total_actual_value
