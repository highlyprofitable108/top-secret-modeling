import os
import scipy.stats
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Visualization:
    def __init__(self, template_dir, target_variable):
        self.template_dir = template_dir
        self.TARGET_VARIABLE = "summary." + target_variable

    def visualize_simulation_results(self, simulation_results, most_likely_outcome, output, bins=50):
        """
        Visualizes the simulation results using a histogram and calculates 
        the rounded most likely outcome.
        """

        # Plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=simulation_results, name='Simulation Results', nbinsx=bins))
        fig.add_shape(type="line", x0=most_likely_outcome, x1=most_likely_outcome, y0=0, y1=1, yref='paper', line=dict(color="Red"))
        fig.update_layout(title="Monte Carlo Simulation Results", xaxis_title="Target Value", yaxis_title="Density")

        # Calculate the median prediction
        median_margin = np.median(simulation_results)
        # Calculate the expected value
        expected_value = np.mean(simulation_results)
        # Calculate the 95% confidence interval
        confidence_level = 0.95
        sigma = np.std(simulation_results)
        confidence_interval = scipy.stats.norm.interval(confidence_level, loc=expected_value, scale=sigma/np.sqrt(len(simulation_results)))

        # Add lines for median, expected value, and confidence interval
        fig.add_shape(type="line", x0=median_margin, x1=median_margin, y0=0, y1=1, yref='paper', line=dict(color="Blue", dash="dash"), name="Median")
        fig.add_shape(type="line", x0=expected_value, x1=expected_value, y0=0, y1=1, yref='paper', line=dict(color="Purple", dash="dot"), name="Expected Value")
        fig.add_shape(type="line", x0=confidence_interval[0], x1=confidence_interval[0], y0=0, y1=1, yref='paper', line=dict(color="Green", dash="longdash"), name="Lower CI Bound")
        fig.add_shape(type="line", x0=confidence_interval[1], x1=confidence_interval[1], y0=0, y1=1, yref='paper', line=dict(color="Red", dash="longdash"), name="Upper CI Bound")

        # Convert the Plotly figure to HTML
        plotly_html_string = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Format the output for better readability
        formatted_output = f"""
        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h2>Simulation Summary:</h2>
            <pre style="font-size: 16px; color: #333;">{output}</pre>
        </div>
        """

        # Combine the formatted output and the Plotly HTML
        full_html = plotly_html_string + formatted_output

        # Path to save the visualization as an HTML file
        simulation_path = os.path.join(self.template_dir, 'simulation_distribution.html')

        # Save the combined HTML
        try:
            with open(simulation_path, "w") as f:
                f.write(full_html)
        except IOError as e:
            # Handle potential file write issues
            print(f"Error writing to file: {e}")

        # Calculate the rounded most likely outcome
        rounded_most_likely_outcome = (round(most_likely_outcome * 2) / 2)

        # Format the rounded most likely outcome with a leading + sign for positive values
        if rounded_most_likely_outcome > 0:
            formatted_most_likely_outcome = f"+{rounded_most_likely_outcome:.2f}"
        else:
            formatted_most_likely_outcome = f"{rounded_most_likely_outcome:.2f}"

        return formatted_most_likely_outcome

    def compare_simulated_to_actual(self, simulation_results, actual_results):
        model_covered = 0

        for idx in range(len(simulation_results)):
            actual_difference = actual_results[idx]
            predicted_difference = simulation_results[idx]

            if actual_difference > np.mean(predicted_difference):
                model_covered += 1

        model_cover_rate = model_covered / len(simulation_results) * 100

        return model_cover_rate

    def evaluate_and_recommend(self, simulation_results, historical_df):
        historical_df = historical_df.reset_index(drop=True)
        correct_recommendations = 0
        total_bets = 0
        total_ev = 0

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(columns=['Date', 'Home Team', 'Vegas Odds', 'Modeling Odds', 'Away Team', 'Recommended Bet', 'Expected Value (%)', 'Home Points', 'Away Points', 'Result with Spread', 'Actual Covered', 'Bet Outcome', 'Actual Value ($)'])

        for idx, row in historical_df.iterrows():
            actual_home_points = row['summary.home.points']
            actual_away_points = row['summary.away.points']
            home_team = row['summary.home.name']
            away_team = row['summary.away.name']
            spread_odds = row[self.TARGET_VARIABLE]
            date = row['scheduled']

            predicted_difference = simulation_results[idx]

            # Check if the game has actual results or if it's a future game
            if pd.isna(actual_home_points) or pd.isna(actual_away_points):
                # Future game
                actual_covered = None
                bet_outcome = None
                actual_value = None
                actual_difference = None
                recommended_bet = None
                ev_percentage = None
            else:
                # Past game with actual results
                actual_difference = (actual_home_points + spread_odds) - actual_away_points

                # Determine who actually covered based on spread odds
                if actual_difference < 0:
                    actual_covered = "away"
                elif actual_difference > 0:
                    actual_covered = "home"
                elif actual_difference == 0:
                    recommended_bet = "push"

                # Recommendation based on model
                reccommendation_calc = spread_odds - np.mean(predicted_difference)
                if reccommendation_calc < 0:
                    recommended_bet = "away"
                elif reccommendation_calc > 0:
                    recommended_bet = "home"
                elif reccommendation_calc == 0:
                    recommended_bet = "push"

                # Check historical performance
                if recommended_bet == actual_covered and recommended_bet is not None:
                    correct_recommendations += 1
                    total_bets += 1
                    bet_outcome = "Win"
                    actual_value = 100  # Profit from a winning bet
                elif recommended_bet == "push" or actual_covered == "push":
                    bet_outcome = "Push"
                    actual_value = 0  # No profit, no loss from a push
                elif recommended_bet is not None:
                    total_bets += 1
                    bet_outcome = "Loss"
                    actual_value = -110  # Loss from a losing bet

                # Calculate the model's implied probability
                if recommended_bet == "home":
                    model_probability = 1 / (1 + 10 ** (np.mean(predicted_difference) / 10))
                elif recommended_bet == "away":
                    model_probability = 1 / (1 + 10 ** (-np.mean(predicted_difference) / 10))
                else:
                    model_probability = None

                # Calculate the Vegas implied probability
                if recommended_bet == "home":
                    vegas_probability = 1 / (1 + 10 ** (spread_odds / 10))
                elif recommended_bet == "away":
                    vegas_probability = 1 / (1 + 10 ** (-spread_odds / 10))
                else:
                    vegas_probability = None

                # Calculate expected value (adjusted for model vs. Vegas odds)
                if vegas_probability is not None and model_probability is not None:
                    potential_profit = 100  # Assuming a winning bet returns $100
                    amount_bet = 100  # Assuming a bet amount of $100
                    ev = (potential_profit * model_probability) - (amount_bet * vegas_probability)
                    ev_percentage = ev  # Express EV as a percentage of the bet
                    total_ev += ev  # Accumulate the total expected value

            # Append to results DataFrame
            new_row = pd.Series({
                'Date': date,
                'Home Team': home_team,
                'Vegas Odds': spread_odds,
                'Modeling Odds': np.mean(predicted_difference),
                'Away Team': away_team,
                'Recommended Bet': recommended_bet,
                'Expected Value (%)': ev_percentage,
                'Home Points': actual_home_points,
                'Away Points': actual_away_points,
                'Result with Spread': actual_difference,
                'Actual Covered': actual_covered,
                'Bet Outcome': bet_outcome,
                'Actual Value ($)': actual_value
            })

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # Round all values in the DataFrame to 2 decimals
        results_df = results_df.round(2)

        # Define paths to save the results
        # csv_path = os.path.join(self.template_dir, 'betting_recommendation_results.csv')
        html_path = os.path.join(self.template_dir, 'betting_recommendation_results.html')

        # Save results to CSV
        # try:
        #     results_df.to_csv(csv_path, index=False)
        # except IOError as e:
        #     print(f"Error writing to CSV file: {e}")

        # Save results to HTML
        try:
            results_df.to_html(html_path, index=False, classes='table table-striped')
        except IOError as e:
            print(f"Error writing to HTML file: {e}")

        recommendation_accuracy = 0
        average_ev_percent = 0
        total_actual_value = 0

        if total_bets:
            # Calculate the total actual value from all bets
            total_actual_value = results_df['Actual Value ($)'].sum()

            recommendation_accuracy = correct_recommendations / total_bets * 100
            average_ev = total_ev / total_bets
            average_ev_percent = (average_ev / 110) * 100  # Adjust 110 if your standard bet amount is different

        return recommendation_accuracy, average_ev_percent, total_actual_value

    def visualize_value_opportunity(self, simulation_results, perceived_value):
        # Calculate the expected value
        expected_value = np.mean(simulation_results)
        value_opportunity = expected_value - perceived_value

        # Plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=simulation_results, name='Simulation Results'))
        fig.add_shape(type="line", x0=perceived_value, x1=perceived_value, y0=0, y1=1, yref='paper', line=dict(color="Yellow"), name="Perceived Value")
        fig.add_shape(type="line", x0=expected_value, x1=expected_value, y0=0, y1=1, yref='paper', line=dict(color="Purple", dash="dot"), name="Expected Value")
        fig.update_layout(title="Model vs Perceived Value", xaxis_title="Value", yaxis_title="Density")

        # Convert the Plotly figure to HTML
        plotly_html_string = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Path to save the visualization as an HTML file
        value_opportunity_path = os.path.join(self.template_dir, 'value_opportunity_distribution.html')

        # Save the combined HTML
        try:
            with open(value_opportunity_path, "w") as f:
                f.write(plotly_html_string)
        except IOError as e:
            # Handle potential file write issues
            print(f"Error writing to file: {e}")

        return value_opportunity
