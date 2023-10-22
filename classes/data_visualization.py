import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Visualization:
    def __init__(self, template_dir):
        self.template_dir = template_dir

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
        feature_importance_path = os.path.join(self.template_dir, 'simulation_distribution.html')

        # Save the combined HTML
        try:
            with open(feature_importance_path, "w") as f:
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

            if actual_difference > predicted_difference:
                model_covered += 1

        model_cover_rate = model_covered / len(simulation_results) * 100

        return model_cover_rate

    def evaluate_and_recommend(self, simulation_results, historical_df):
        historical_df = historical_df.reset_index(drop=True)
        correct_recommendations = 0
        total_bets = 0
        total_ev = 0

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(columns=['Date', 'Home Team', 'Home Points', 'Vegas Odds', 'Modeling Odds', 'Away Team', 'Away Points', 'Recommended Bet', 'Result with Spread', 'Actual Covered', 'Bet Outcome', 'Expected Value (%)', 'Actual Value ($)'])

        for idx, row in historical_df.iterrows():
            actual_home_points = row['summary.home.points']
            actual_away_points = row['summary.away.points']
            home_team = row['summary.home.name']
            away_team = row['summary.away.name']
            spread_odds = row['summary.odds.spread']
            date = row['scheduled']

            predicted_difference = simulation_results[idx]

            # Check if the game has actual results or if it's a future game
            if pd.isna(actual_home_points) or pd.isna(actual_away_points):
                # Future game
                actual_covered = ""
                bet_outcome = ""
                actual_value = np.nan
                actual_difference = np.nan
            else:
                # Past game with actual results
                actual_difference = (actual_home_points + spread_odds) - actual_away_points

                # Determine who actually covered based on spread odds
                if actual_difference < 0:
                    actual_covered = "away"
                elif actual_difference > 0:
                    actual_covered = "home"
                else:
                    actual_covered = "push"

                # Check historical performance
                if recommended_bet == actual_covered:
                    correct_recommendations += 1
                    total_bets += 1
                    bet_outcome = "Win"
                    actual_value = 100  # Profit from a winning bet
                elif recommended_bet == "push" or actual_covered == "push":
                    bet_outcome = "Push"
                    actual_value = 0  # No profit, no loss from a push
                else:
                    total_bets += 1
                    bet_outcome = "Loss"
                    actual_value = -110  # Loss from a losing bet

            # Recommendation based on model
            reccommendation_calc = spread_odds - predicted_difference
            if reccommendation_calc < 0:
                recommended_bet = "away"
            elif reccommendation_calc > 0:
                recommended_bet = "home"
            else:
                recommended_bet = "push"  # or any default recommendation

            # Calculate the model's implied probability
            if recommended_bet == "home":
                model_probability = 1 / (1 + 10 ** (predicted_difference / 10))
            else:
                model_probability = 1 / (1 + 10 ** (-predicted_difference / 10))

            # Calculate the Vegas implied probability
            if recommended_bet == "home":
                vegas_probability = 1 / (1 + 10 ** (spread_odds / 10))
            else:
                vegas_probability = 1 / (1 + 10 ** (-spread_odds / 10))

            # Calculate expected value (adjusted for model vs. Vegas odds)
            ev_fraction = model_probability - vegas_probability  # This is the EV as a fraction of the bet

            # Express EV as a percentage of the bet
            ev_percentage = ev_fraction * 100

            total_ev += ev_percentage

            # Check historical performance
            if recommended_bet == actual_covered:
                correct_recommendations += 1
                total_bets += 1
                bet_outcome = "Win"
                actual_value = 100  # Profit from a winning bet
            elif recommended_bet == "push" or actual_covered == "push":
                bet_outcome = "Push"
                actual_value = 0  # No profit, no loss from a push
            else:
                total_bets += 1
                bet_outcome = "Loss"
                actual_value = -110  # Loss from a losing bet

            # Append to results DataFrame
            new_row = pd.Series({
                'Date': date,
                'Home Team': home_team,
                'Home Points': actual_home_points,
                'Vegas Odds': spread_odds,
                'Modeling Odds': predicted_difference,
                'Away Team': away_team,
                'Away Points': actual_away_points,
                'Recommended Bet': recommended_bet,
                'Result with Spread': actual_difference,
                'Actual Covered': actual_covered,
                'Bet Outcome': bet_outcome,
                'Expected Value (%)': ev_percentage,
                'Actual Value ($)': actual_value
            })

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # Round all values in the DataFrame to 2 decimals
        results_df = results_df.round(2)

        # Save results to CSV
        results_df.to_csv('betting_recommendation_results.csv', index=False)

        # Calculate the total actual value from all bets
        total_actual_value = results_df['Actual Value ($)'].sum()

        recommendation_accuracy = correct_recommendations / total_bets * 100
        average_ev = total_ev / total_bets
        average_ev_percent = (average_ev / 110) * 100

        return recommendation_accuracy, average_ev_percent, total_actual_value
