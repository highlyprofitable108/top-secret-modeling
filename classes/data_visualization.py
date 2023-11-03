import os
import shap
import base64
import random
import logging
import scipy.stats
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore


class Visualization:
    def __init__(self, template_dir=None, target_variable="odd.spread_close"):
        self.template_dir = template_dir
        self.TARGET_VARIABLE = target_variable

    def visualize_simulation_results(self, simulation_results, most_likely_outcome, output, game_number, bins=50):
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
        individual_simulation_path = os.path.join(self.template_dir, f'/tests/simulation_distribution_results_game_{game_number:04d}.html')

        # Save the combined HTML for the individual game
        try:
            with open(individual_simulation_path, "w") as f:
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

    def generate_simulation_distribution_page(self, num_games):
        links_html = ""

        # Define CSS styles for the links and divs
        link_style = "color: blue; font-size: 18px; text-decoration: underline; margin-bottom: 10px;"
        div_style = "background-color: #f7f7f7; padding: 15px; border-radius: 5px; margin-bottom: 20px;"

        # Generate links and formatted outputs for all the individual game simulation results
        for i in range(1, num_games + 1):
            link = f'<a href="simulation_distribution_results_game_{i:04d}.html" target="_blank" style="{link_style}">Game {i} Simulation Results</a>'
            links_html += f"<div style='{div_style}'>{link}</div>"

        # Combine the links and formatted outputs into a single HTML page
        full_html = f"""
        <html>
        <head>
            <title>Simulation Distribution</title>
        </head>
        <body>
            <h2>Available Simulation Results:</h2>
            {links_html}
        </body>
        </html>
        """

        # Path to save the simulation distribution page
        simulation_distribution_path = os.path.join(self.template_dir, 'simulation_distribution.html')

        # Save the combined HTML
        try:
            with open(simulation_distribution_path, "w") as f:
                f.write(full_html)
        except IOError as e:
            # Handle potential file write issues
            print(f"Error writing to file: {e}")

    def generate_value_opportunity_page(self, num_games):
        links_html = ""

        # Define CSS styles for the links and divs
        link_style = "color: blue; font-size: 18px; text-decoration: underline; margin-bottom: 10px;"
        div_style = "background-color: #f7f7f7; padding: 15px; border-radius: 5px; margin-bottom: 20px;"

        # Generate links and formatted outputs for all the individual game simulation results
        for i in range(1, num_games + 1):
            link = f'<a href="value_opportunity_results_game_{i:04d}.html" target="_blank" style="{link_style}">Game {i} Value Opportunity</a>'
            links_html += f"<div style='{div_style}'>{link}</div>"

        # Combine the links and formatted outputs into a single HTML page
        full_html = f"""
        <html>
        <head>
            <title>Value Opportunity</title>
        </head>
        <body>
            <h2>Available Simulation Results:</h2>
            {links_html}
        </body>
        </html>
        """

        # Path to save the valuopportunity page
        value_opportunity_distribution_path = os.path.join(self.template_dir, 'value_opportunity_distribution.html')

        # Save the combined HTML
        try:
            with open(value_opportunity_distribution_path, "w") as f:
                f.write(full_html)
        except IOError as e:
            # Handle potential file write issues
            print(f"Error writing to file: {e}")        

    def compare_simulated_to_actual(self, simulation_results, actual_results):
        model_covered = 0

        for idx in range(len(simulation_results)):
            actual_difference = actual_results[idx]
            predicted_difference = simulation_results[idx]

            if actual_difference > np.mean(predicted_difference):
                model_covered += 1

        model_cover_rate = model_covered / len(simulation_results) * 100

        return model_cover_rate

    def determine_probability_of_win(self, predicted_difference_list, vegas_line):
        """
        Determine the probability of winning the bet based on the value of the point spread,
        especially when crossing key numbers.

        :param predicted_difference: the predicted point difference from the model
        :param vegas_line: the point spread from Vegas
        :return: the probability of winning the bet
        """
        predicted_difference = np.median(predicted_difference_list)
        # Define the value of each half-point around key numbers
        point_values = {
            # Key numbers and their immediate surroundings have higher values
            3.0: 1.0, 3.5: 0.9, 2.5: 0.9,
            7.0: 0.7, 7.5: 0.6, 6.5: 0.6,
            # Other half-points within 7 points of the spread
            6.0: 0.5, 4.5: 0.4, 5.5: 0.4, 4.0: 0.3, 5.0: 0.3,
            2.0: 0.2, 1.5: 0.15, 1.0: 0.1, 0.5: 0.05, 8.0: 0.1, 8.5: 0.05,
            # Values for points greater than 8.5
        }
        # Assign a value of 0.01 for any half-point greater than 8.5
        for half_point in range(9, 15):  # Assuming we don't go beyond 14.5
            point_values[half_point] = 0.01
            point_values[half_point + 0.5] = 0.01

        # Calculate the edge your model has over the Vegas line
        model_edge = predicted_difference - vegas_line

        # Check if the edge crosses a key number
        if (vegas_line < 7 and predicted_difference > 7) or (vegas_line > 7 and predicted_difference < 7):
            # Crossing the key number of 7 adds significant value
            crossing_value = 0.1  # This value should be determined based on historical data
        else:
            crossing_value = 0

        # Find the closest half-point value to the model edge
        closest_half_point = min(point_values.keys(), key=lambda x: abs(x - model_edge))
        value_adjustment = point_values.get(closest_half_point, 0.01)  # Default to 0.01 if not in the dictionary

        # Adjust the base probability by the value of the closest half-point and crossing value
        base_probability = 0.5  # Starting from a base probability of 50%
        adjusted_probability = base_probability + (value_adjustment * 0.1) + crossing_value

        # Ensure the probability is within reasonable bounds
        adjusted_probability = min(max(adjusted_probability, 0), 1)

        return adjusted_probability

    def calculate_ev(self, vegas_line, predicted_difference, potential_profit, amount_bet):
        # Use the model to predict the outcome or to determine the probability of winning
        P_win = self.determine_probability_of_win(predicted_difference, vegas_line)
        P_loss = 1 - P_win

        # Calculate EV
        ev = (P_win * potential_profit) - (P_loss * amount_bet)
        return ev

    def create_summary_dashboard(self, results_df, total_bets, recommendation_accuracy, average_ev_percent, total_actual_value):
        # Helper function to calculate win percentage and format records
        def format_records(records):
            records = records.reindex(['win', 'loss', 'push'], fill_value=0)
            win_pct = records['win'] / (records['win'] + records['loss']) if (records['win'] + records['loss']) > 0 else 0
            win_pct = "{:.2f}%".format(win_pct * 100)  # Format as percentage with two decimals
            records_df = records.to_frame().reset_index()
            records_df.columns = ['Bet Outcome', 'Count']  # Rename columns for clarity
            win_pct_row = pd.DataFrame([['win pct', win_pct]], columns=['Bet Outcome', 'Count'])
            return pd.concat([records_df, win_pct_row], ignore_index=True)

        # Calculate additional metrics
        overall_record = results_df['Bet Outcome'].value_counts()
        negative_ev_record = results_df[results_df['Expected Value (%)'] < 0]['Bet Outcome'].value_counts()
        positive_ev_record = results_df[results_df['Expected Value (%)'] > 0]['Bet Outcome'].value_counts()
        # positive_ev_2_5_record = results_df[results_df['Expected Value (%)'] > 2.5]['Bet Outcome'].value_counts()
        # positive_ev_5_record = results_df[results_df['Expected Value (%)'] > 5]['Bet Outcome'].value_counts()

        # Format records and calculate win percentage
        overall_record_df = format_records(overall_record)
        negative_ev_record_df = format_records(negative_ev_record)
        positive_ev_record_df = format_records(positive_ev_record)
        # positive_ev_2_5_record_df = format_records(positive_ev_2_5_record)
        # positive_ev_5_record_df = format_records(positive_ev_5_record)

        # Convert to HTML without the index
        overall_record_html = overall_record_df.to_html(index=False, classes='table')
        negative_ev_record_html = negative_ev_record_df.to_html(index=False, classes='table')
        positive_ev_record_html = positive_ev_record_df.to_html(index=False, classes='table')
        # positive_ev_2_5_record_html = positive_ev_2_5_record_df.to_html(index=False, classes='table')
        # positive_ev_5_record_html = positive_ev_5_record_df.to_html(index=False, classes='table')

        # Create a model vs Vegas odds chart
        fig = go.Figure(data=go.Scatter(
            x=results_df['Vegas Odds'],
            y=results_df['Modeling Odds'],
            mode='markers',
            marker=dict(size=10, opacity=0.5)
        ))

        # Set plot titles and labels
        fig.update_layout(
            title='Model vs. Vegas Odds',
            xaxis_title='Vegas Odds',
            yaxis_title='Modeling Odds',
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True)
        )

        # Define the path for saving the image
        chart_path = os.path.join(self.template_dir, 'model_vs_vegas_chart.png')

        # Save the figure as a PNG image
        fig.write_image(chart_path)

        # Convert the image to a Base64 string
        with open(chart_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()

        # Embed the Base64 string directly into the HTML
        embedded_image_html = f'<img src="data:image/png;base64,{encoded_string}" alt="Model vs Vegas Odds Chart">'

        # Create HTML content
        html_content = f"""
            <div class="container">
                <div class="stats-card">
                    <h2>Key Metrics</h2>
                    <p>Total Bets: <strong>{total_bets}</strong></p>
                    <p>Overall Accuracy: <strong>{recommendation_accuracy:.2f}%</strong></p>
                    <p>Average EV Percent: <strong>{average_ev_percent:.2f}%</strong></p>
                    <p>Total Actual Value: <strong>${total_actual_value:.2f}</strong></p>
                </div>
                <div class="stats-card">
                    <h2>Overall Record</h2>
                    {overall_record_html}
                </div>
                <div class="stats-card">
                    <h2>Positive EV Record</h2>
                    {positive_ev_record_html}
                </div>
                <div class="stats-card">
                    <h2>Negative EV Record</h2>
                    {negative_ev_record_html}
                </div>
            </div>
        """

        # Path to save the valuopportunity page
        summary_dash_path = os.path.join(self.template_dir, 'summary_dash.html')

        # Write the HTML content to a file
        with open(summary_dash_path, 'w') as f:
            f.write(html_content)

    def evaluate_and_recommend(self, simulation_results, historical_df):
        historical_df = historical_df.reset_index(drop=True)
        correct_recommendations = 0
        total_bets = 0
        total_ev = 0
        actual_covered = None

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(columns=['Date', 'Home Team', 'Vegas Odds', 'Modeling Odds', 'Away Team', 'Recommended Bet', 'Expected Value (%)', 'Home Points', 'Away Points', 'Result with Spread', 'Actual Covered', 'Bet Outcome', 'Actual Value ($)'])

        for idx, row in historical_df.iterrows():
            actual_home_points = row['summary.home.points']
            actual_away_points = row['summary.away.points']
            home_team = row['summary.home.name']
            away_team = row['summary.away.name']
            vegas_line = row["summary." + self.TARGET_VARIABLE]
            date = row['scheduled']

            predicted_difference = simulation_results[idx]

            # Check if the game has odss or if it's a future game
            if pd.isna(vegas_line):
                # Future game no spread
                recommended_bet = None
                ev_percentage = None
                bet_outcome = None
                actual_value = None
                actual_difference = None
            else:
                # Past game with actual results
                actual_difference = (actual_home_points + vegas_line) - actual_away_points

                # Calculate the recommendation based on the model
                reccommendation_calc = vegas_line - np.mean(predicted_difference)
                if reccommendation_calc < 0:
                    recommended_bet = "away"
                elif reccommendation_calc > 0:
                    recommended_bet = "home"
                elif reccommendation_calc == 0:
                    recommended_bet = "push"
                else:
                    recommended_bet = None

                # Calculate EV
                potential_profit = 100  # Assuming a winning bet returns $100
                amount_bet = 110  # Assuming a bet amount of $100
                ev = self.calculate_ev(vegas_line, predicted_difference, potential_profit, amount_bet)
                ev_percentage = (ev / amount_bet) * 100 # Express EV as a percentage of the bet
                total_ev += ev  # Accumulate the total expected value

                # If actual results are available:
                if actual_difference:
                    # Determine who actually covered based on spread odds.
                    if actual_difference < 0:
                        actual_covered = "away"
                    elif actual_difference > 0:
                        actual_covered = "home"
                    else:
                        actual_covered = "push"

                    # Check historical performance.
                    if recommended_bet == actual_covered:
                        correct_recommendations += 1
                        total_bets += 1
                        bet_outcome = "win"
                        actual_value = 100  # Profit from a winning bet
                    elif recommended_bet == "push" or actual_covered == "push":
                        bet_outcome = "push"
                        actual_value = 0  # No profit, no loss from a push
                    else:
                        total_bets += 1
                        bet_outcome = "loss"
                        actual_value = -110  # Loss from a losing bet

            # Append to results DataFrame
            new_row = pd.Series({
                'Date': date,
                'Home Team': home_team,
                'Vegas Odds': vegas_line,
                'Modeling Odds': np.median(predicted_difference),
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

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(results_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[results_df[col].tolist() for col in results_df.columns],  # Convert each column's values to a list
                    fill_color='lavender',
                    align='left'))
        ])

        # Update layout for a better visual appearance if needed
        fig.update_layout(
            title='Betting Results',
            title_x=0.5,
            margin=dict(l=10, r=10, t=30, b=10)  # Adjust margins to fit the layout
        )

        # Convert the Plotly figure to an HTML string
        betting_results_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Define paths to save the results
        betting_results_path = os.path.join(self.template_dir, 'betting_recommendation_results.html')

        # Save the combined HTML
        try:
            with open(betting_results_path, "w") as f:
                f.write(betting_results_html)
        except IOError as e:
            # Handle potential file write issues
            print(f"Error writing to file: {e}")

        recommendation_accuracy = 0
        average_ev_percent = 0
        total_actual_value = 0

        if total_bets:
            # Calculate the total actual value from all bets
            total_actual_value = results_df['Actual Value ($)'].sum()

            recommendation_accuracy = (correct_recommendations / total_bets) * 100
            average_ev = total_ev / total_bets
            average_ev_percent = (average_ev / 110) * 100  # Adjust 110 if your standard bet amount is different

        self.create_summary_dashboard(results_df, total_bets, recommendation_accuracy, average_ev_percent, total_actual_value)

        return recommendation_accuracy, average_ev_percent, total_actual_value

    def visualize_value_opportunity(self, simulation_results, perceived_value, game_number):
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
        value_opportunity_path = os.path.join(self.template_dir, f'/tests/value_opportunity_results_game_{game_number:04d}.html')

        # Save the combined HTML
        try:
            with open(value_opportunity_path, "w") as f:
                f.write(plotly_html_string)
        except IOError as e:
            # Handle potential file write issues
            print(f"Error writing to file: {e}")

        return value_opportunity

    def visualize_feature_importance(self, feature_importance_df):
        """Visualize feature importance using Plotly."""
        fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance',
                     color='Highlight', color_discrete_map={'Important': 'red', 'Related to Target': 'blue', 'Important and Related': 'purple', 'Just Data': 'gray'})
        feature_importance_path = os.path.join(self.template_dir, 'importance.html')
        fig.write_html(feature_importance_path)
        return feature_importance_path

    def visualize_coefficients(self, coef_df):
        """Visualize feature coefficients using Plotly."""
        # Create a bar chart using Plotly
        fig = px.bar(coef_df,
                     x='Feature',
                     y='Coefficient',
                     title='Feature Coefficients',
                     labels={'Coefficient': 'Coefficient Value', 'Feature': 'Feature Name'},
                     color='Coefficient',  # Color bars by coefficient value
                     color_continuous_scale='balance'  # Use a diverging color scale
                     )

        # Adjust layout for better visualization
        fig.update_layout(barmode='relative', showlegend=False)

        coef_df = os.path.join(self.template_dir, 'importance.html')
        fig.write_html(coef_df)
        return coef_df

    # ENHANCE AND OPTIMIZE EDA OUTPUTS
    def plot_interactive_correlation_heatmap(self, df, importances):
        """Plots an interactive correlation heatmap using Plotly."""
        try:
            # Save the target variable column
            df_target = df[self.TARGET_VARIABLE].copy()

            # If df has more than 50 columns, select only the 50 most important ones
            if df.shape[1] > 50:
                X = df.drop(columns=[self.TARGET_VARIABLE])

                # Get the top 50 features based on importance
                top_50_features = X.columns[importances.argsort()[-50:]]

                # Filter df to only include these top 50 features and the target variable
                df = df[top_50_features]

                # Add the target variable back to the dataframe
                df[self.TARGET_VARIABLE] = df_target

            corr = df.corr()

            # Using Plotly to create an interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                hoverongaps=False,
                colorscale=[[0, "red"], [0.5, "white"], [1, "blue"]],  # Setting colorscale with baseline at 0
                zmin=-.8,  # Setting minimum value for color scale
                zmax=.8,   # Setting maximum value for color scale
                showscale=True,  # Display color scale bar
            ))

            # Adding annotations to the heatmap
            annotations = []
            for i, row in enumerate(corr.values):
                for j, value in enumerate(row):
                    annotations.append(
                        {
                            "x": corr.columns[j],
                            "y": corr.columns[i],
                            "font": {"color": "black"},
                            "text": str(round(value, 2)),
                            "xref": "x1",
                            "yref": "y1",
                            "showarrow": False
                        }
                    )
            fig.update_layout(annotations=annotations, title='Correlation Heatmap')

            heatmap_path = os.path.join(self.template_dir, 'interactive_heatmap.html')
            fig.write_html(heatmap_path)

            return heatmap_path
        except Exception as e:
            logging.error(f"Error generating interactive correlation heatmap: {e}")
            return None

    def plot_interactive_histograms(self, df):
        """Plots interactive histograms for each numerical column using Plotly."""
        histograms_dir = os.path.join(self.template_dir, 'interactive_histograms')
        os.makedirs(histograms_dir, exist_ok=True)

        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            fig = px.histogram(df, x=col, title=f'Histogram of {col}', nbins=30)
            fig.write_html(os.path.join(histograms_dir, f'{col}_histogram.html'))

        return histograms_dir

    def plot_boxplots(self, df):
        """Plots box plots for each numerical column in the dataframe."""
        try:
            # Create a directory to store all boxplot plots
            boxplots_dir = os.path.join(self.static_dir, 'boxplots')
            os.makedirs(boxplots_dir, exist_ok=True)

            # Loop through each numerical column and create a boxplot
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                fig = px.box(df, y=col, title=f'Boxplot of {col}')
                fig.write_html(os.path.join(boxplots_dir, f'{col}_boxplot.html'))

            logging.info("Boxplots generated successfully")
            return boxplots_dir

        except Exception as e:
            logging.error(f"Error generating boxplots: {e}")
            return None

    def generate_descriptive_statistics(self, df):
        """Generates descriptive statistics for each column in the dataframe and saves it as an HTML file."""
        try:
            # Generating descriptive statistics
            descriptive_stats = df.describe(include='all')

            # Transposing the DataFrame
            descriptive_stats = descriptive_stats.transpose()

            # Saving the descriptive statistics to an HTML file
            descriptive_stats_path = os.path.join(self.template_dir, 'descriptive_statistics.html')
            descriptive_stats.to_html(descriptive_stats_path, classes='table table-bordered', justify='center')

            return descriptive_stats_path
        except Exception as e:
            logging.error(f"Error generating descriptive statistics: {e}")
            return None

    def generate_data_quality_report(self, df):
        """Generates a data quality report for the dataframe and saves it as an HTML file."""
        try:
            # Initializing an empty dictionary to store data quality metrics
            data_quality_report = {}

            # Checking for missing values
            data_quality_report['missing_values'] = df.isnull().sum()

            # Checking for duplicate rows
            data_quality_report['duplicate_rows'] = df.duplicated().sum()

            # Checking data types of each column
            data_quality_report['data_types'] = df.dtypes

            # Checking for outliers using Z-score
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data_quality_report['outliers'] = df[numeric_cols].apply(lambda x: np.abs(zscore(x)) > 3).sum()

            # Converting the dictionary to a DataFrame
            data_quality_df = pd.DataFrame(data_quality_report)

            # Saving the data quality report to an HTML file
            data_quality_report_path = os.path.join(self.template_dir, 'data_quality_report.html')
            data_quality_df.to_html(data_quality_report_path, classes='table table-bordered', justify='center')

            return data_quality_report_path
        except Exception as e:
            logging.error(f"Error generating data quality report: {e}")
            return None

    def visualize_shap_summary(self, shap_values, explainer, X):
        """Visualize SHAP summary plot and save as HTML."""

        # Save the SHAP summary plot as an HTML file
        shap_summary_path = os.path.join(self.template_dir, 'shap_summary.html')
        shap.save_html(shap_summary_path, shap.force_plot(explainer.expected_value, shap_values, X))

        random_indices = random.sample(range(X.shape[0]), 20)

        saved_paths = []

        for idx, instance_index in enumerate(random_indices, start=1):
            # Create a SHAP force plot for the specific instance
            force_plot = shap.force_plot(explainer.expected_value, shap_values[instance_index], X.iloc[instance_index])

            # Save the plot as an HTML file
            force_plot_path = os.path.join(self.template_dir, f'shap_force_plot_{idx:02}.html')
            shap.save_html(force_plot_path, force_plot)

            saved_paths.append(force_plot_path)
