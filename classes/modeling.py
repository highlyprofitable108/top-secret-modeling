import os
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde, t, norm


class Modeling:
    def __init__(self, loaded_model, loaded_scaler, home_field_adjust, static_dir):
        self.LOADED_MODEL = loaded_model
        self.LOADED_SCALER = loaded_scaler
        self.HOME_FIELD_ADJUST = home_field_adjust
        self.static_dir = static_dir

    def monte_carlo_simulation(self, df, standard_deviation_df, num_simulations=500):
        logging.info(df.head())
        logging.info("Starting Monte Carlo Simulation...")

        simulation_results = []

        # List to store sampled_df for each iteration
        sampled_data_list = []

        start_time = time.time()
        with tqdm(total=num_simulations, ncols=100) as pbar:  # Initialize tqdm with total number of simulations
            for _ in range(num_simulations):
                sampled_df = df.copy()
                for column in sampled_df.columns:
                    base_column = column.replace('_difference', '').replace('_ratio', '')
                    if base_column in standard_deviation_df.columns:
                        mean_value = df[column].iloc[0]
                        stddev_value = standard_deviation_df[base_column].iloc[0]
                        sampled_value = np.random.normal(mean_value, stddev_value)
                        sampled_df[column] = sampled_value

                # Append sampled_df to the list
                sampled_data_list.append(sampled_df)

                modified_df = sampled_df.dropna(axis=1, how='any')
                scaled_df = self.LOADED_SCALER.transform(modified_df)

                try:
                    prediction = self.LOADED_MODEL.predict(scaled_df)
                    adjusted_prediction = prediction[0] + self.HOME_FIELD_ADJUST
                    simulation_results.append(adjusted_prediction)
                except Exception as e:
                    logging.error(f"Error during prediction: {e}")

                pbar.update(1)  # Increment tqdm progress bar
                if time.time() - start_time > 10:
                    pbar.set_postfix_str("Running simulations...")
                    start_time = time.time()

        # After obtaining simulation_results
        kernel = gaussian_kde(simulation_results)
        most_likely_outcome = simulation_results[np.argmax(kernel(simulation_results))]

        # Append simulation_results to the CSV files
        combined_sampled_data = pd.concat(sampled_data_list, axis=0, ignore_index=True)
        combined_file_path = os.path.join(self.static_dir, 'combined_sampled_data.csv')
        if os.path.exists(combined_file_path):
            combined_sampled_data.to_csv(combined_file_path, mode='a', header=False, index=False)
        else:
            combined_sampled_data.to_csv(combined_file_path, index=False)

        simulation_df = pd.DataFrame(simulation_results, columns=['Simulation_Result'])
        simulation_file_path = os.path.join(self.static_dir, 'simulation_results.csv')
        if os.path.exists(simulation_file_path):
            simulation_df.to_csv(simulation_file_path, mode='a', header=False, index=False)
        else:
            simulation_df.to_csv(simulation_file_path, index=False)


        logging.info("Monte Carlo Simulation Completed!")
        return simulation_results, most_likely_outcome

    def compute_confidence_interval(self, data, confidence=0.95):
        """Compute the confidence interval for a given dataset."""
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), np.std(a)/np.sqrt(n)

        # Use t-distribution for small sample sizes and normal distribution for larger sample sizes
        if n < 30:
            h = se * t.ppf((1 + confidence) / 2., n-1)
        else:
            h = se * norm.ppf((1 + confidence) / 2.)

        return m-h, m+h

    def analyze_simulation_results(self, simulation_results):
        """
        Analyzes the simulation results to compute the range of outcomes, 
        standard deviation, and the most likely outcome.
        """

        # Constants to define the bounds for filtering the results
        LOWER_PERCENTILE = 0.1
        UPPER_PERCENTILE = 0.9

        # Calculate the lower and upper bounds based on percentiles
        lower_bound_value = np.percentile(simulation_results, LOWER_PERCENTILE * 100)
        upper_bound_value = np.percentile(simulation_results, UPPER_PERCENTILE * 100)

        # Filter the results based on the calculated bounds
        filtered_results = [result for result in simulation_results if lower_bound_value <= result <= upper_bound_value]

        # Save raw simulation results to a CSV file
        pd.DataFrame(simulation_results, columns=['Simulation_Result']).to_csv(os.path.join(self.static_dir, 'simulation_results.csv'), index=False)

        # Calculate the range of outcomes based on the filtered results
        range_of_outcomes = (min(filtered_results), max(filtered_results))

        # Calculate the standard deviation based on the filtered results
        standard_deviation = np.std(filtered_results)

        # Calculate confidence intervals
        confidence_interval = self.compute_confidence_interval(simulation_results)

        return range_of_outcomes, standard_deviation, confidence_interval

    def analysis_explanation(self, range_of_outcomes, confidence_interval, most_likely_outcome, standard_deviation):
        explanation = """
            Let's imagine we're trying to guess how many candies are in a big jar!

            We think there might be between {low_guess:.2f} and {high_guess:.2f} candies.
            We're pretty sure (like, 95% sure!) that the number of candies is between {low_confidence:.2f} and {high_confidence:.2f}.
            Our best guess is that there are {most_likely:.2f} candies in the jar.
            Our guesses are kind of {spread_out}. That number is {std_dev:.2f}.

            Now, let's see if we're right!
        """.format(
            low_guess=range_of_outcomes[0],
            high_guess=range_of_outcomes[1],
            low_confidence=confidence_interval[0],
            high_confidence=confidence_interval[1],
            most_likely=most_likely_outcome,
            spread_out="all over the place" if standard_deviation > 2 else "close together",
            std_dev=standard_deviation
        )
        return explanation
