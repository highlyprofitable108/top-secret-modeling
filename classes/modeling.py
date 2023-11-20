import os
import logging
import time
import shap
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib  # Moved to top as it's used in multiple methods

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Ensemble Classes
class SimpleAveragingEnsemble:
    def __init__(self, models):
        """
        Ensemble that averages predictions from multiple models.

        Args:
            models (list): List of models to be used in the ensemble.
        """
        self.models = models

    def predict(self, X):
        """
        Predicts the target using the average of predictions from all models.

        Args:
            X (array-like): Feature set for making predictions.

        Returns:
            array: Average predictions from all models.
        """
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)


class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        """
        Ensemble that uses predictions from base models as input for a meta-model.

        Args:
            base_models (list): List of base models.
            meta_model (model): The meta model to make final predictions.
        """
        self.base_models = base_models
        self.meta_model = meta_model

    def predict(self, X):
        """
        Predicts the target using a meta-model that takes as input the predictions of base models.

        Args:
            X (array-like): Feature set for making predictions.

        Returns:
            array: Predictions made by the meta-model.
        """
        meta_features = np.column_stack([model.predict(X) for model in self.base_models])
        return self.meta_model.predict(meta_features)


class WeightedAveragingEnsemble:
    def __init__(self, models, weights):
        """
        Ensemble that averages predictions from multiple models with given weights.

        Args:
            models (list): List of models to be used in the ensemble.
            weights (list): List of weights for each model's predictions.
        """
        self.models = models
        self.weights = weights

    def predict(self, X):
        """
        Predicts the target using weighted average of predictions from all models.

        Args:
            X (array-like): Feature set for making predictions.

        Returns:
            array: Weighted average predictions from all models.
        """
        predictions = [model.predict(X) for model in self.models]
        weighted_predictions = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_predictions


# Main Modeling class
class Modeling:
    def __init__(self, loaded_model=None, loaded_scaler=None, home_field_adjust=2.7, static_dir=None):
        """
        Class for managing machine learning models, including training, prediction, and evaluation.

        Args:
            loaded_model (model, optional): Pre-trained model, if available.
            loaded_scaler (scaler, optional): Pre-trained scaler, if available.
            home_field_adjust (float): Adjustment factor for home field advantage.
            static_dir (str, optional): Directory for saving models and other static files.
        """
        self.LOADED_MODEL = loaded_model
        self.LOADED_SCALER = loaded_scaler
        self.HOME_FIELD_ADJUST = home_field_adjust
        self.static_dir = static_dir

    def save_model(self, file_path):
        """
        Saves the model and scaler to a file.

        Args:
            file_path (str): Path where the model and scaler will be saved.
        """
        joblib.dump((self.LOADED_MODEL, self.LOADED_SCALER), file_path)

    def load_model(self, file_path):
        """
        Loads a model and scaler from a file.

        Args:
            file_path (str): Path to the file from which the model and scaler are loaded.
        """
        self.LOADED_MODEL, self.LOADED_SCALER = joblib.load(file_path)

    def monte_carlo_simulation(self, game_prediction_df, home_team, away_team, num_simulations=250):
        """
        Performs Monte Carlo simulations to predict game outcomes.

        Args:
            game_prediction_df (pd.DataFrame): DataFrame with game prediction data.
            home_team (str): Home team name.
            away_team (str): Away team name.
            num_simulations (int): Number of simulations to run.

        Returns:
            tuple: Simulation results, home team name, away team name.
        """
        logging.info(f"Starting Monte Carlo Simulation for {home_team} vs {away_team}...")
        
        # Identify and separate standard deviation columns for simulations
        stddev_columns = [col for col in game_prediction_df.columns if col.endswith('_stddev')]
        stddev_df = game_prediction_df[stddev_columns].copy()
        game_prediction_df = game_prediction_df.drop(columns=stddev_columns)

        simulation_results = []
        sampled_data_list = []  # Stores sampled data for each simulation

        start_time = time.time()
        with tqdm(total=num_simulations, ncols=1000, desc=f"Simulating {home_team} vs {away_team}", mininterval=10) as pbar:
            for sim_num in range(num_simulations):
                # Sampling new data for each simulation
                sampled_df = game_prediction_df.copy()
                for column in sampled_df.columns:
                    base = column.replace('_difference', '').replace('_ratio', '')
                    stddev_column = f"{base}_stddev"
                    if stddev_column in stddev_df.columns:
                        mean_value = sampled_df[column].iloc[0]
                        stddev_value = stddev_df[stddev_column].iloc[0]
                        sampled_df[column] = np.random.normal(mean_value, stddev_value)

                sampled_data_list.append(sampled_df)

                # Prepare data for prediction
                modified_df = sampled_df.dropna(axis=1, how='any')
                try:
                    scaled_df = self.LOADED_SCALER.transform(modified_df)
                    prediction = self.LOADED_MODEL.predict(scaled_df)
                    simulation_results.append(prediction[0])
                except Exception as e:
                    logging.error(f"Error during prediction in simulation {sim_num}: {e}")
                    continue

                pbar.update(1)  # Update progress bar
                if time.time() - start_time > 10:
                    pbar.set_postfix_str("Running simulations...")
                    start_time = time.time()

        logging.info(f"Finished Monte Carlo Simulation for {home_team} vs {away_team}.")

        # Save combined data and simulation results to CSV files
        combined_sampled_data = pd.concat(sampled_data_list, axis=0, ignore_index=True)
        combined_file_path = os.path.join(self.static_dir, 'combined_sampled_data.csv')
        simulation_df = pd.DataFrame(simulation_results, columns=['Simulation_Result'])
        simulation_file_path = os.path.join(self.static_dir, 'simulation_results.csv')
        # Method to write data to CSV could be refactored into a separate utility function
        self._write_to_csv(combined_sampled_data, combined_file_path)
        self._write_to_csv(simulation_df, simulation_file_path)

        logging.info("Monte Carlo Simulation Completed!")
        return simulation_results, home_team, away_team

    def compute_shap_values(self, model, X, model_type):
        """
        Computes SHAP values for model interpretation.

        Args:
            model (model object): The machine learning model.
            X (pd.DataFrame): The feature set for SHAP computation.
            model_type (str): The type of the model.

        Returns:
            tuple: SHAP values and the explainer object.
        """
        # Handling different types of models for SHAP value computation
        if model_type in ["random_forest", "gradient_boosting"]:
            explainer = shap.TreeExplainer(model)
        elif model_type in ["linear", "lasso", "ridge"]:
            explainer = shap.LinearExplainer(model, X)
        elif model_type == "svm":
            explainer = shap.KernelExplainer(model.predict, X)
        # elif model_type == "simple_averaging_ensemble":
        #    shap_values_list = [self.compute_shap_values(sub_model, X, self._determine_model_type(sub_model)) for sub_model in model]
        #    shap_values = np.mean(shap_values_list, axis=0)
        #    return shap_values, None
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        shap_values = explainer.shap_values(X)
        return shap_values, explainer

    # Additional helper methods for refactoring and utility
    def _write_to_csv(self, df, file_path):
        """
        Writes DataFrame to CSV, appending to existing file if present.

        Args:
            df (pd.DataFrame): DataFrame to write to CSV.
            file_path (str): Path to the CSV file.
        """
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            df.to_csv(file_path, index=False)

    def _determine_model_type(self, model):
        """
        Determines the type of a given model.

        Args:
            model (model object): The machine learning model.

        Returns:
            str: Type of the model.
        """
        # Logic to determine the model type based on its characteristics
        # Example: return type(model).__name__
        return "unknown"

    def analysis_explanation(self, range_of_outcomes, confidence_interval, most_likely_outcome, standard_deviation):
        """
        Generates a textual explanation of a statistical analysis.

        Args:
            range_of_outcomes (tuple): Low and high estimates.
            confidence_interval (tuple): Low and high values of the confidence interval.
            most_likely_outcome (float): Most likely outcome value.
            standard_deviation (float): Standard deviation of the outcomes.

        Returns:
            str: A formatted string explaining the analysis.
        """
        # A metaphorical explanation using the example of guessing the number of candies in a jar
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

    def identify_top_features(self, X, y, importances):
        """
        Identifies the top features based on importance scores and correlation with the target.

        Args:
            X (pd.DataFrame): Feature data.
            y (array-like): Target variable.
            importances (array-like): Importance scores for the features.

        Returns:
            tuple: Top features based on importance and correlation.
        """
        # Calculate correlations of features with the target
        correlations = X.corrwith(y).abs()
        top_20_percent = int(np.ceil(0.20 * len(importances)))

        # Identify top features based on importance and correlation
        top_importance_features = X.columns[importances.argsort()[-top_20_percent:]]
        top_correlation_features = correlations.nlargest(top_20_percent).index.tolist()
        return top_importance_features, top_correlation_features

    def train_model(self, X, y, model_type, grid_search_params=None):
        """
        Trains a model of a specified type with the given data.

        Args:
            X (pd.DataFrame): Feature data.
            y (array-like): Target variable.
            model_type (str): Type of model to train.
            grid_search_params (dict, optional): Parameters for grid search, if applicable.

        Returns:
            model: The trained model.
        """
        logging.info(f"Training model of type: {model_type}")

        # Method for each model type could be refactored into separate private methods for cleaner code
        if model_type == "random_forest":
            return self.train_random_forest(X, y)
        elif model_type == "linear":
            return self.train_linear_regression(X, y)
        elif model_type == "svm":
            return self.train_svm(X, y, grid_search_params)
        elif model_type == "gradient_boosting":
            return self.train_gradient_boosting(X, y, grid_search_params)
        elif model_type == "lasso":
            return self.train_lasso_regression(X, y, grid_search_params)
        elif model_type == "ridge":
            return self.train_ridge_regression(X, y, grid_search_params)
        elif model_type == "stacking_ensemble":
            base_models = [...]
            meta_model = GradientBoostingRegressor()
            return self.train_stacking_ensemble(base_models, meta_model, X, y)
        elif model_type == "stacking_ensemble":
            base_models = [
                self.train_random_forest(X, y, grid_search_params),
                self.train_linear_regression(X, y),
                self.train_svm(X, y, grid_search_params)
            ]
            meta_model = GradientBoostingRegressor()
            base_models, trained_meta_model = self.train_stacking_ensemble(base_models, meta_model, X, y)
            return StackingEnsemble(base_models, trained_meta_model)
        elif model_type == "simple_averaging_ensemble":
            models = [
                self.train_random_forest(X, y, grid_search_params),
                self.train_linear_regression(X, y),
                self.train_svm(X, y, grid_search_params),
                self.train_gradient_boosting(X, y, grid_search_params)
            ]
            return SimpleAveragingEnsemble(models)
        elif model_type == "weighted_averaging_ensemble":
            return self.train_weighted_averaging_ensemble(X, y, grid_search_params)
        else:
            raise ValueError(f"The model type '{model_type}' specified is not supported.")

    def train_random_forest(self, X, y):
        """
        Trains a RandomForestRegressor model, optionally with hyperparameter tuning.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.
            grid_search_params (dict, optional): Parameters for grid search.

        Returns:
            RandomForestRegressor or GridSearchCV object: Trained model.
        """
        logging.info("Training RandomForestRegressor with hyperparameter tuning...")
        grid_search_params = {
            'n_estimators': [100],
            'max_depth': [None, 10],
        }
        model = GridSearchCV(RandomForestRegressor(random_state=108), grid_search_params, cv=3, verbose=2)
        model.fit(X, y)
        return model

    def train_linear_regression(self, X, y):
        """
        Trains a Linear Regression model.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.

        Returns:
            LinearRegression: Trained model.
        """
        logging.info("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X, y)
        return model

    def train_svm(self, X, y, grid_search_params=None):
        """
        Trains a Support Vector Machine (SVM) for regression, optionally with hyperparameter tuning.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.
            grid_search_params (dict, optional): Parameters for grid search.

        Returns:
            SVR or GridSearchCV object: Trained model.
        """
        logging.info("Training SVM with hyperparameter tuning...")
        if grid_search_params is None:
            grid_search_params = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
            }
        model = GridSearchCV(SVR(), grid_search_params, cv=3, verbose=2)
        model.fit(X, y)
        return model

    def train_gradient_boosting(self, X, y, grid_search_params=None):
        """
        Trains a Gradient Boosting Regressor, optionally with hyperparameter tuning.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.
            grid_search_params (dict, optional): Parameters for grid search.

        Returns:
            GradientBoostingRegressor or GridSearchCV object: Trained model.
        """
        logging.info("Training Gradient Boosting Regressor with hyperparameter tuning...")
        if grid_search_params is None:
            grid_search_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
            }
        model = GridSearchCV(GradientBoostingRegressor(random_state=108), grid_search_params, cv=3, verbose=2)
        model.fit(X, y)
        return model

    def train_lasso_regression(self, X, y, grid_search_params=None):
        """
        Trains a Lasso Regression model with hyperparameter tuning.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.

        Returns:
            GridSearchCV object: Trained model with optimal alpha.
        """
        grid_search_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }
        lasso = GridSearchCV(Lasso(), grid_search_params, cv=5)
        lasso.fit(X, y)
        return lasso

    def train_ridge_regression(self, X, y, grid_search_params=None):
        """
        Trains a Ridge Regression model with hyperparameter tuning.

        Args:
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.

        Returns:
            GridSearchCV object: Trained model with optimal alpha.
        """
        grid_search_params = {
            'alpha': [0.01, 0.1, 1, 10, 100, 1000]
        }
        ridge = GridSearchCV(Ridge(), grid_search_params, cv=5)
        ridge.fit(X, y)
        return ridge

    def train_stacking_ensemble(self, base_models, meta_model, X, y):
        """
        Trains a stacking ensemble model.

        Args:
            base_models (list): List of base models.
            meta_model (model): Meta model to use for final predictions.
            X (pd.DataFrame): Feature set.
            y (pd.Series): Target variable.

        Returns:
            tuple: List of trained base models and the trained meta model.
        """
        logging.info("Training stacking ensemble...")
        # Train each of the base models
        for model in base_models:
            model.fit(X, y)

        # Use predictions of base models as features for the meta model
        meta_features = np.column_stack([model.predict(X) for model in base_models])
        cloned_meta_model = clone(meta_model)
        cloned_meta_model.fit(meta_features, y)

        return base_models, cloned_meta_model

    def retrain_model(self, new_data, target_column, **training_args):
        """
        Retrains the loaded model with new data.

        Args:
            new_data (pd.DataFrame): New dataset for retraining the model.
            target_column (str): Name of the target variable column in the new data.
            **training_args: Additional arguments for the fit method of the model.
        """
        # Splitting features and target variable
        X = new_data.drop(target_column, axis=1)
        y = new_data[target_column]
        # Retraining the loaded model
        self.LOADED_MODEL.fit(X, y, **training_args)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns, model_type, grid_search_params=None):
        """
        Trains and evaluates a model using training, test, and blind test datasets.

        Args:
            X_train, y_train: Training features and target variable.
            X_test, y_test: Test features and target variable.
            X_blind_test, y_blind_test: Blind test features and target variable.
            feature_columns (list): List of feature column names.
            model_type (str): Type of model to train.
            grid_search_params (dict, optional): Parameters for GridSearchCV.

        Returns:
            Trained model or None in case of an error.
        """
        logging.info("Training and evaluating the model...")

        try:
            # Convert numpy arrays to dataframes
            X_train_df = pd.DataFrame(X_train, columns=feature_columns)
            model = self.train_model(X_train_df, y_train, model_type, grid_search_params)

            # Evaluate model on test and blind test data
            for dataset, dataset_name in zip([(X_test, y_test), (X_blind_test, y_blind_test)], ['Test Data', 'Blind Test Data']):
                X_df, y_data = dataset
                y_pred = model.predict(pd.DataFrame(X_df, columns=feature_columns))
                mae, mse, r2 = mean_absolute_error(y_data, y_pred), mean_squared_error(y_data, y_pred), r2_score(y_data, y_pred)
                logging.info(f"Performance on {dataset_name}: MAE: {mae}, MSE: {mse}, R^2: {r2}")

            return model
        except Exception as e:
            logging.error(f"Error in train_and_evaluate: {e}")
            return None

    def evaluate_model(self, test_data, target_column):
        """
        Evaluates the loaded model on the given test data.

        Args:
            test_data (pd.DataFrame): Test dataset.
            target_column (str): Name of the target variable in the test dataset.

        Returns:
            float: Mean Squared Error of the model on the test data.
        """
        # Splitting features and target variable for evaluation
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
        predictions = self.LOADED_MODEL.predict(X_test)

        # Calculating Mean Squared Error
        mse = mean_squared_error(y_test, predictions)
        return mse
