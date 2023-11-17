import os
import logging
import time
import shap
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import gaussian_kde, norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Ensemble Classes
class SimpleAveragingEnsemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)


class StackingEnsemble:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def predict(self, X):
        meta_features = np.column_stack([model.predict(X) for model in self.base_models])
        return self.meta_model.predict(meta_features)


class WeightedAveragingEnsemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        weighted_predictions = sum(w * p for w, p in zip(self.weights, predictions))
        return weighted_predictions


# Modeling class
class Modeling:
    def __init__(self, loaded_model=None, loaded_scaler=None, home_field_adjust=2.7, static_dir=None):
        self.LOADED_MODEL = loaded_model
        self.LOADED_SCALER = loaded_scaler
        self.HOME_FIELD_ADJUST = home_field_adjust
        self.static_dir = static_dir

    # Utility Methods
    def save_model(self, file_path):
        import joblib
        joblib.dump((self.LOADED_MODEL, self.LOADED_SCALER), file_path)

    def load_model(self, file_path):
        import joblib
        self.LOADED_MODEL, self.LOADED_SCALER = joblib.load(file_path)

    def monte_carlo_simulation(self, game_prediction_df, num_simulations=250):
        logging.info("Starting Monte Carlo Simulation...")
        # Separate standard deviation columns into a new DataFrame
        stddev_columns = [col for col in game_prediction_df.columns if col.endswith('_stddev')]
        logging.info(f"Identified standard deviation columns: {stddev_columns}")

        stddev_df = game_prediction_df[stddev_columns].copy()
        # Remove stddev columns from the original DataFrame
        game_prediction_df = game_prediction_df.drop(columns=stddev_columns)
        
        simulation_results = []

        # List to store sampled_df for each iteration
        sampled_data_list = []

        start_time = time.time()
        with tqdm(total=num_simulations, ncols=1000) as pbar:  # Initialize tqdm with total number of simulations
            for sim_num in range(num_simulations):
                sampled_df = game_prediction_df.copy()

                for column in sampled_df.columns:
                    base = column.replace('_difference', '').replace('_ratio', '')
                    stddev_column = f"{base}_stddev"
                    if stddev_column in stddev_df.columns:
                        mean_value = sampled_df[column].iloc[0]
                        stddev_value = stddev_df[stddev_column].iloc[0]
                        sampled_value = np.random.normal(mean_value, stddev_value)
                        sampled_df[column] = sampled_value
                        logging.info(f"Sim {sim_num}, Column: {column}, Mean: {mean_value}, StdDev: {stddev_value}, Sampled: {sampled_value}")

                # Append sampled_df to the list
                sampled_data_list.append(sampled_df)

                modified_df = sampled_df.dropna(axis=1, how='any')
                scaled_df = self.LOADED_SCALER.transform(modified_df)
                logging.info(f"Sim {sim_num}, Scaled DataFrame for prediction: {scaled_df}")

                try:
                    prediction = self.LOADED_MODEL.predict(scaled_df)
                    simulation_results.append(prediction)
                    logging.info(f"Sim {sim_num}, Prediction: {prediction[0]}")
                except Exception as e:
                    logging.error(f"Error during prediction: {e}")

                pbar.update(1)  # Increment tqdm progress bar
                if time.time() - start_time > 10:
                    pbar.set_postfix_str("Running simulations...")
                    start_time = time.time()

        # Flatten the simulation_results if it's a list of arrays
        flat_simulation_results = np.array(simulation_results).flatten()

        # Now pass the flattened array to gaussian_kde
        # Ensure that flat_simulation_results is one-dimensional
        if flat_simulation_results.ndim == 1:
            kernel = gaussian_kde(flat_simulation_results)
            most_likely_outcome = flat_simulation_results[np.argmax(kernel(flat_simulation_results))]
        else:
            logging.error("Simulation results are not one-dimensional, cannot use gaussian_kde.")
        

        # Append simulation_results to the CSV files
        combined_sampled_data = pd.concat(sampled_data_list, axis=0, ignore_index=True)
        combined_file_path = os.path.join(self.static_dir, 'combined_sampled_data.csv')
        if os.path.exists(combined_file_path):
            combined_sampled_data.to_csv(combined_file_path, mode='a', header=False, index=False)
        else:
            combined_sampled_data.to_csv(combined_file_path, index=False)

        simulation_df = pd.DataFrame(flat_simulation_results, columns=['Simulation_Result'])
        simulation_file_path = os.path.join(self.static_dir, 'simulation_results.csv')
        if os.path.exists(simulation_file_path):
            simulation_df.to_csv(simulation_file_path, mode='a', header=False, index=False)
        else:
            simulation_df.to_csv(simulation_file_path, index=False)

        logging.info("Monte Carlo Simulation Completed!")
        time.sleep(60)
        return flat_simulation_results, most_likely_outcome

    def compute_shap_values(self, model, X, model_type):
        """Compute SHAP values for a given model and dataset based on model type."""

        # For tree-based models
        if model_type in ["random_forest", "gradient_boosting"]:
            explainer = shap.TreeExplainer(model)

        # For linear models
        elif model_type in ["linear", "lasso", "ridge"]:
            explainer = shap.LinearExplainer(model, X)

        # For SVM
        elif model_type == "svm":
            explainer = shap.KernelExplainer(model.predict, X)

        # For ensemble methods
        elif model_type == "simple_averaging_ensemble":
            shap_values_list = []
            for sub_model in model:
                sub_model_type = ...  # You'd need a way to determine the type of each sub-model
                sub_shap_values, _ = self.compute_shap_values(sub_model, X, sub_model_type)
                shap_values_list.append(sub_shap_values)
            # Average the SHAP values from each model in the ensemble
            shap_values = np.mean(shap_values_list, axis=0)
            return shap_values, None  # No specific explainer for ensemble method

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        shap_values = explainer.shap_values(X)
        return shap_values, explainer


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

    def identify_top_features(self, X, y, importances):
        """Identify the top features based on importance and correlation."""
        correlations = X.corrwith(y).abs()
        top_20_percent = int(np.ceil(0.20 * len(importances)))
        top_importance_features = X.columns[importances.argsort()[-top_20_percent:]]
        top_correlation_features = correlations.nlargest(top_20_percent).index.tolist()
        return top_importance_features, top_correlation_features

    # Model Training
    def train_model(self, X, y, model_type, grid_search_params=None):
        logging.info(f"Training model of type: {model_type}")

        if model_type == "random_forest":
            return self.train_random_forest(X, y)
        elif model_type == "linear":
            return self.train_linear_regression(X, y)
        elif model_type == "svm":
            return self.train_svm(X, y)
        elif model_type == "gradient_boosting":
            return self.train_gradient_boosting(X, y)
        elif model_type == "lasso":
            return self.train_lasso_regression(X, y)
        elif model_type == "ridge":
            return self.train_ridge_regression(X, y)
        elif model_type == "simple_averaging_ensemble":
            models = [
                self.train_random_forest(X, y),
                self.train_linear_regression(X, y),
                self.train_svm(X, y),
                self.train_gradient_boosting(X, y)
            ]
            return SimpleAveragingEnsemble(models)
        elif model_type == "stacking_ensemble":
            base_models = [
                self.train_random_forest(X, y, grid_search_params),
                self.train_linear_regression(X, y),
                self.train_svm(X, y, grid_search_params)
            ]
            meta_model = GradientBoostingRegressor()
            base_models, trained_meta_model = self.train_stacking_ensemble(base_models, meta_model, X, y)
            return StackingEnsemble(base_models, trained_meta_model)
        elif model_type == "weighted_averaging_ensemble":
            # Split the data into training and validation sets for model performance evaluation
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train models on the subset of the training data
            model1 = self.train_random_forest(X_train_sub, y_train_sub, grid_search_params)
            model2 = self.train_linear_regression(X_train_sub, y_train_sub)
            model3 = self.train_svm(X_train_sub, y_train_sub, grid_search_params)
            model4 = self.train_gradient_boosting(X_train_sub, y_train_sub, grid_search_params)

            # Evaluate models on the validation set
            models = [model1, model2, model3, model4]
            model_errors = [mean_squared_error(y_val, model.predict(X_val)) for model in models]

            # Calculate weights inversely proportional to errors (models with lower error get higher weight)
            weights = [1/error for error in model_errors]
            normalized_weights = [weight/sum(weights) for weight in weights]

            return WeightedAveragingEnsemble(models, normalized_weights)
        else:
            raise ValueError(f"The model type '{model_type}' specified is not supported.")

    def train_random_forest(self, X, y, grid_search_params):
        """Train a RandomForestRegressor with hyperparameter tuning."""
        logging.info("Training RandomForestRegressor with hyperparameter tuning...")
        # if not grid_search_params:
        grid_search_params = {
            'n_estimators': [100],
            'max_depth': [None, 10],
        }
        model = GridSearchCV(RandomForestRegressor(random_state=108), grid_search_params, cv=3, verbose=2)
        model.fit(X, y)
        return model

    def train_linear_regression(self, X, y):
        """Train a Linear Regression model."""
        logging.info("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X, y)
        return model

    def train_svm(self, X, y, grid_search_params):
        """Train a Support Vector Machine for regression with hyperparameter tuning."""
        logging.info("Training SVM with hyperparameter tuning...")
        if not grid_search_params:
            grid_search_params = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
            }
        model = GridSearchCV(SVR(), grid_search_params, cv=3, verbose=2)
        model.fit(X, y)
        return model

    def train_gradient_boosting(self, X, y, grid_search_params):
        """Train a Gradient Boosting Regressor with hyperparameter tuning."""
        logging.info("Training Gradient Boosting Regressor with hyperparameter tuning...")
        if not grid_search_params:
            grid_search_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
            }
        model = GridSearchCV(GradientBoostingRegressor(random_state=108), grid_search_params, cv=3, verbose=2)
        model.fit(X, y)
        return model

    def train_lasso_regression(self, X, y):
        grid_search_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }
        lasso = GridSearchCV(Lasso(), grid_search_params, cv=5)
        lasso.fit(X, y)
        return lasso

    def train_ridge_regression(self, X, y):
        grid_search_params = {
            'alpha': [0.01, 0.1, 1, 10, 100, 1000]
        }
        ridge = GridSearchCV(Ridge(), grid_search_params, cv=5)
        ridge.fit(X, y)
        return ridge

    def train_stacking_ensemble(self, base_models, meta_model, X, y):
        """
        Train a stacking ensemble.

        :param base_models: List of base models to be trained.
        :param meta_model: The meta model to be trained on top of the base models.
        :param X: Features.
        :param y: Target variable.
        :return: Trained base models and meta model.
        """
        logging.info("Training stacking ensemble...")

        # Train base models
        for model in base_models:
            model.fit(X, y)

        # Get predictions from base models to be used as features for the meta model
        meta_features = np.column_stack([model.predict(X) for model in base_models])

        # Train the meta model
        cloned_meta_model = clone(meta_model)
        cloned_meta_model.fit(meta_features, y)

        return base_models, cloned_meta_model

    def retrain_model(self, new_data, target_column, **training_args):
        X = new_data.drop(target_column, axis=1)
        y = new_data[target_column]
        self.LOADED_MODEL.fit(X, y, **training_args)

    # Evaluation Methods
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns, model_type, grid_search_params=None):
        """
        Train and evaluate the model on test and blind test data.

        :param X_train: Training features.
        :param y_train: Training target variable.
        :param X_test: Test features.
        :param y_test: Test target variable.
        :param X_blind_test: Blind test features.
        :param y_blind_test: Blind test target variable.
        :param feature_columns: List of feature column names.
        :param model_type: Type of the model to be trained.
        :param grid_search_params: Hyperparameters for GridSearchCV.
        :return: Trained model.
        """
        logging.info("Training and evaluating the model...")

        try:
            # Convert numpy arrays back to dataframes to preserve feature names
            X_train_df = pd.DataFrame(X_train, columns=feature_columns)

            # Train the model using the factory method
            model = self.train_model(X_train_df, y_train, model_type, grid_search_params)

            for dataset, dataset_name in zip([(X_test, y_test), (X_blind_test, y_blind_test)], ['Test Data', 'Blind Test Data']):
                X_df, y_data = dataset
                y_pred = model.predict(pd.DataFrame(X_df, columns=feature_columns))
                mae = mean_absolute_error(y_data, y_pred)
                mse = mean_squared_error(y_data, y_pred)
                r2 = r2_score(y_data, y_pred)
                logging.info(f"Performance on {dataset_name}: MAE: {mae}, MSE: {mse}, R^2: {r2}")

            return model
        except Exception as e:
            logging.error(f"Error in train_and_evaluate: {e}")
            return None

    def evaluate_model(self, test_data, target_column):
        from sklearn.metrics import mean_squared_error
        X_test = test_data.drop(target_column, axis=1)
        y_test = test_data[target_column]
        predictions = self.LOADED_MODEL.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse
