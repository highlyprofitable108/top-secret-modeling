import base64
import logging
import os
from io import BytesIO
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import shap
import json
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import zscore


matplotlib.use('Agg')


class ModelVisualization:
    def __init__(self, template_dir=None, target_variable="odd.spread_close"):
        """
        Initializes the ModelVisualization class.

        Args:
            template_dir (str): Directory path to save generated reports.
            target_variable (str): The target variable used in the model.
        """
        self.template_dir = template_dir
        self.target_variable = target_variable

    def generate_descriptive_statistics(self, df):
        """
        Generates an HTML report of descriptive statistics for a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            None: Saves the report as an HTML file in the specified template directory.
        """
        try:
            # Calculating descriptive statistics
            descriptive_stats = df.describe(include='all').transpose()

            # Convert DataFrame to HTML with Tailwind CSS for styling
            html_content = descriptive_stats.to_html(classes="custom-table", border=0, index=True)

            # Full HTML structure with additional styling and structure
            full_html_content = f"""
            <div class="container mx-auto mt-5 bg-white p-8 rounded-lg shadow-md">
                <h2 class="text-2xl font-bold mb-4 text-gray-800">Descriptive Statistics</h2>
                {html_content}
            </div>
            """

            # Save the HTML content to a file
            descriptive_stats_path = os.path.join(self.template_dir, 'descriptive_statistics.html')
            with open(descriptive_stats_path, 'w') as file:
                file.write(full_html_content)

        except Exception as e:
            logging.error(f"Error generating descriptive statistics: {e}")

    def generate_data_quality_report(self, df):
        """
        Generates an HTML report for data quality, including missing values and outliers.

        Args:
            df (pd.DataFrame): The DataFrame to analyze.

        Returns:
            None: Saves the report as an HTML file in the specified template directory.
        """
        try:
            # Calculating data quality metrics
            data_quality_report = {
                'missing_values': df.isnull().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes,
                'outliers': df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(zscore(x)) > 3).sum()
            }
            data_quality_df = pd.DataFrame(data_quality_report)

            # Convert DataFrame to HTML with Tailwind CSS for styling
            html_content = data_quality_df.to_html(classes="custom-table", border=0, index=True)

            # Full HTML structure with additional styling and structure
            full_html_content = f"""
            <div class="container mx-auto mt-5 bg-white p-8 rounded-lg shadow-md">
                <h2 class="text-2xl font-bold mb-4 text-gray-800">Data Quality Report</h2>
                {html_content}
            </div>
            """

            # Save the HTML content to a file
            data_quality_report_path = os.path.join(self.template_dir, 'data_quality_report.html')
            with open(data_quality_report_path, 'w') as file:
                file.write(full_html_content)

        except Exception as e:
            logging.error(f"Error generating data quality report: {e}")

    def calculate_feature_coefficients(self, feature_columns, coefficients):
        """
        Generates an HTML report showing the feature coefficients from a model.

        Args:
            feature_columns (list): List of feature names.
            coefficients (list): List of coefficients corresponding to the features.

        Returns:
            None: Saves the report as an HTML file in the specified template directory.
        """
        try:
            # Preparing data for report
            coef_data = {
                'Feature': feature_columns,
                'Coefficient': coefficients
            }
            coef_df = pd.DataFrame(coef_data)

            # Sort by the absolute value of coefficients and convert to HTML
            html_content = coef_df.sort_values(by='Coefficient', key=abs, ascending=False).to_html(classes="custom-table", border=0, index=True)

            # Full HTML structure with additional styling and structure
            full_html_content = f"""
            <div class="container mx-auto mt-5 bg-white p-8 rounded-lg shadow-md">
                <h2 class="text-2xl font-bold mb-4 text-gray-800">Feature Coefficients</h2>
                {html_content}
            </div>
            """

            # Save the HTML content to a file
            feature_coef_path = os.path.join(self.template_dir, 'feature_coef_report.html')
            with open(feature_coef_path, 'w') as file:
                file.write(full_html_content)

        except Exception as e:
            logging.error(f"Error generating feature coefficients report: {e}")

    def correlation_heatmap(self, df, importances):
        """
        Generates an interactive correlation heatmap using Plotly.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            importances (list): List of feature importance scores.

        Returns:
            None: Saves the heatmap as an HTML file.
        """
        try:
            # Preserve the target variable for later use
            df_target = df[self.target_variable].copy()

            # Select top 50 most important features if there are more than 50 columns
            if df.shape[1] > 50:
                # Excluding target variable for feature importance analysis
                X = df.drop(columns=[self.target_variable])
                top_50_features = X.columns[importances.argsort()[-50:]]
                df = df[top_50_features]
                df[self.target_variable] = df_target

            # Compute the correlation matrix
            corr = df.corr()

            # Create Plotly Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                hoverongaps=False,
                colorscale=[[0, "red"], [0.5, "white"], [1, "blue"]],
                zmin=-.8,
                zmax=.8,
                showscale=True,
            ))

            # Annotations for heatmap values
            annotations = [{
                "x": corr.columns[j],
                "y": corr.columns[i],
                "font": {"color": "black"},
                "text": str(round(value, 2)),
                "xref": "x1",
                "yref": "y1",
                "showarrow": False
            } for i, row in enumerate(corr.values) for j, value in enumerate(row)]
            fig.update_layout(annotations=annotations, title='Correlation Heatmap')

            # Save heatmap as HTML
            heatmap_path = os.path.join(self.template_dir, 'interactive_heatmap.html')
            fig.write_html(heatmap_path)
        except Exception as e:
            logging.error(f"Error generating interactive correlation heatmap: {e}")

    def model_interpretation(self, estimator, X_train, feature_names):
        """
        Generates a SHAP summary plot for model interpretation.

        Args:
            estimator (model object): The trained model for which SHAP values are calculated.
            X_train (pd.DataFrame): The training data.
            feature_names (list): List of feature names.

        Returns:
            str: HTML representation of the SHAP summary plot image.
        """
        # Ensure X_train has feature names for SHAP analysis
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_names)

        # Initialize SHAP explainer and calculate SHAP values
        explainer = shap.Explainer(estimator, X_train)
        shap_values = explainer.shap_values(X_train)

        # Generate and save the SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)

        # Convert plot to Base64 encoded string for HTML embedding
        base64_encoded_result = base64.b64encode(img.getvalue()).decode('utf-8')
        return f'<div class="image-container"><img src="data:image/png;base64,{base64_encoded_result}" class="mx-auto" /></div>'

    def performance_metrics_summary(self, estimator, X_test, y_test):
        """
        Calculates and returns HTML representation of performance metrics.

        Args:
            estimator (model object): The trained model used for prediction.
            X_test (pd.DataFrame): Test data for prediction.
            y_test (pd.Series): True values for comparison.

        Returns:
            str: HTML table of performance metrics.
        """
        # Predict and calculate metrics
        y_pred = estimator.predict(X_test)
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }

        # Convert metrics to DataFrame and then to HTML
        metrics_df = pd.DataFrame([metrics])
        html_table = metrics_df.to_html(border=0, index=False)

        # Add Tailwind CSS styling to the HTML table
        styled_html = html_table.replace('<table ', '<table class="table-auto border-collapse border border-gray-400 " ')
        styled_html = styled_html.replace('<th>', '<th class="px-4 py-2">')
        styled_html = styled_html.replace('<td>', '<td class="border px-4 py-2">')

        return styled_html

    def check_data_leakage(self, df, feature_columns, target_column):
        """
        Checks for potential data leakage by evaluating correlations.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            feature_columns (list): List of feature columns.
            target_column (str): The target column name.

        Returns:
            dict: Insights regarding data leakage.
        """
        # Calculate correlations between features and target
        correlation_matrix = df[feature_columns + [target_column]].corr()
        high_correlation = correlation_matrix[target_column].abs().sort_values(ascending=False)

        # Construct insights dict, including high correlations and reminders for manual checks
        leakage_insights = {
            'High Correlation with Target': high_correlation,
            'Temporal Checks': 'Manual Check Required',
            'Data Processing Review': 'Manual Review Required'
        }

        # Add specific checks for temporal alignment if applicable

        return leakage_insights

    def log_decision(self, decision_description):
        """
        Logs a decision description to the decision log.

        Args:
            decision_description (str): Description of the decision made.
        """
        # Append the decision description to the decision log
        self.decision_log.append(decision_description)

    def document_model_decisions(self, file_name='model_decisions.json'):
        """
        Saves the logged decisions to a JSON file.

        Args:
            file_name (str): The name of the file to save the decisions.
        """
        # Save the decision log to a JSON file
        with open(file_name, 'w') as file:
            json.dump(self.decision_log, file, indent=4)

    def generate_consolidated_report(self, shap_html, performance_metrics_html, file_name='consolidated_model_report.html'):
        """
        Generates a consolidated HTML report of model analysis, including performance metrics and SHAP visualization.

        Args:
            shap_html (str): HTML content of the SHAP visualization.
            performance_metrics_html (str): HTML content of the performance metrics.
            file_name (str): The name of the file to save the report.

        Returns:
            None: Saves the report as an HTML file in the specified template directory.
        """
        # Full path to the consolidated report file
        full_file_path = os.path.join(self.template_dir, file_name)

        # Constructing HTML content for the consolidated report
        html_content = f"""
        <div class="container mx-auto mt-5 bg-white p-8 rounded-lg shadow-md">
            <h2 class="text-2xl font-bold mb-4 text-gray-800">Model Analysis Report</h2>

            <div class="mb-8">
                <h3 class="text-xl font-semibold mb-2 text-gray-700">Performance Metrics</h3>
                {performance_metrics_html}
            </div>

            <div class="mb-8">
                <h3 class="text-xl font-semibold mb-2 text-gray-700">Model Interpretation (SHAP)</h3>
                {shap_html}
            </div>

        </div>
        """

        # Write the HTML content to the file
        with open(full_file_path, 'w') as file:
            file.write(html_content)
