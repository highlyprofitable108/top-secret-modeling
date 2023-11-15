import base64
import logging
import os
from io import BytesIO
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import shap
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import zscore


matplotlib.use('Agg')


class ModelVisualization:
    def __init__(self, template_dir=None, target_variable="odd.spread_close"):
        self.template_dir = template_dir
        self.target_variable = target_variable

    def generate_descriptive_statistics(self, df):
        try:
            descriptive_stats = df.describe(include='all').transpose()

            # Translating to HTML using Tailwind CSS classes
            html_content = descriptive_stats.to_html(classes="custom-table", border=0, index=True)

            # Full HTML structure
            full_html_content = f"""
            <div class="container mx-auto mt-5 bg-white p-8 rounded-lg shadow-md">
                <h2 class="text-2xl font-bold mb-4 text-gray-800">Descriptive Statistics</h2>
                {html_content}
            </div>
            """

            # Saving to an HTML file
            descriptive_stats_path = os.path.join(self.template_dir, 'descriptive_statistics.html')
            with open(descriptive_stats_path, 'w') as file:
                file.write(full_html_content)

        except Exception as e:
            logging.error(f"Error generating descriptive statistics: {e}")
            return None

    def generate_data_quality_report(self, df):
        try:
            data_quality_report = {
                'missing_values': df.isnull().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes,
                'outliers': df.select_dtypes(include=[np.number]).apply(lambda x: np.abs(zscore(x)) > 3).sum()
            }
            
            data_quality_df = pd.DataFrame(data_quality_report)

            # Translating to HTML using Tailwind CSS classes
            html_content = data_quality_df.to_html(classes="custom-table", border=0, index=True)

            # Full HTML structure
            full_html_content = f"""
            <div class="container mx-auto mt-5 bg-white p-8 rounded-lg shadow-md">
                <h2 class="text-2xl font-bold mb-4 text-gray-800">Data Quality Report</h2>
                {html_content}
            </div>
            """

            # Saving to an HTML file
            data_quality_report_path = os.path.join(self.template_dir, 'data_quality_report.html')
            with open(data_quality_report_path, 'w') as file:
                file.write(full_html_content)

        except Exception as e:
            logging.error(f"Error generating data quality report: {e}")
            return None

    def calculate_feature_coefficients(self, feature_columns, coefficients):
        try:
            coef_data = {
                'Feature': feature_columns,
                'Coefficient': coefficients
            }
            coef_df = pd.DataFrame(coef_data)
            
            html_content = coef_df.sort_values(by='Coefficient', key=abs, ascending=False).to_html(classes="custom-table", border=0, index=True)

            # Full HTML structure
            full_html_content = f"""
            <div class="container mx-auto mt-5 bg-white p-8 rounded-lg shadow-md">
                <h2 class="text-2xl font-bold mb-4 text-gray-800">Feature Coefficients</h2>
                {html_content}
            </div>
            """

            # Saving to an HTML file
            feature_coef_path = os.path.join(self.template_dir, 'feature_coef_report.html')
            with open(feature_coef_path, 'w') as file:
                file.write(full_html_content)

        except Exception as e:
            logging.error(f"Error generating data quality report: {e}")
            return None

    def correlation_heatmap(self, df, importances):
        """Plots an interactive correlation heatmap using Plotly."""
        try:
            # Save the target variable column
            df_target = df[self.target_variable].copy()

            # If df has more than 50 columns, select only the 50 most important ones
            if df.shape[1] > 50:
                X = df.drop(columns=[self.target_variable])

                # Get the top 50 features based on importance
                top_50_features = X.columns[importances.argsort()[-50:]]

                # Filter df to only include these top 50 features and the target variable
                df = df[top_50_features]

                # Add the target variable back to the dataframe
                df[self.target_variable] = df_target

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
        except Exception as e:
            logging.error(f"Error generating interactive correlation heatmap: {e}")
            return None

    def model_interpretation(self, estimator, X_train, feature_names):
        # Ensure X_train is a DataFrame with feature names
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=feature_names)

        # Initialize the SHAP explainer
        explainer = shap.Explainer(estimator, X_train)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_train)

        # Visualization: Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, show=False)
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close()
        img.seek(0)

        # Convert the image to Base64 and decode
        base64_encoded_result = base64.b64encode(img.getvalue()).decode('utf-8')

        # Return the image in HTML
        return f'<div class="image-container"><img src="data:image/png;base64,{base64_encoded_result}" class="mx-auto" /></div>'

    def performance_metrics_summary(self, estimator, X_test, y_test):
        y_pred = estimator.predict(X_test)

        # Regression metrics
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }

        metrics_df = pd.DataFrame([metrics])

        # Convert DataFrame to HTML
        html_table = metrics_df.to_html(border=0, index=False)

        # Custom styling using Tailwind CSS
        styled_html = html_table.replace('<table ', '<table class="table-auto border-collapse border border-gray-400 " ')
        styled_html = styled_html.replace('<th>', '<th class="px-4 py-2">')
        styled_html = styled_html.replace('<td>', '<td class="border px-4 py-2">')

        return styled_html
    
    def check_data_leakage(self, df, feature_columns, target_column):
        # Check for direct correlation between features and target
        correlation_matrix = df[feature_columns + [target_column]].corr()
        high_correlation = correlation_matrix[target_column].abs().sort_values(ascending=False)

        # Check temporal alignment if necessary
        # [Add any time-based checks here]

        # Manual inspection reminders
        leakage_insights = {
            'High Correlation with Target': high_correlation,
            'Temporal Checks': 'Manual Check Required',
            'Data Processing Review': 'Manual Review Required'
        }

        # This can be more elaborate based on specific checks you implement
        return leakage_insights

    def log_decision(self, decision_description):
        self.decision_log.append(decision_description)

    def document_model_decisions(self, file_name='model_decisions.json'):
        with open(file_name, 'w') as file:
            json.dump(self.decision_log, file, indent=4)

    def generate_consolidated_report(self, feature_coefficients_html, shap_html, performance_metrics_html, file_name='consolidated_model_report.html'):
        # Full path to the file
        full_file_path = os.path.join(self.template_dir, file_name)
    
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

        with open(full_file_path, 'w') as file:
            file.write(html_content)
