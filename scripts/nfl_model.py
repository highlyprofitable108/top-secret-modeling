import sqlite3
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import yaml

# Load the configuration
with open(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'), 'r') as stream:
    config = yaml.safe_load(stream)

# Define base directory for data
BASE_DIR = os.path.expandvars(config['default']['base_dir'])
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATABASE_PATH = os.path.join(DATA_DIR, config['database']['database_name'])


def preprocess_nfl_data(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Perform the join operation
    query = """
        SELECT c.*, p.power_rank
        FROM consolidated c
        LEFT JOIN power_rank p ON c.team_id = p.team_id AND DATE(c.game_date) = DATE(p.game_date)
    """
    df = pd.read_sql_query(query, conn)

    # Advanced Metrics
    # Defensive Efficiency
    df['sack_rate'] = df['defensive_sacks'] / df['attempts']

    # Play Diversity
    df['play_diversity_ratio'] = df['rush_plays'] / df['play_count']

    # Turnover Margin
    df['turnover_margin'] = (df['interceptions_made'] + df['fumble_recoveries']) - (df['interceptions'] + df['lost_fumbles'])

    # Special Teams Efficiency
    # df['field_goal_efficiency'] = df['field_goals_made'] / (df['attempts'] - df['field_goals_made'] + 1)  # +1 to avoid division by zero

    # Success Rate
    df['pass_success_rate'] = df['completions'] / df['attempts']
    df['rush_success_rate'] = (df['rush_attempts'] * df['avg_rush_yards'] >= 4).astype(int) / df['rush_attempts']

    # Drop rows with missing values for scoring_differential
    df.dropna(subset=['scoring_differential'], inplace=True)

    # Extract the derived metrics along with team_id and game_date into a new DataFrame
    columns_to_extract = ['team_id', 'game_date', 'sack_rate', 'play_diversity_ratio', 'turnover_margin', 'pass_success_rate', 'rush_success_rate']
    df_advanced = df[columns_to_extract]

    # Save the new DataFrame to the SQLite database as a new table
    df_advanced.to_sql('consolidated_advanced', conn, if_exists='replace', index=False)

    # Drop unneeded columns
    columns_to_keep = [
                        'scoring_differential', 'rating', 'rush_plays',
                        'avg_pass_yards', 'attempts', 'tackles_made',
                        'redzone_successes', 'turnovers', 'redzone_attempts',
                        'opponent_team_id', 'home_or_away',
                        'sack_rate', 'play_diversity_ratio', 'turnover_margin',
                        'pass_success_rate', 'rush_success_rate'
                       ]
    df = df[columns_to_keep]

    # Feature Selection
    numerical_features = ['rating', 'rush_plays', 'avg_pass_yards', 'attempts',
                          'tackles_made', 'redzone_attempts',
                          'redzone_successes', 'turnovers',
                          'sack_rate', 'play_diversity_ratio',
                          'turnover_margin', 'pass_success_rate',
                          'rush_success_rate'
                          ]
    categorical_features = ['opponent_team_id', 'home_or_away']

    # One-Hot Encode Categorical Features
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_features = encoder.fit_transform(df[categorical_features])
    df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)
    encoded_columns = df.columns

    # Drop rows with any NaN values
    df.dropna(inplace=True)

    # Normalize/Standardize Numerical Features
    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Split the Data
    X = df.drop('scoring_differential', axis=1)  # Features
    y = df['scoring_differential']  # Target

    print(df)

    # Splitting into train and temp sets (80% train, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=108)

    # Splitting the temp set into test and blind_test sets (50% test, 50% blind_test)
    X_test, X_blind_test, y_test, y_blind_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=108)

    # Save the Blind Test Set to a File
    blind_test_data = pd.concat([X_blind_test, y_blind_test], axis=1)
    blind_test_data.to_csv(os.path.join(DATA_DIR, 'blind_test_data.csv'), index=False)

    # Print the shape of the data
    print(f"Shape of the training data: {X_train.shape}")
    print(f"Shape of the test data: {X_test.shape}")
    print(f"Shape of the blind test data: {X_blind_test.shape}")

    # Print the first few rows of the training data for inspection
    print("\nFirst few rows of the training data:")
    print(X_train.head())

    return X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, encoder, encoded_columns


def train_and_evaluate(X_train, y_train, X_test, y_test, X_blind, y_blind):
    # Initialize and train the Linear Regression model
    model = RandomForestRegressor(n_estimators=100, random_state=108)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Load the blind test data
    df_blind = pd.read_csv(os.path.join(DATA_DIR, 'blind_test_data.csv'))

    # Separate features and target
    X_blind = df_blind.drop(columns=['scoring_differential'])
    y_blind = df_blind['scoring_differential']

    # Predict using the trained model
    y_pred_blind = model.predict(X_blind)

    # Evaluate the model's performance
    mae_blind = mean_absolute_error(y_blind, y_pred_blind)
    mse_blind = mean_squared_error(y_blind, y_pred_blind)
    r2_blind = r2_score(y_blind, y_pred_blind)

    print(f'Mean Absolute Error on Blind Test Data: {mae_blind}')
    print(f'Mean Squared Error on Blind Test Data: {mse_blind}')
    print(f'R^2 Score on Blind Test Data: {r2_blind}')

    return model


# Usage
X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, encoder, encoded_columns = preprocess_nfl_data(DATABASE_PATH)
model = train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test)

# Save the model and related files to the models directory
joblib.dump(model, os.path.join(MODEL_DIR, 'trained_nfl_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'data_scaler.pkl'))
joblib.dump(encoder, os.path.join(MODEL_DIR, 'data_encoder.pkl'))
joblib.dump(encoded_columns, os.path.join(MODEL_DIR, 'encoded_columns.pkl'))


feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
