import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from src.utils import logger
from config import OUTPUT_FILE, PROCESSED_DIR, SCORE_PATTERNS

# from Project_soccer.src.utils import logger
# from config import PROCESSED_DIR, SCORE_PATTERNS

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'score_pdediction_model.pkl')

def train_prediction_model():
    """Train a model to predict specific score patterns"""
    # Load the prediction dataset
    dataset_path = os.path.join(PROCESSED_DIR, 'prediction_dataset.csv')
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}. Run feature engineering first.")
        return None
    
    df = pd.read_csv(dataset_path)

    # Define features and target variables
    feature_columns = [
        'Home_2-1_frequency', 'Home_1-1_frequency', 'Home_1-2_frequency',
        'Home_Form_Last5', 'Home_Avg_Goals_Scored', 'Home_Avg_Goals_Conceded', 'Home_Avg_xG',
        'Away_2-1_frequency', 'Away_1-1_frequency', 'Away_1-2_frequency'
        'Away_Form_Last5', 'Away_Avg_Goals_Scored', 'Away_Avg_Goals_Conceded', 'Away_Avg_xG'
    ]

    # Create a multiclass target where 0=not in patterns, 1='2-1', 2='1-1', 3='1-2'
    df['score_class'] = 0
    for i, pattern in enumerate(SCORE_PATTERNS, 1):
        pattern_name = pattern['name']
        df.loc[df[f'is_{pattern_name}'] == 1, 'score_class'] = i

        x = df[feature_columns]
        y = df['score_class']

        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)

        # Evaluate the model
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['Other', '2-1', '1-1', '1-2']))

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        logger.info("\nFeature Importance:")
        logger.info(feature_importance)

        # Save the model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        
        return model
    
def predict_upcoming_match(home_team, away_team, season = 2024):
    """Predict the score for an upcoming match"""
    # Load the model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}. Train the model first.")
        return None
    
    model = joblib.load(MODEL_PATH)

    # Load team features 
    team_features_path = os.path.join(PROCESSED_DIR, 'team_features.csv')
    if not os.path.exists(team_features_path):
        logger.error(f"Team features not found at {team_features_path}.")
        return None
    
    team_features = pd.read_csv(team_features_path)

    # Get home team features
    home_features = team_features[
        (team_features['Team'] == home_team) &
        (team_features['Season'] == season)
    ]

    # Get away team features
    away_features = team_features[
        (team_features['Team'] == away_team) &
        (team_features['Season'] == season)
    ]

    if len(home_features) == 0 or len(away_features) == 0:
        logger.error(f"Features not found for {home_team} or {away_team} in season {season}")
        return None
    
    # Prepare features for prediction
    match_features = {
        'Home_2-1_frequency': home_features.iloc[0]['2-1_frequency'],
        'Home_1-1_frequency': home_features.iloc[0]['1-1_frequency'],
        'Home_1-2_frequency': home_features.iloc[0]['1-2_frequency'],
        'Home_Form_Last5': home_features.iloc[0]['Form_Last5'],
        'Home_Avg_Goals_Scored': home_features.iloc[0]['Avg_Goals_Scored'],
        'Home_Avg_Goals_Conceded': home_features.iloc[0]['Avg_Goals_Conceded'],
        'Home_Avg_xG': home_features.iloc[0]['Avg_xG'],
        'Away_2-1_frequency': away_features.iloc[0]['2-1_frequency'],
        'Away_1-1_frequency': away_features.iloc[0]['1-1_frequency'],
        'Away_1-2_frequency': away_features.iloc[0]['1-2_frequency'],
        'Away_Form_Last5': away_features.iloc[0]['Form_Last5'],
        'Away_Avg_Goals_Scored': away_features.iloc[0]['Avg_Goals_Scored'],
        'Away_Avg_Goals_Conceded': away_features.iloc[0]['Avg_Goals_Conceded'],
        'Away_Avg_xG': away_features.iloc[0]['Avg_xG']
    }

    # Make prediction
    x_pred = pd.DataFrame([match_features])
    prediction_class = model.predict(x_pred)[0]
    probabilities = model.predict_proba(x_pred)[0]

    # Map prediction class to score
    if prediction_class == 0:
        predicted_score = "Other score"
    else:
        predicted_score = SCORE_PATTERNS[prediction_class - 1]['name']

    # Format probabilities
    prob_dict = {
        'Other score': probabilities[0],
    }
    for i, pattern in enumerate(SCORE_PATTERNS, 1):
        prob_dict[pattern['name']] = probabilities[i]

    # Sort probabilities in descending order
    sorted_probs = {k: v for k, v in sorted(
        prob_dict.items(), key=lambda item: item[1], reverse=True)}
    
    result = {
        'home_team': home_team,
        'away_team': away_team,
        'most_likely_score': predicted_score,
        'probabilities': sorted_probs
    }

    return result

def batch_predict_matches(match_list):
    """Predict scores for multiple upcoming matches"""
    results = []
    for match in match_list:
        home_team = match['home_team']
        away_team = match['away_team']
        season = match.get('season', 2024)

        prediction = predict_upcoming_match(home_team, away_team, season)
        if prediction:
            results.append(prediction)

    return results
