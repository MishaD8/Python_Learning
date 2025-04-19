import csv
import os
import pandas as pd
import numpy as np

from utils import logger
from config import OUTPUT_FILE, PROCESSED_DIR, SCORE_PATTERNS

def calculate_score_pattern_features():
    """Calculate score pattern features for all teams"""

    #Read the data into a pandas DataFrame
    df = pd.read_csv(OUTPUT_FILE)

    #Get unique teams and seasons
    teams = set(df['Home Team'].unique()) | set(df['Away Team'].unique())
    seasons = df['Season'].unique()

    # Initialize a dictionary to store results
    team_features = {}

    for season in season:
        season_df = df[df['Season'] == season]

        for team in teams:
            # Get matches where team played home
            home_matches = season_df[season_df['Home Team'] == team]
            # Get matches where team played away
            away_matches = season_df[season_df['Away Team'] == team]

            total_matches = len(home_matches) + len(away_matches)

            if total_matches == 0:
                continue

            # Calculate features for each pattern
            for pattern in SCORE_PATTERNS:
                pattern_count = 0

                # Count home matches with this pattern
                pattern_count += len(home_matches[
                    (home_matches['Goals Home'] == pattern['home_score']) &
                    (home_matches['Goals Away'] == pattern['away_score'])
                ])

                # Count away matches with this pattern
                pattern_count += len(away_matches[
                    (away_matches['Goals Home'] == pattern['home_score']) &
                    (away_matches['Goals Away'] == pattern['away_score'])
                ])

                # Calculate frequency
                pattern_frequency = pattern_count / total_matches

                # Store in dictionary
                if team not in team_features:
                    team_features[team] = {}

                if season not in team_features[team]:
                    team_features[team][season] = {}

                team_features[team][season][f"{pattern['name']}_frequency"] = pattern_frequency

    # Add additional features (team form, avg goals, etc.)
    team_features = add_team_form_features(df, team_features)

    # Save features to file
    output_file = os.path.join(PROCESSED_DIR, 'team_features.csv')

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        # Write header
        header = ['Team', 'Season', '2-1_frequency', '1-1_frequency', '1-2_frequency',
                    'Form_Last5', 'Avg_Goals_Scored', 'Avg_Goals_Conceded' 'Avg_xG']
        writer.writerow(header) # type: ignore

        # Write data
        for team in team_features:
            for season in team_features[team]:
                row = [
                    team,
                    season,
                    team_features[team][season].get('2-1_frequency', 0),
                    team_features[team][season].get('1-1_frequency', 0),
                    team_features[team][season].get('1-2_frequency', 0),
                    team_features[team][season].get('Form_Last5', 0),
                    team_features[team][season].get('Avg_Goals_Scored', 0),
                    team_features[team][season].get('Avg_Goals_Conceded', 0),
                    team_features[team][season].get('Avg_xG', 0)
                ]
                writer.writerow(row) # type: ignore

    logger.info(f"Feature engineering complete. Output saved to {output_file}")
    return team_features

def add_team_form_features(df, team_features):
    """Add team form and other useful features"""
    # Group by team and season
    for team in team_features:
        for season in team_features[team]:
            season_df = df[df['Season'] == season]

            # Get matches where team played
            team_matches = season_df[
                (season_df['Home Team'] == team) |
                (season_df['Away Team'] == team)
            ].sort_values(by='Date')

            if len(team_matches) == 0:
                continue

            # Calculate average goals scored
            goals_scored = []
            for _, match in team_matches.iterrows():
                if match['Home Team'] == team:
                    goals_scored.append(match['Goals Home'])
                else:
                    goals_scored.append(match['Goals Away'])

            # Calculate average goals conceded
            goals_scored = []
            for _, match in team_matches.iterrows():
                if match['Home Team'] == team:
                    goals_conceded.append(match['Goals Away']) # type: ignore
                else:
                    goals_conceded.append(match['Goals Home']) # type: ignore

            # Calculate average xG
            xg_values = []
            for _, match in team_matches.iterrows():
                if match['Home Team'] == team:
                    xg_values.append(match['xG Home'])
                else:
                    xg_values.append(match['xG Away'])

            # Calculate form (points in last 5 matches)
            form_points = []
            for _, match in team_matches.iterrows():
                if match['Winner'] == team:
                    form_points.append(3)
                elif match['Winner'] == 'Draw':
                    form_points.append(1)
                else:
                    form_points.append(0)

            # Last 5 matches form
            last_5_form = sum(form_points[-5:]) if len(form_points) >= 5 else sum(form_points)

            # Add to features dictionary
            team_features[team][season]['Avg_Goals_Scored'] = sum(goals_scored) / len(goals_scored)
            team_features[team][season]['Avg_Goals_Conceded'] = sum(goals_conceded) / len(goals_conceded) # type: ignore
            team_features[team][season]['Avg_xG'] = sum(xg_values) / len(xg_values)
            team_features[team][season]['Form_Last5'] = last_5_form

    return team_features

def prepare_match_prediction_dataset():
    """Prepare dataset for match score prediction"""
    # Read raw data
    df = pd.read_csv(OUTPUT_FILE)

    # Read team features
    team_features_file = os.path.join(PROCESSED_DIR, 'team_features.csv')
    team_features_df = pd.read_csv(team_features_file)

    # Create a new dataframe for prediction
    prediction_data = []

    for _, match in df.iterrows():
        season = match['Season']
        home_team = match['Home Team']
        away_team = match['Away Team']

        # Get home team features
        home_features = team_features_df[
            (team_features_df['Team'] == home_team) &
            (team_features_df['Season'] == season)
        ]

        # Get away team features
        away_features = team_features_df[
            (team_features_df['Team'] == away_team) &
            (team_features_df['Season'] == season)
        ]

        if len(home_features) == 0 or len(away_features) == 0:
            continue

        # Combine features
        match_data = {
            'Season': season,
            'Date': match['Date'],
            'Home_Team': home_team,
            'Away_Team': away_team,
            'Goals_Home': match['Goals Home'],
            'Goals_Away': match['Goals Away'],
            'Score': f"{match['Goals Home']}-{match['Goals Away']}",

            # Add home team features
            'Home_2-1_frequency': home_features.iloc[0]['2-1_frequency'],
            'Home_1-1_frequency': home_features.iloc[0]['1-1_frequency'],
            'Home_1-2_frequency': home_features.iloc[0]['1-2_frequency'],
            'Home_Form_Last5': home_features.iloc[0]['Form_Last5'],
            'Home_Avg_Goals_Scored': home_features.iloc[0]['Avg_Goals_Scored'],
            'Home_Avg_Goals_Conceded': home_features.iloc[0]['Avg_Goals_Conceded'],
            'Home_Avg_xG': home_features.iloc[0]['Avg_xG'],

            # Add away team features
            'Away_2-1_frequency': away_features.iloc[0]['2-1_frequency'],
            'Away_1-1_frequency': away_features.iloc[0]['1-1_frequency'],
            'Away_1-2_frequency': away_features.iloc[0]['1-2_frequency'],
            'Away_Form_Last5': away_features.iloc[0]['Form_Last5'],
            'Away_Avg_Goals_Scored': away_features.iloc[0]['Avg_Goals_Scored'],
            'Away_Avg_Goals_Conceded': away_features.iloc[0]['Avg_Goals_Conceded'],
            'Away_Avg_xG': away_features.iloc[0]['Avg_xG'],

            # Target variables (one-hot encoded for the specific scores)
            'is_2-1': 1 if match['Goals Home'] == 2 and match['Goals Away'] == 1 else 0,
            'is_1-1': 1 if match['Goals Home'] == 1 and match['Goals Away'] == 1 else 0,
            'is_1-2': 1 if match['Goals Home'] == 1 and match['Goals Away'] == 2 else 0
        }

        prediction_data.append(match_data)

    # Convert to DataFrame
    pred_df = pd.DataFrame(prediction_data)

    # Save to file
    output_file = os.path.join(PROCESSED_DIR, 'prediction_dataset.csv')
    pred_df.to_csv(output_file, index=False)

    logger.info(f"Prediction dataset prepared. Output saved to {output_file}")
    return pred_df

if __name__ == "__main__":
    logger.info("Starting feature engineering process")
    calculate_score_pattern_features()
    prepare_match_prediction_dataset()
    logger.info("Feature engineering complete")