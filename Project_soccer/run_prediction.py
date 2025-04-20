#!/usr/bin/env python3

import argparse
import json
from Project_soccer.prediction_model import train_prediction_model, predict_upcoming_match, batch_predict_matches
from src.utils import logger #type: ignore

def parse_args():
    parser = argparse.ArgumentParser(description='Predict soccer match scores.')

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train model command
    train_parser = subparsers.add_parser('train', help='Train the prediction model')

    # Predict single match command
    predict_parser = subparsers.add_parser('predict', help='Predict a single match')
    predict_parser.add_argument('home_team', help='Home team name')
    predict_parser.add_argument('away_team', help='Away team name')
    predict_parser.add_argument('--season', type=int, default=2024, help='Season (default: 2024)')

    # Batch predict command
    batch_parser = subparsers.add_parser('batch', help='Predict multiple matches from a JSON file')
    batch_parser.add_argument('input_file', help='JSON file with matches to predict')
    batch_parser.add_argument('output_file', help='Output JSON file for predictions')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == 'train':
        logger.info("Training prediction model...")
        train_prediction_model()

    elif args.command == 'predict':
        logger.info(f"Predicting match: {args.home_team} vs {args.away_team}")
        prediction = predict_upcoming_match(args.home_team, args.away_team, args.season)

        if prediction:
            print("\n🏆 Match Prediction 🏆")
            print(f"{prediction['home_team']} vs {prediction['away_team']}")
            print(f"Most likely score: {prediction['most_likely_score']}")
            print("\nProbabilities:")
            for score, prob in prediction['probabilities'].items():
                print(f"  {score}: {prob:.2%}")

    elif args.command == 'batch':
        logger.info(f"Batch predicting matches from {args.input_file}")

        try:
            with open(args.input_file, 'r') as f:
                matches = json.load(f)

            predictions = batch_predict_matches(matches)

            with open(args.output_file, 'w') as f:
                json.dump(predictions, f, indent=4)

            logger.info(f"Batch predictions saved to {args.output_file}")
            print(f"Successfully predicted {len(predictions)} matches!")

        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
        
    else:
        print("Please specify a command: train, predict, or batch")
        print("For help, use: python run_prediction.py -h")

if __name__ =="__main__":
    main()