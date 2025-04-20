#!/usr/bin/env python3
from src.incremental_update import update_recent_fixtures #type: ignore
from src.feature_engineering import calculate_score_pattern_features, prepare_match_prediction_dataset #type: ignore
from src.utils import logger #type: ignore

if __name__ == "__main__":
    logger.info("Starting incremental update workflow")

    logger.info("Step 1: Updating recent fixtures")
    updated_count = update_recent_fixtures(days_back = 7)

    if updated_count > 0:
        logger.info("Step 2: Regenerating team features")
        calculate_score_pattern_features()

        logger.info("Step 3: Updating prediction dataset")
        prepare_match_prediction_dataset()

        logger.info(f"Incremental update complete. Processed {updated_count} new fixtures.")
    else:
        logger.info("No new fixtures found. Skipping feature engineering steps.")

    logger.info("Incremental update workflow complete")
