from src.data_collection import main as collect_data #type: ignore
from src.feature_engineering import calculate_score_pattern_features, prepare_match_prediction_dataset #type: ignore
from src.utils import logger #type: ignore

if __name__ == "__main__":
    logger.info("Starting full data collection and processing workflow")

    logger.info("Step 1: Collecting full dataset from all seasons")
    collect_data()

    logger.info("Step 2: Generating team features")
    calculate_score_pattern_features()

    logger.info("Step 3: Preparing prediction dataset")
    prepare_match_prediction_dataset()

    logger.info("Full data collection and processing workflow complete")
