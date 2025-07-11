import pandas as pd
import os
from pandas.core.common import random_state
from pandas.core.frame import console
from sklearn.model_selection import train_test_split
import logging

#ensure the "logs" directory exists
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

#logging configuration
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG") # DEBUG -> Detailed information duting development and truobleshooting

console_handler = logging.StreamHandler() # Console(terminal) outpot
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

# Format the date in the log file
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a csv file."""
    try:
        df = pd.read_csv(data_url, encoding='latin-1')
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading data; %s", e)
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process the data."""
    try:
        df.drop(columns = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
        df.rename(columns = {"v1" : "target", "v2" : "text"}, inplace = True)
        logger.debug("Data Processed successfully")
        return df
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during processing: %s", e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test data sets"""

    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok = True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index = False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index = False)
        logger.debug("Train and Test Data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error while saving data: %s", e)
        raise

def main():
    """Main function"""
    try:
        test_size = 0.2
        data_path = "https://raw.github.com/Excergic/Dataset/main/spam.csv"
        df = load_data(data_url = data_path)
        final_df = process_data(df)
        train_df, test_df = train_test_split(final_df, test_size = test_size, random_state = 2)
        save_data(train_data = train_df, test_data = test_df, data_path = "../data")
    except Exception as e:
        logger.error("Failed to completem the data ingstion process: %s", e)
        print(f"error: {e}")

if __name__ == "__main__":
    main()