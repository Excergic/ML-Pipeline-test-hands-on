from cgi import test
import formatter
import os
import logging
from nltk.tokenize.sonority_sequencing import punctuation
from numpy.char import isalnum
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')

#Ensure the "logs" directory exists
log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

#logging configuration
logger = logging.getLogger("preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Transform text
def transform_text(text) :
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punkcuation and stemming.
    """
    ps = PorterStemmer()
    
    #convert to lowercase
    text = text.lower()

    #Tokenize the text
    text = nltk.word_tokenize(text) # ["congratulations!, you won lottary"] -> ["congratulations", "!", "you", "won", "lottary"]

    #remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()] # ["congratulations", "!", "you", "won", "lottary"] -> ["congratulations", "you", "won", "lottary"]

    #remove stopwords
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #["congratulations", "you", "won", "lottary"] -> ["congratulations", "won", "lottary"]

    #stemming
    text = [ps.stem(word) for word in text]

    #join the token
    return " ".join(text) # ["congratulations", "won", "lottary"] -> "congratulations won lottary"

def process_df(df, text_column="text", target_column="target") :
    """
    Preprocess the dataframe by applying the transform_text function to the text column.
    """
    try:
        logger.debug("Starting preprocessing for dataframe")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug("Taarget column encoded")

        #Revove dupolicate raw
        df = df.drop_duplicates(keep = "first")
        logger.debug("Duplicates removed")

        # Apply text transformation to the specific text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")
        return df
    
    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise

    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        raise

def main():
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """

    try:
        #fetch data from data/raw
        train_data = pd.read_csv("../data/raw/train.csv")  
        test_data = pd.read_csv("../data/raw/test.csv")
        logger.debug("Data loaded successfully")

        train_processed_data = process_df(train_data, text_column="text", target_column="target")
        test_processed_data = process_df(test_data, text_column="text", target_column="target")

        #store the data inside data/interim
        data_path = os.path.join("../data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.debug("Data processed and saved to %s", data_path)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("Empty DataFrame: %s", e)
    except Exception as e:
        logger.error("Failed to complete the data transformation process: %s", e)
        print(f"Error: {e}")
    
if __name__ == "__main__":
    main()