import nltk
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
import os
import yaml
from dotenv import load_dotenv

def load_hf_token():
    """
    Load the Hugging Face token from the .env file.

    Returns:
        str: Hugging Face token.

    Raises:
        ValueError: If the token is not found in the .env file.
    """
    # Load environment variables from .env file
    secrets_config = get_config()
    
    base_dir = os.path.dirname(os.getcwd()) 
    full_config_path = os.path.join(base_dir,secrets_config['SECRETS_FILE_PATH'])
    load_dotenv(full_config_path)

    # Get the Hugging Face token from the environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in .env file. Please add 'HF_TOKEN=your_token' to the .env file.")
    
    return hf_token

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Suppress warnings
warnings.filterwarnings("ignore")

# Import preprocessing module
from preprocessing import Preprocessing
preprocess = Preprocessing()

def get_config(config_path="config/config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    
    # Get the absolute path to the config file
    base_dir = os.path.dirname(os.getcwd()) 
    full_config_path = os.path.join(base_dir,config_path)
    
    try:
        with open(full_config_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    
def import_data():
    """
    Load and preprocess the requirements data.

    Args:
        file_path (str): Path to the CSV file containing requirements data.

    Returns:
        tuple: A tuple containing:
            - DataFrame with processed requirements and class labels.
            - Processed corpus (list of tokenized requirements).
            - Vocabulary list.
            - Frequency distribution of vocabulary.
            - Weighted average of word frequencies per document.
    """
    config = get_config()
    file_path = config["DATA_FILE_PATH"]
    base_dir = os.path.dirname(os.getcwd()) 

    # Load data
    data = pd.read_csv(os.path.join(base_dir,file_path))
    data['requirements'] = data['feature'] + ", " + data['benefit'] + '.'

    # Create a DataFrame with requirements and class labels
    df = pd.DataFrame(
        list(zip(data['requirements'], data['application_domain'])),
        columns=['requirements', 'class']
    )

    # Map class labels to numeric values
    df['n_class'] = df['class']
    df['n_class'].replace(
        ['Health', 'Energy', 'Entertainment', 'Safety', 'Other'],
        [0, 1, 2, 3, 4],
        inplace=True
    )
    labels = df['n_class']
    namelabels = data['application_domain']
    
    # Preprocess requirements
    corpus, corp, all_vocab, freq, wavg = preprocess.preprocessing(df['requirements'])

    return df, corpus, corp, all_vocab, freq, wavg, labels, namelabels