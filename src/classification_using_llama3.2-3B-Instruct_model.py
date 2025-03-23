import nltk
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from time import sleep
import os
import json
import torch
from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_exponential
from utils import import_data,load_hf_token, get_class_combinations
import logging
from huggingface_hub import login
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

def convert_to_dict(response):
    """
    Convert the LLM response to a Python dictionary.
    """
    try:
        return eval(response.replace("```json", "").replace("```", ""))
    except Exception as e:
        logging.error(f"Error converting response to dict: {e}")
        raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Hugging Face pipeline
logging.info("Initializing Hugging Face pipeline...")
# Load the Hugging Face token from the .env file
hf_token = load_hf_token()
login(token=hf_token)

generator = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
logging.info("Pipeline initialized successfully.")

def message(classes, requirement):

    fallback_text = "If a requirement does not clearly fit into any of these categories, or if you are uncertain about its classification, assign it to 'Other'. " if "Other" in classes else ""

    valid_classes = [x for x in classes if x != "Other"]
    categories_text = ", ".join(valid_classes)
    json_categories = valid_classes + (["Other"] if "Other" in classes else [])

    system_message = {
        "role": "system",
        "content": (
            f"You are an expert Requirement Engineer. Your task is to classify smart home requirements into exactly one of the following categories: {categories_text}.\n"
            f"{fallback_text}\n\n"
            "Your response must be strictly in JSON format with two fields: 'application category' (one of the given categories) and 'rationale' (a brief explanation), following this structure:\n"
            "```json\n"
            "{"
            f'    "application category": "<one of {json_categories}>",\n'
            '    "rationale": "<brief explanation>"\n'
            "}"
            "```\n"
            "Do not provide multiple classifications, introduce new labels, or include any extra text or formatting."
        )
    }

   # Construct the user message
    user_message = {
        "role": "user",
        "content": f"Classify the following smart home requirement: {requirement}"
    }

    return [system_message, user_message]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def classify_using_llm(requirement, classes):
    """
    Classify a single requirement using the LLM.

    Args:
        requirement (str): Requirement text to classify.
        classes (list): List of valid categories.

    Returns:
        dict: Classification result with category and rationale.
    """
    messages = message(classes, requirement)
    outputs = generator(
        messages,
        temperature=0.1,
        max_length=1024,
    )
    output = outputs[0]["generated_text"][-1]['content'].strip()
    try:
        return convert_to_dict(output)
    except Exception as e:
        logging.error(f"Error processing requirement: {requirement}. Error: {e}")
        raise

def classify_requirements(df, classes):
    """
    Classify multiple requirements sequentially with progress tracking.

    Args:
        df (pd.DataFrame): DataFrame containing requirements.
        classes (list): List of valid categories.

    Returns:
        pd.DataFrame: DataFrame with classification results.
    """
    requirements = df['requirements'].tolist()

    for req in tqdm(requirements, total=len(requirements), desc="Processing"):
        try:
            output = classify_using_llm(req, classes)
            if output:
                df.loc[df['requirements'] == req, 'llm_assigned_category'] = output.get('application category', None)
                df.loc[df['requirements'] == req, 'llm_rationale'] = output.get('rationale', None)
        except Exception as e:
            logging.error(f"Error processing requirement '{req}': {e}")

    return df

def evaluate_classification(df):
    """
    Calculate macro precision, recall, and F1 score for the classification results.

    Args:
        df (pd.DataFrame): DataFrame containing true and predicted labels.

    Returns:
        dict: Dictionary containing precision, recall, and F1 scores.
    """
    # Create label mappings
    true_labels = df['class'].unique()
    pred_labels = df['llm_assigned_category'].unique()
    
    # Create label encoders
    true_label_map = {label: idx for idx, label in enumerate(true_labels)}
    pred_label_map = {label: idx for idx, label in enumerate(pred_labels)}
    
    # Convert labels to numeric
    y_true = df['class'].map(true_label_map)
    y_pred = df['llm_assigned_category'].map(pred_label_map)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1, 3)
    }
    
    logging.info(f"Classification Metrics: {metrics}")
    return metrics

def main():
    """
    Main function to classify requirements using LLM.
    """
    # Load and preprocess data using import_data from utils.py
    df, _, _, _, _, _, labels, _ = import_data()

    all_class_names = get_class_combinations(labels)
    mapping = {0: "Health", 1: "Energy", 2: "Entertainment", 3: "Safety", 4: "Other"}
    all_class_names = [[mapping[i] for i in classes] for classes in all_class_names]
    print(all_class_names)
    
    for classes in all_class_names:
        logging.info(f"Processing classes: {classes}")
        filtered_df = df[df['class'].isin(classes)].reset_index(drop=True)
        filtered_df["llm_assigned_category"] = ""
        filtered_df["llm_rationale"] = ""
        filtered_df = filtered_df.iloc[:3]
        # Classify requirements
        classified_df = classify_requirements(filtered_df, classes)

        # Evaluate classification performance
        metrics = evaluate_classification(classified_df)
        print(metrics)
        # Save results
        output_file = f"classification_({'_'.join(classes)})_using_llm.csv"
        classified_df.to_csv(output_file, index=False)
        logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
