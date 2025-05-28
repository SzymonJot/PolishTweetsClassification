import re
from sklearn.utils import resample
import torch
from transformers import AutoTokenizer
import spacy
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import csv
from EMOJI_TO_POLISH import emoji_to_polish
from POLISH_STOP_WORDS import polish_stopwords
from sklearn.model_selection import train_test_split
nlp = spacy.load("pl_core_news_lg")

def replace_emoji(text: str) -> str:
    """
    Replaces emojis in a given text based on the emoji dictionary provided.

    Args:
        text (str): The input string containing emojis.
        emoji_dict (dict): A dictionary where keys are emojis and values are replacements.

    Returns:
        str: The text with emojis replaced.
    """
    for emoji, replacement in emoji_to_polish.items():
        text = text.replace(emoji, f" {replacement}")
    return text

def preprocess_tweet(tweet):
    # Replace URLs, mentions, and hashtags with spaces
    tweet = re.sub(r'http\S+|www\S+|https\S+', ' ', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+|#\w+', ' ', tweet)

    # Replace underscores with spacess
    tweet = re.sub(r'_', ' ', tweet)

    # Remove unwanted characters, keeping letters, numbers, and spaces
    tweet = re.sub(r'[^a-zA-ZĄąĆćĘęŁłŃńÓóŚśŹźŻż0-9\s]', ' ', tweet)

    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()

    return tweet

def create_tokenize_function(tokenizer):
  def tokenize_function(examples):
    result = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    result['labels'] = [int(label) for label in examples['label']] 
    return result
  return tokenize_function

def remove_stops(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in polish_stopwords]
        return ' '.join(filtered_words)
    return text

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def analyze_sentiment(text,tokenizer,model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    sentiment_id = torch.argmax(probabilities, dim=1).item()
    sentiment = ["Negative", "Neutral", "Positive"][sentiment_id]
    confidence = probabilities[0][sentiment_id].item()
    return sentiment, confidence

def balance_text_data(df, label_col='labels', text_col='text', random_state=42):
    """Balances dataset using mixed sampling strategies"""
    np.random.seed(random_state)
    label_counts = df[label_col].value_counts()
    target_size = int(np.mean(label_counts))
    balanced_dfs = []
    
    for label in label_counts.index:
        df_class = df[df[label_col] == label]
        if len(df_class) > target_size:
            # Undersample
            df_class = resample(df_class, replace=False, 
                              n_samples=target_size, 
                              random_state=random_state)
        elif len(df_class) < target_size:
            # Oversample with noise
            n_samples = target_size - len(df_class)
            samples_to_duplicate = min(n_samples, len(df_class))
            df_duplicated = resample(df_class, 
                                   replace=True,
                                   n_samples=samples_to_duplicate, 
                                   random_state=random_state)
            df_class = pd.concat([df_class, df_duplicated])
        balanced_dfs.append(df_class)
    
    return pd.concat(balanced_dfs).sample(frac=1, random_state=random_state)

def tokenize_datasets(df:pd.DataFrame, model_name:str, max_length = 128, column='text'):
    dataset = Dataset.from_pandas(df)

    if 'index_level_0' in dataset.column_names:
        dataset = dataset.remove_columns('index_level_0')
    if 'Unnamed: 0' in dataset.column_names:
        dataset = dataset.remove_columns('Unnamed: 0')

        
    # Tokenize both datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize training data
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples[column], 
                                 truncation=True, 
                                 padding="max_length",
                                 max_length=max_length),
        batched=True,
        remove_columns=[column]
    )

    tokenized_dataset.set_format("torch", 
                             columns=["input_ids", 
                                     "attention_mask", 
                                     "labels"])
    
    if '__index_level_0__' in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.remove_columns('__index_level_0__')

    return tokenized_dataset

def prepare_datasets(df,model:str, test_size=0.2, random_state=None, max_length=128, column='text') -> DatasetDict:
    """
    Splits the dataset into training and testing sets and tokenizes the text data.
    Args:
        df (pd.DataFrame): The input DataFrame containing the text data and labels.
        model (str): The name of the model to use for tokenization.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int, optional): Random seed for reproducibility.
        max_length (int): Maximum length of the tokenized sequences.
        column (str): The name of the column containing the text data.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    tokenized_train = tokenize_datasets(df = train,max_length=max_length,model_name=model,column=column)
    tokenized_test = tokenize_datasets(df = test,max_length=max_length,model_name=model,column=column)

    dataset_dict = DatasetDict({
            'train': tokenized_train,
            'test': tokenized_test
        })
    
    return dataset_dict

