from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import copy
from transformers import EvalPrediction, Trainer, TrainerCallback, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import os
import scipy
import torch
from datasets import Dataset, DatasetDict

class MasterCSVLoggerCallback(TrainerCallback):
    def __init__(self, master_file_path="all_runs_metrics.csv", run_id=None):
        self.master_file_path = master_file_path
        self.run_id = run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        import os
        if not os.path.exists(self.master_file_path) or os.path.getsize(self.master_file_path) == 0:
            with open(self.master_file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["run_id", "epoch", "global_step", "eval_loss", "eval_accuracy", "eval_f1_macro"])

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        with open(self.master_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.run_id,
                state.epoch,
                state.global_step,
                metrics.get("eval_loss"),
                metrics.get("eval_accuracy"),
                metrics.get("eval_f1_macro"),
            ])

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "f1_0": f1_score(labels, predictions, average=None)[0],
        "f1_1": f1_score(labels, predictions, average=None)[1],
        "f1_2": f1_score(labels, predictions, average=None)[2],
    }

def get_misclassified(trainer, tokenized_dataset, model_name, train_test_seed, preprocessing):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get predictions
    predictions = trainer.predict(tokenized_dataset["test"])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Get probability scores using softmax
    probabilities = scipy.special.softmax(predictions.predictions, axis=1)
    
    # Find misclassified samples
    misclassified_indices = np.where(predicted_labels != true_labels)[0]
    
    # Create dictionary with detailed information
    misclassified_samples = {
        "Index": misclassified_indices,
        "True_Label": true_labels[misclassified_indices],
        "Predicted_Label": predicted_labels[misclassified_indices],
        "Text": [tokenizer.decode(tokenized_dataset["test"]["input_ids"][i], 
                                skip_special_tokens=True) 
                for i in misclassified_indices],
        "Confidence": [probabilities[i][predicted_labels[i]] 
                      for i in misclassified_indices],
        "Prob_Class_0": probabilities[misclassified_indices, 0],
        "Prob_Class_1": probabilities[misclassified_indices, 1],
        "Prob_Class_2": probabilities[misclassified_indices, 2]
    }
    
    # Create DataFrame and add text length
    df_misclassified = pd.DataFrame(misclassified_samples)
    df_misclassified['Text_Length'] = df_misclassified['Text'].str.len()
    
    # Add timestamp to filename to avoid overwriting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'./errors/misclassified_{model_name[:4]}_{train_test_seed}_{preprocessing}_{timestamp}.csv'
    
    # Save with proper encoding
    df_misclassified.to_csv(filename, index=False, encoding='utf-8')
    print(f"Saved misclassified examples to {filename}")
    
    return df_misclassified

def apply_transformations(df, transformations, name):
    """Apply a sequence of transformations to the dataframe's text column"""
    
    processed_df = copy.deepcopy(df)
    current_name = name
    for func in transformations:
        current_name += f'_{func.__name__}'
        file_path = f'./CachedProcessing/processed_data{current_name}.csv' 

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            processed_df['text'] = processed_df['text'].apply(func) 
            processed_df.to_csv(file_path, index= False)
            
    return processed_df

def transform_data(processing_pipelines,dataset: pd.DataFrame, csv_dir='TrainingData/')->dict:
    """
    Transforms the dataset using the specified processing pipelines and saves the results to CSV files.
    Args:
        processing_pipelines (dict): A dictionary where keys are pipeline names and values are lists of functions to apply.
        csv_dir (str): Directory where the processed CSV files will be saved or loaded from.
    """
    datasets = {}
    for name, pipeline in processing_pipelines.items():
        csv_filename = f"{csv_dir}processed_data_{name}.csv"  # Assume CSV files are named after the dataset 'name'
        
        if os.path.exists(csv_filename):  # Check if the CSV file exists
            print(csv_filename)
            # If CSV exists, load the dataset from the CSV file
            datasets[name] = pd.read_csv(csv_filename)
        else:
            # If CSV does not exist, apply the transformation pipeline
            print(f"{name} didn't found in directory. Applying function...")
            processed_df = apply_transformations(dataset, pipeline, name=name)
            processed_df.to_csv(csv_filename, index=False)  
            datasets[name] = processed_df
    return datasets

def train_and_evaluate(key, tokenized_dataset, base_model_name, train_test_seed, model_seed):

    # Reload model for each run to reset weights
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/results_{key}_{base_model_name}_{train_test_seed}",
        num_train_epochs=3,
        per_device_train_batch_size=8,  
        per_device_eval_batch_size=32,
        warmup_ratio=0.05,
        weight_decay=0.01,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro", 
        greater_is_better=True,
        seed=model_seed,
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=25,
        gradient_accumulation_steps=2,
        max_grad_norm=15.0,
        save_total_limit=0,  # Keep only last 1 checkpoints
        group_by_length=True,
    )

    # Initialize custom trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[MasterCSVLoggerCallback(run_id=f'{key}_{base_model_name}_{train_test_seed}')]
    )


    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results, trainer

def run_evaluation(tokenized_datasets: dict, model:str, train_test_seed:int, model_seed:int = 12) -> None:
    results = []

    for key, tokenized_dataset in tokenized_datasets.items():
        try:
            print(f"\n{'='*40}")
            print(f"Testing preprocessing variant: {key}")
            print(f"{'='*40}")
            print(f'Training model: {model}')
            metrics, trainer = train_and_evaluate(key, tokenized_dataset,model,train_test_seed, model_seed = model_seed)
            get_misclassified(trainer,tokenized_dataset=tokenized_dataset,model_name=model,train_test_seed=train_test_seed,preprocessing=key)

            results.append({
                "model":model,
                "preprocessing": key,
                "accuracy": metrics["eval_accuracy"],
                "macro_f1": metrics["eval_f1_macro"],
                "weighted_f1": metrics["eval_f1_weighted"],
                "neutral_f1": metrics["eval_f1_0"],
                "positive_f1": metrics["eval_f1_1"],
                "negative_f1": metrics["eval_f1_2"],
                "epochs": metrics["epoch"]
            })

            print(results)

        except Exception as e:
            print(f"Error with variant {key}: {str(e)}")

    return results

