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
from sklearn.model_selection import StratifiedKFold
from functions import tokenize_datasets, aggregate_metrics

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

def train_and_evaluate(strategy_name,tokenized_dataset, base_model_name, train_test_seed, model_seed):

    # Reload model for each run to reset weights
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/results_{strategy_name}_{base_model_name}_{train_test_seed}",
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
        callbacks=[MasterCSVLoggerCallback(run_id=f'{strategy_name}_{base_model_name}_{train_test_seed}')]
    )


    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results, trainer



class WeightedTrainer(Trainer):
    def __init__(self, class_weights, label_smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if labels is not None:
            # Use label smoothing + class weights
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(model.device),
                label_smoothing=self.label_smoothing
            )
            loss = loss_fct(
                logits.view(-1, model.config.num_labels), 
                labels.view(-1)
            )
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss
    
# Calculating class weights
def get_class_weights(labels, floor=1.0):
    """Robust weight calculation with missing class handling"""
    valid_classes = [0, 1, 2]
    label_counts = {cls: max(np.sum(labels == cls), 1) for cls in valid_classes}
    total_samples = sum(label_counts.values())
    
    weights = {
        cls: max(total_samples / (len(valid_classes) * count), floor)
        for cls, count in label_counts.items()
    }
    return torch.tensor([weights[cls] for cls in valid_classes], dtype=torch.float32)

def train_evaluate_grid(params, tokenized_dataset, model_name):
    # Get and validate labels
    train_labels = np.array(tokenized_dataset["train"]["labels"])
    
    # Class weight calculation with floor
    class_weights = get_class_weights(train_labels, floor=params.get("class_weight_floor", 1.0))

    # Model initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
   
    # Training arguments with imbalance optimizations
    args = TrainingArguments(
        output_dir=f"./grid_results/{hash(str(params))}",
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_ratio=0.1,  
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=params['model_seed'],
        optim="adamw_torch",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=params.get("grad_accum_steps", 2),
        report_to="none",
        logging_steps=50,
        lr_scheduler_type="cosine",
         
    )

    # Initialize trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[MasterCSVLoggerCallback(run_id=f'{model_name}_{"_".join(map(str, params.values()))}')], 
    )

    # Train with validation checks
    try:
        trainer.train()
        eval_results = trainer.evaluate()
        return {
            **params,
            "accuracy": eval_results["eval_accuracy"],
            "macro_f1": eval_results["eval_f1_macro"],
            "weighted_f1": eval_results["eval_f1_weighted"],
            "neutral_f1": eval_results["eval_f1_0"],
            "positive_f1": eval_results["eval_f1_1"],
            "negative_f1": eval_results["eval_f1_2"],
        }
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None
    
def run_evaluation(tokenized_dataset: dict, params: dict,model:str, strategy_name:str) -> None:
    results = []
    try:
   
        print(f"\n{'='*40}")
        print(f"Testing preprocessing variant: {strategy_name}")
        print(f"{'='*40}")
        print(f'Training model: {model}')
        eval_results,trainer = train_and_evaluate(strategy_name, tokenized_dataset,model,train_test_seed = params['train_seed'], model_seed = params['model_seed'])
        get_misclassified(trainer,tokenized_dataset=tokenized_dataset,model_name=model,train_test_seed=params['train_seed'],preprocessing=strategy_name)
        print(results)

    except Exception as e:
        print(f"Error with variant {strategy_name}: {str(e)}")

    return {
            **params,
            "accuracy": eval_results["eval_accuracy"],
            "macro_f1": eval_results["eval_f1_macro"],
            "weighted_f1": eval_results["eval_f1_weighted"],
            "neutral_f1": eval_results["eval_f1_0"],
            "positive_f1": eval_results["eval_f1_1"],
            "negative_f1": eval_results["eval_f1_2"],
        }

def cross_val_score(df, params = None, model_name = 'allegro/herbert-base-cased', strategy_name = None):
    required = ['train_seed', 'model_seed']
    if any(param not in params for param in required):
        raise ValueError("Missing required config parameters.")
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=params['train_seed'])
    all_fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df['text'], df['labels'])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Tokenize if needed
        train_df_tokenized = tokenize_datasets(df = train_df, model_name=model_name, max_length=128, column='text')
        val_df_tokenized = tokenize_datasets(df = val_df, model_name=model_name, max_length=128, column='text')

        dataset_dict = DatasetDict({
            'train': train_df_tokenized,
            'test': val_df_tokenized
        })
        # Train model and get metrics (e.g., accuracy, F1, etc.)
        if strategy_name:
            metrics = run_evaluation(params = params, tokenized_dataset = dataset_dict, model = model_name, strategy_name= strategy_name)
        else:
            metrics = train_evaluate_grid(params = params, tokenized_dataset= dataset_dict, model_name = model_name)
            
        all_fold_metrics.append(metrics)

    # Return average metrics across folds
    return aggregate_metrics(all_fold_metrics)


def train_evaluate(params, tokenized_dataset, model_name):
    # Get and validate labels
    train_labels = np.array(tokenized_dataset["train"]["labels"])
    
    # Class weight calculation with floor
    class_weights = get_class_weights(train_labels, floor=params.get("class_weight_floor", 1.0))

    # Model initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=3
    )
   
    # Training arguments with imbalance optimizations
    args = TrainingArguments(
        output_dir=f"./trained/{hash(str(params))}",
        num_train_epochs=params["epochs"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=32,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_ratio=0.1,  
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        seed=params['model_seed'],
        optim="adamw_torch",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=params.get("grad_accum_steps", 2),
        report_to="none",
        logging_steps=50,
        lr_scheduler_type="cosine",
         
    )

    # Initialize trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[MasterCSVLoggerCallback(run_id=f'{model_name}_{"_".join(map(str, params.values()))}')], 
    )

    # Train with validation checks
    try:
        trainer.train()
        eval_results = trainer.evaluate()
        return trainer, eval_results
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None
