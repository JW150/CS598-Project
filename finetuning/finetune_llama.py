import os
import torch
import wandb
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import random  # For simulating human feedback
from rouge_score import rouge_scorer
import bert_score

# Set verbosity for transformers logs
transformers.logging.set_verbosity_info()

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning LLM on MIMIC-IV summarization.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to MIMIC-IV dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save model and logs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for training (cuda or cpu).")
    parser.add_argument("--evaluation", action="store_true", help="Evaluate the model instead of training.")
    parser.add_argument("--evaluation_model_path", type=str, default=None, help="Path to the evaluation model.")
    return parser.parse_args()

# Load the MIMIC-IV dataset
def load_mimic_dataset(data_path):
    dataset = load_dataset('json', data_files=data_path)
    return dataset

# Prepare the model for LoRA fine-tuning
def prepare_model(model_name_or_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = model.to(device)
    
    lora_config = LoraConfig(
        r=8,  # Rank of LoRA adapters
        alpha=16,
        dropout=0.1
    )
    
    model = get_peft_model(model, lora_config)
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

# Simulate human feedback loop for reinforcement learning
def human_feedback_loop(generated_summary, reference_summary):
    # Simulate a human rating (1-5 scale for simplicity)
    # In real implementation, this would involve real human feedback
    feedback_score = random.randint(1, 5)  # Random feedback as a placeholder for now
    print(f"Human feedback: {feedback_score} for generated summary.")
    return feedback_score

# Training loop using SFTTrainer with RL-HITL
def train_model(model, tokenizer, dataset, output_path, device, epochs=3):
    # Initialize DataLoader for training and evaluation
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=data_collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=output_path,
            evaluation_strategy="epoch",
            save_steps=500,
            logging_dir=os.path.join(output_path, "logs"),
            report_to="wandb",  # Log to Weights & Biases
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_steps=100,
            save_total_limit=2,
            fp16=True,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Initialize Weights & Biases
    wandb.init(project="mimic-iv-finetuning", entity="your-wandb-entity")

    # Training loop with RL-HITL
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # Forward pass
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            labels = inputs['labels']
            
            # Compute loss (Cross-Entropy Loss)
            loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Update parameters
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.model.zero_grad()

            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Training Loss: {avg_loss}")
        
        # Log the average loss to Weights & Biases
        wandb.log({"epoch": epoch+1, "avg_train_loss": avg_loss})

        # Simulate RL-HITL Feedback loop
        if epoch % 2 == 0:  # Simulating feedback every two epochs
            print("Simulating human feedback...")
            for i, batch in enumerate(eval_dataloader):
                generated_summary = model.generate(batch["input_ids"])
                reference_summary = batch["labels"]
                feedback_score = human_feedback_loop(generated_summary, reference_summary)
                # Use feedback_score to adjust loss or reward (this can be added to RL optimization)
                
                # Log the feedback score to see it during training
                wandb.log({"feedback_score": feedback_score})

        # Evaluate after each epoch
        evaluate_model(model, tokenizer, eval_dataloader, device)

    # Save final model checkpoint
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

# Improved Evaluation loop with broader metrics
def evaluate_model(model, tokenizer, eval_dataloader, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in eval_dataloader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']
        
        # Compute loss (Cross-Entropy Loss)
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()

        # Collect predictions and labels for metrics
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.append(predictions)
        all_labels.append(labels)

    avg_eval_loss = total_loss / len(eval_dataloader)
    print(f"Evaluation Loss: {avg_eval_loss}")
    
    # Log the evaluation loss to Weights & Biases
    wandb.log({"eval_loss": avg_eval_loss})

    # Add broader evaluation metrics (ROUGE, BERTScore, etc.)
    rouge_score = calculate_rouge(all_predictions, all_labels)
    bert_score = calculate_bertscore(all_predictions, all_labels)
    print(f"ROUGE Score: {rouge_score}, BERTScore: {bert_score}")

    # Log additional metrics to Weights & Biases
    wandb.log({"rouge_score": rouge_score, "bertscore": bert_score})

# Function to calculate ROUGE score using rouge_score library
def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    num_samples = len(predictions)

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for metric in total_rouge:
            total_rouge[metric] += scores[metric].fmeasure

    avg_rouge = {metric: score / num_samples for metric, score in total_rouge.items()}
    return avg_rouge

# Function to calculate BERTScore using bert_score library
def calculate_bertscore(predictions, references):
    P, R, F1 = bert_score.score(predictions, references, lang='en')
    return F1.mean().item()  # Return the average F1 score

# Main function to run the script
def main():
    args = parse_args()
    
    # Load the dataset
    dataset = load_mimic_dataset(args.data_path)
    
    # Prepare the model
    model, tokenizer = prepare_model(args.model_name_or_path, args.device)

    # If evaluation flag is passed, evaluate the model
    if args.evaluation:
        if args.evaluation_model_path is None:
            raise ValueError("Please specify the model path for evaluation.")
        evaluate_model(model, tokenizer, dataset["validation"], args.device)
    else:
        # Otherwise, train the model
        train_model(model, tokenizer, dataset, args.output_path, args.device, epochs=3)

if __name__ == "__main__":
    main()
