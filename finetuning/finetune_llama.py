import os
import argparse
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModel
import evaluate
from rouge_score import rouge_scorer
import wandb
from tqdm import tqdm

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
        r=8, # Rank of LoRA adapters
        alpha=16,
        dropout=0.1
    )
    
    model = get_peft_model(model, lora_config)
    model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

# Training loop with SFTTrainer
def train_model(model, tokenizer, dataset, output_path, device):
    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        args=transformers.TrainingArguments(
            output_dir=output_path,
            evaluation_strategy="epoch",
            save_steps=500,
            logging_dir=os.path.join(output_path, "logs"),
            report_to="wandb",  # Log to Weights & Biases
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Initialize Weights & Biases
    wandb.init(project="mimic-iv-finetuning", entity="your-wandb-entity")
    
    trainer.train()

# Evaluation loop
def evaluate_model(model, tokenizer, dataset, evaluation_model_path, device):
    model = AutoModelForCausalLM.from_pretrained(evaluation_model_path)
    model = model.to(device)
    model.eval()
    
    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")
    sari = evaluate.load("sari")
    
    generated_summaries = []
    references = []

    for example in tqdm(dataset["test"]):
        input_text = example["note"]
        reference_summary = example["summary"]
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate summary
        with torch.no_grad():
            generated_ids = model.generate(inputs["input_ids"], max_length=512, num_beams=4)
            generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        generated_summaries.append(generated_summary)
        references.append(reference_summary)

    # Compute metrics
    rouge_scores = rouge.compute(predictions=generated_summaries, references=references)
    bert_score_scores = bert_score.compute(predictions=generated_summaries, references=references)
    sari_scores = sari.compute(predictions=generated_summaries, references=references)

    # Print the metrics
    print("Rouge Scores:", rouge_scores)
    print("BERTScore:", bert_score_scores)
    print("SARI Score:", sari_scores)

    # Log the results to Weights & Biases
    wandb.log({"rouge": rouge_scores, "bertscore": bert_score_scores, "sari": sari_scores})

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
        evaluate_model(model, tokenizer, dataset, args.evaluation_model_path, args.device)
    else:
        # Otherwise, train the model
        train_model(model, tokenizer, dataset, args.output_path, args.device)

if __name__ == "__main__":
    main()
