import os
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

def main():
    # 1. Initialize Weights & Biases
    wandb.init(project=os.environ.get("WANDB_PROJECT", "scai-jam-demo"))

    # 2. Load dataset and model names from environment so the template is easy to customize
    dataset_name = os.environ.get("HF_DATASET", "databricks/databricks-dolly-15k")
    model_name = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-0.5B")
    hf_token = os.environ.get("HF_TOKEN")

    # 3. Load Dataset (Hugging Face Datasets)
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, use_auth_token=hf_token)

    # 4. Load Tokenizer & Model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        torch_dtype=dtype,
    )
    model = model.to(device)

    # 5. Apply LoRA Config
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. Preprocess Data
    def format_and_tokenize(examples):
        prompt_parts = []
        for i in range(len(examples["instruction"])):
            instruction = examples.get("instruction", [""] * len(examples["instruction"]))[i]
            context = examples.get("context", [""] * len(examples["instruction"]))[i]
            response = examples.get("response", [""] * len(examples["instruction"]))[i]
            prompt_parts.append(
                f"Instruction: {instruction}\n\nContext: {context}\n\nResponse: {response}"
            )

        tokenized = tokenizer(
            prompt_parts,
            padding="max_length",
            truncation=True,
            max_length=256,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_datasets = dataset.map(
        format_and_tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
    train_test_split = train_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # 7. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="/persistent/results",
        evaluation_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=20,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        weight_decay=0.01,
        save_total_limit=1,
        report_to="wandb",
        push_to_hub=False,
        hub_model_id="your-hf-username/scai-jam-lora-model",
    )

    # 8. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # 9. Train!
    print("Starting training...")
    trainer.train()

    trainer.save_model("/persistent/results/final_model")
    wandb.finish()

if __name__ == "__main__":
    main()
