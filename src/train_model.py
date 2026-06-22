from transformers import LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments, PreTrainedTokenizerFast, DataCollatorForLanguageModeling
from datasets import load_from_disk
from pathlib import Path
import torch

# Paths
DATASET_PATH = Path("data/dataset")
TOKENIZER_PATH = Path("data/tokenizer.json")
OUTPUT_DIR = Path("models/novel-llm-llama")

def train():
    # Load dataset
    dataset = load_from_disk(str(DATASET_PATH))
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"

    # Modern Llama Config (GQA + RoPE)
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2, # GQA
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    model = LlamaForCausalLM(config)
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training args (Optimized for VRAM)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        save_steps=100,
        logging_steps=10,
        learning_rate=5e-4,
        fp16=True,
        gradient_checkpointing=True, # Critical for 4GB VRAM
        optim="adamw_torch",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator,
    )
    
    trainer.train()
    
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Training finished.")

if __name__ == "__main__":
    train()
