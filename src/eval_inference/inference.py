from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import torch
from pathlib import Path

# Paths
MODEL_PATH = Path("models/novel-llm-llama")
TOKENIZER_PATH = Path("data/tokenizer.json")

def load_model():
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_PATH))
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    
    # ADR-0013: 特殊トークンの設定
    special_tokens = {
        "additional_special_tokens": [
            "<|start_of_metadata|>",
            "<|end_of_metadata|>",
            "<|start_of_story|>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Load model
    model = LlamaForCausalLM.from_pretrained(str(MODEL_PATH))
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model, tokenizer

def generate_with_rag(prompt, context_data):
    """
    RAG-style context injection for novel writing.
    context_data: dict containing 'plot', 'character_description'
    """
    model, tokenizer = load_model()
    
    # Construct context-aware prompt
    rag_prompt = f"Context: Plot: {context_data['plot']}. Characters: {context_data['character_description']}.\n\nWrite next scene: {prompt}"
    
    inputs = tokenizer(rag_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Placeholder inference test
    context = {
        "plot": "Hero finds a mysterious artifact.",
        "character_description": "Hero is brave, slightly reckless."
    }
    print(generate_with_rag("The hero reached out...", context))
