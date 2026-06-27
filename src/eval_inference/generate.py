import argparse
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from pathlib import Path

# モデルパスを修正後の場所に合わせる
def generate_text(prompt, model_path="models/output", max_new_tokens=100):
    print(f"Loading model and tokenizer from {model_path}...")
    # モデルディレクトリから直接トークナイザーをロード
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Inference Tool")
    parser.add_argument("--prompt", type=str, help="The input prompt for the model")
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Enter prompt: ")
        
    print(f"Prompt: {prompt}")
    # パスが正しいか確認して実行
    try:
        result = generate_text(prompt)
        print(f"Generated:\n{result}")
    except Exception as e:
        print(f"Error during generation: {e}")
