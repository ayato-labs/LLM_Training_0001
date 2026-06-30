import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path
from transformers import LlamaForCausalLM, AutoTokenizer

# プロジェクトルートのパス解決
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# テスト項目 (評価プロンプト) の定義
TEST_CASES = [
    {
        "id": "TC-01",
        "name": "キャラクター知識と言葉遣いの検証 (ジグ)",
        "prompt": (
            "作品名: 魔女と傭兵\n"
            "ジャンル: ハイファンタジー〔ファンタジー〕\n"
            "会話率(全体): 31.76%\n"
            "会話率(章): 20.00%\n"
            "感情: neutral\n"
            "文字数: 500\n"
            "タグ: R15,残酷な描写あり,男主人公,ダンジョン,傭兵,魔女,現地人主人公,双刃剣,主人公強い\n\n"
            "ジグは双刃剣を構え、"
        ),
        "target": "ジグの戦闘描写、武器『双刃剣』の言及、傭兵らしい語彙の発現"
    },
    {
        "id": "TC-02",
        "name": "文体ステアリングの検証 (高会話率指定)",
        "prompt": (
            "作品名: 魔女と傭兵\n"
            "ジャンル: ハイファンタジー〔ファンタジー〕\n"
            "会話率(全体): 31.76%\n"
            "会話率(章): 80.00%\n"
            "感情: neutral\n"
            "文字数: 300\n"
            "タグ: R15,残酷な描写あり,男主人公,ダンジョン,傭兵,魔女,現地人主人公,双刃剣,主人公強い\n\n"
            "シアーシャは「"
        ),
        "target": "会話率80%指定に対し、セリフ括弧『「』『」』が高頻度で出現するか"
    },
    {
        "id": "TC-03",
        "name": "感情トーン制御の検証 (ポジティブ)",
        "prompt": (
            "作品名: 異世界のんびり農家\n"
            "ジャンル: ハイファンタジー〔ファンタジー〕\n"
            "会話率(全体): 25.00%\n"
            "会話率(章): 15.00%\n"
            "感情: positive\n"
            "文字数: 400\n"
            "タグ: 日常,農業,のんびり\n\n"
            "村のみんなが集まり、嬉しそうに"
        ),
        "target": "感情positive指定に対し、明るい・楽しい・美味しいなどのポジティブな語彙の選択"
    },
    {
        "id": "TC-04",
        "name": "感情トーン制御の検証 (ネガティブ)",
        "prompt": (
            "作品名: Ｒｅ：ゼロから始める異世界生活\n"
            "ジャンル: ハイファンタジー〔ファンタジー〕\n"
            "会話率(全体): 35.00%\n"
            "会話率(章): 10.00%\n"
            "感情: negative\n"
            "文字数: 400\n"
            "タグ: 残酷な描写あり,死に戻り\n\n"
            "スバルは絶望的な状況のなか、"
        ),
        "target": "感情negative指定に対し、恐怖・悲しみ・冷たいなどの不穏な語彙の選択"
    },
    {
        "id": "TC-05",
        "name": "別ドメイン（文芸・歴史）の再現性検証",
        "prompt": (
            "作品名: 淡海乃海　水面が揺れる時\n"
            "ジャンル: 推理〔文芸〕\n"
            "会話率(全体): 30.00%\n"
            "会話率(章): 15.00%\n"
            "感情: neutral\n"
            "文字数: 500\n"
            "タグ: 戦国,歴史\n\n"
            "我が朽木家は、"
        ),
        "target": "戦国歴史物としての語彙（家、領地、武士など）および語り口調の再現"
    }
]

def clean_sentencepiece_spaces(text):
    """SentencePieceのデコード時に生じる不自然な文字間スペースをトリミングする"""
    # 連続するスペースや日本語文字の間の半角スペースを結合
    # ただし英単語同士のスペースは維持する
    import re
    # 日本語文字に挟まれた半角スペースを除去
    text = re.sub(r'([ぁ-んァ-ヶー一-龠々])\s+([ぁ-んァ-ヶー一-龠々])', r'\1\2', text)
    # 記号と日本語の間のスペースを除去
    text = re.sub(r'([ぁ-んァ-ヶー一-龠々])\s+([「」『』、。！？])', r'\1\2', text)
    text = re.sub(r'([「」『』、。！？])\s+([ぁ-んァ-ヶー一-龠々])', r'\1\2', text)
    # コロンやカンマの間のスペース除去
    text = re.sub(r'\s*([:\n])\s*', r'\1', text)
    return text

def run_evaluation(model_path="models/output", max_new_tokens=200):
    print(f"Initializing Evaluator using model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}", file=sys.stderr)
        sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = BASE_DIR / "logs"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"eval_report_{timestamp}.md"
    
    report_content = []
    report_content.append(f"# Model Inference Evaluation Report")
    report_content.append(f"* **Execution Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"* **Model Path**: `{model_path}`")
    report_content.append(f"* **Device**: `{model.device}`")
    report_content.append(f"\n---\n")
    report_content.append(f"## 1. Evaluation Results Summary")
    report_content.append(f"Here are the outputs for the structured test cases designed to measure style-steering and domain knowledge preservation.\n")
    
    for case in TEST_CASES:
        print(f"Running Test: {case['id']} - {case['name']}...")
        prompt = case["prompt"]
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_output = clean_sentencepiece_spaces(raw_output)
        
        report_content.append(f"### [{case['id']}] {case['name']}")
        report_content.append(f"* **Target Ability**: {case['target']}")
        report_content.append(f"#### Prompt Prefix:")
        report_content.append(f"```text\n{prompt}\n```")
        report_content.append(f"#### Model Generated Output:")
        report_content.append(f"```text\n{cleaned_output}\n```")
        report_content.append(f"\n")
        
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
        
    print(f"Evaluation report written successfully to: {report_file}")
    return report_file

if __name__ == "__main__":
    run_evaluation()
