"""
Cross-scale benchmark suite for Novel LLM.
Evaluates style control, character consistency, and domain adaptation
across different model sizes (150M, 3B, 7B).

Usage:
    python -m src.evaluation.benchmark --model-path models/output
    python -m src.evaluation.benchmark --model-path models/output --output logs/benchmark.json
"""
import argparse
import json
import re
import sys
import torch
import mlflow
from datetime import datetime
from pathlib import Path
from transformers import LlamaForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))


# ============================================================
# Benchmark Test Cases (expanded from TC-01~05)
# ============================================================
BENCHMARK_CASES = [
    # --- Style Control ---
    {
        "id": "STYLE-01",
        "category": "style_control",
        "name": "高会話率制御",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: テスト作品\n"
            "ジャンル: ファンタジー\n"
            "会話率(全体): 50.00%\n"
            "会話率(章): 90.00%\n"
            "感情: neutral\n"
            "文字数: 500\n"
            "タグ: テスト\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "冒険者たちが集まり、"
        ),
        "target": "会話括弧「」が90%以上を占める",
        "metric": "dialogue_ratio",
        "threshold": 0.7,
    },
    {
        "id": "STYLE-02",
        "category": "style_control",
        "name": "低会話率制御",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: テスト作品\n"
            "ジャンル: ファンタジー\n"
            "会話率(全体): 50.00%\n"
            "会話率(章): 5.00%\n"
            "感情: neutral\n"
            "文字数: 500\n"
            "タグ: テスト\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "森の中を進むと、"
        ),
        "target": "会話括弧「」が5%以下",
        "metric": "dialogue_ratio",
        "threshold": 0.15,
    },
    {
        "id": "STYLE-03",
        "category": "style_control",
        "name": "感情ポジティブ制御",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: テスト作品\n"
            "ジャンル: 日常\n"
            "会話率(全体): 30.00%\n"
            "会話率(章): 20.00%\n"
            "感情: positive\n"
            "文字数: 500\n"
            "タグ: テスト\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "朝の光が差し込み、"
        ),
        "target": "明るい・楽しい・美味しい等のポジティブ語彙",
        "metric": "sentiment_positive",
        "threshold": 0.3,
    },
    {
        "id": "STYLE-04",
        "category": "style_control",
        "name": "感情ネガティブ制御",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: テスト作品\n"
            "ジャンル: ダークファンタジー\n"
            "会話率(全体): 30.00%\n"
            "会話率(章): 20.00%\n"
            "感情: negative\n"
            "文字数: 500\n"
            "タグ: テスト\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "闇に包まれた街で、"
        ),
        "target": "暗い・恐ろしい・絶望等のネガティブ語彙",
        "metric": "sentiment_negative",
        "threshold": 0.3,
    },
    # --- Character Consistency ---
    {
        "id": "CHAR-01",
        "category": "character_consistency",
        "name": "キャラ名前一貫性",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: 魔女と傭兵\n"
            "ジャンル: ハイファンタジー\n"
            "会話率(全体): 31.76%\n"
            "会話率(章): 20.00%\n"
            "感情: neutral\n"
            "文字数: 500\n"
            "タグ: ジグ,シアーシャ\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "ジグは剣を構え、シアーシャに"
        ),
        "target": "「ジグ」「シアーシャ」が正確に出現",
        "metric": "character_name_accuracy",
        "threshold": 0.8,
    },
    # --- Domain Adaptation ---
    {
        "id": "DOMAIN-01",
        "category": "domain_adaptation",
        "name": "歴史物口調",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: 淡海乃海\n"
            "ジャンル: 歴史\n"
            "会話率(全体): 30.00%\n"
            "会話率(章): 15.00%\n"
            "感情: neutral\n"
            "文字数: 500\n"
            "タグ: 戦国,歴史\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "我が朽木家は、"
        ),
        "target": "武家言葉・歴史的語彙の使用",
        "metric": "historical_vocab",
        "threshold": 0.3,
    },
    # --- Japanese Quality ---
    {
        "id": "QUALITY-01",
        "category": "language_quality",
        "name": "文法整合性",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: テスト\n"
            "ジャンル: 日常\n"
            "会話率(全体): 25.00%\n"
            "会話率(章): 10.00%\n"
            "感情: neutral\n"
            "文字数: 300\n"
            "タグ: テスト\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "桜の花が舞い散る中、少女は"
        ),
        "target": "自然な日本語文法（助詞・活用の整合性）",
        "metric": "grammar_score",
        "threshold": 0.5,
    },
]


def _compute_dialogue_ratio(text: str) -> float:
    """「」内の文字数 / 総文字数"""
    total = len(text)
    if total == 0:
        return 0.0
    matches = re.findall(r'「(.*?)」', text, re.DOTALL)
    dialogue_chars = sum(len(m) for m in matches)
    return dialogue_chars / total


def _compute_sentiment(text: str, positive_words: set, negative_words: set) -> dict:
    pos = sum(text.count(w) for w in positive_words)
    neg = sum(text.count(w) for w in negative_words)
    total = pos + neg
    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "label": "neutral"}
    return {
        "positive": pos / total,
        "negative": neg / total,
        "label": "positive" if pos > neg else "negative",
    }


def _compute_character_accuracy(text: str, names: list[str]) -> float:
    found = sum(1 for name in names if name in text)
    return found / len(names) if names else 0.0


def _compute_historical_vocab(text: str) -> float:
    """歴史的語彙の出現率"""
    historical = {'家', '領', '武士', '殿', '姫', '城', '刀', '騎士', '将軍', '兵',
                  '戦', '合戦', '忠義', '主君', '従者', '弓', '槍', '鎧', '陣'}
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    count = sum(text.count(w) for w in historical)
    return min(count / 10, 1.0)  # normalize


def _compute_grammar_score(text: str) -> float:
    """Simple grammar heuristic based on Japanese patterns."""
    score = 1.0
    # Penalize consecutive punctuation
    if re.search(r'[。、]{3,}', text):
        score -= 0.2
    # Penalize broken quotes
    open_q = text.count('「')
    close_q = text.count('」')
    if abs(open_q - close_q) > 1:
        score -= 0.3
    # Penalize English letters mixed randomly
    eng_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
    if eng_ratio > 0.1:
        score -= 0.2
    return max(score, 0.0)


def run_benchmark(model_path: str = "models/output", max_new_tokens: int = 200,
                  output_path: str = None) -> list[dict]:
    """
    Run the full benchmark suite.

    Returns:
        List of benchmark results with metrics.
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    positive_words = {'素晴らしい', '楽しい', '嬉しい', '美しい', '希望', '愛', '優しい', '美味しい', '明るい'}
    negative_words = {'悲しい', '辛い', '憎い', '失敗', '怖い', '醜い', '絶望', '冷たい', '死', '闇'}

    results = []

    for case in BENCHMARK_CASES:
        print(f"  [{case['id']}] {case['name']}...")

        inputs = tokenizer(case["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from output
        generated = raw[len(case["prompt"]):] if raw.startswith(case["prompt"]) else raw

        # Compute metric
        metric_name = case["metric"]
        if metric_name == "dialogue_ratio":
            value = _compute_dialogue_ratio(generated)
        elif metric_name == "sentiment_positive":
            sent = _compute_sentiment(generated, positive_words, negative_words)
            value = sent["positive"]
        elif metric_name == "sentiment_negative":
            sent = _compute_sentiment(generated, positive_words, negative_words)
            value = sent["negative"]
        elif metric_name == "character_name_accuracy":
            names = re.findall(r'[\u4e00-\u9fff]+', case["prompt"].split("タグ:")[-1]) if "タグ:" in case["prompt"] else []
            names = [n.strip() for n in names if len(n) >= 2]
            if not names:
                names = ["ジグ", "シアーシャ"]  # fallback
            value = _compute_character_accuracy(generated, names)
        elif metric_name == "historical_vocab":
            value = _compute_historical_vocab(generated)
        elif metric_name == "grammar_score":
            value = _compute_grammar_score(generated)
        else:
            value = 0.0

        passed = value >= case["threshold"]

        results.append({
            "test_case_id": case["id"],
            "category": case["category"],
            "name": case["name"],
            "metric": metric_name,
            "value": round(value, 4),
            "threshold": case["threshold"],
            "passed": passed,
            "generated_output": generated,
            "target": case["target"],
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
        })

        status = "PASS" if passed else "FAIL"
        print(f"    {status}: {metric_name}={value:.4f} (threshold={case['threshold']})")

    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print(f"\nBenchmark Summary: {passed_count}/{total_count} passed")

    # Save results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

    # Log to MLflow
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        experiment = mlflow.get_experiment_by_name("LLM_Training")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=1,
                order_by=["start_time DESC"],
            )
            if not runs.empty:
                run_id = runs.iloc[0]["run_id"]
                with mlflow.start_run(run_id=run_id):
                    for r in results:
                        mlflow.log_metric(
                            f"benchmark_{r['test_case_id']}",
                            r["value"],
                        )
                    mlflow.log_metric("benchmark_pass_rate", passed_count / total_count)
                    mlflow.log_metric("benchmark_total_passed", passed_count)
                    mlflow.log_metric("benchmark_total_cases", total_count)
                    if output_path:
                        mlflow.log_artifact(output_path, artifact_path="benchmarks")
                    print("[MLflow] Benchmark results logged.")
    except Exception as e:
        print(f"[MLflow] Benchmark logging failed: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Novel LLM Benchmark Suite")
    parser.add_argument("--model-path", default="models/output", help="Path to trained model")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or f"logs/benchmark_{timestamp}.json"

    run_benchmark(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        output_path=output,
    )


if __name__ == "__main__":
    main()
