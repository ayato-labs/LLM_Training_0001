"""
Inference evaluation suite for trained Novel LLM.
Generates structured test outputs, logs to MLflow, and saves locally.

ADR-022: Inference output traceability.
"""
import os
import sys
import json
import re
import torch
import mlflow
from datetime import datetime
from pathlib import Path
from transformers import LlamaForCausalLM, AutoTokenizer

# プロジェクトルートのパス解決
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# テスト項目 (評価プロンプト) の定義 (ADR-0013: 特殊トークン付き)
TEST_CASES = [
    {
        "id": "TC-01",
        "name": "キャラクター知識と言葉遣いの検証 (ジグ)",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: 魔女と傭兵\n"
            "ジャンル: ハイファンタジー〔ファンタジー〕\n"
            "会話率(全体): 31.76%\n"
            "会話率(章): 20.00%\n"
            "感情: neutral\n"
            "文字数: 500\n"
            "タグ: R15,残酷な描写あり,男主人公,ダンジョン,傭兵,魔女,現地人主人公,双刃剣,主人公強い\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "ジグは双刃剣を構え、"
        ),
        "target": "ジグの戦闘描写、武器『双刃剣』の言及、傭兵らしい語彙の発現"
    },
    {
        "id": "TC-02",
        "name": "文体ステアリングの検証 (高会話率指定)",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: 魔女と傭兵\n"
            "ジャンル: ハイファンタジー〔ファンタジー〕\n"
            "会話率(全体): 31.76%\n"
            "会話率(章): 80.00%\n"
            "感情: neutral\n"
            "文字数: 300\n"
            "タグ: R15,残酷な描写あり,男主人公,ダンジョン,傭兵,魔女,現地人主人公,双刃剣,主人公強い\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "シアーシャは「"
        ),
        "target": "会話率80%指定に対し、セリフ括弧『「』『」』が高頻度で出現するか"
    },
    {
        "id": "TC-03",
        "name": "感情トーン制御の検証 (ポジティブ)",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: 異世界のんびり農家\n"
            "ジャンル: ハイファンタジー〔ファンタジー〕\n"
            "会話率(全体): 25.00%\n"
            "会話率(章): 15.00%\n"
            "感情: positive\n"
            "文字数: 400\n"
            "タグ: 日常,農業,のんびり\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "村のみんなが集まり、嬉しそうに"
        ),
        "target": "感情positive指定に対し、明るい・楽しい・美味しいなどのポジティブな語彙の選択"
    },
    {
        "id": "TC-04",
        "name": "感情トーン制御の検証 (ネガティブ)",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: Ｒｅ：ゼロから始める異世界生活\n"
            "ジャンル: ハイファンタジー〔ファンタジー〕\n"
            "会話率(全体): 35.00%\n"
            "会話率(章): 10.00%\n"
            "感情: negative\n"
            "文字数: 400\n"
            "タグ: 残酷な描写あり,死に戻り\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "スバルは絶望的な状況のなか、"
        ),
        "target": "感情negative指定に対し、恐怖・悲しみ・冷たいなどの不穏な語彙の選択"
    },
    {
        "id": "TC-05",
        "name": "別ドメイン（文芸・歴史）の再現性検証",
        "prompt": (
            "<|start_of_metadata|>\n"
            "作品名: 淡海乃海　水面が揺れる時\n"
            "ジャンル: 推理〔文芸〕\n"
            "会話率(全体): 30.00%\n"
            "会話率(章): 15.00%\n"
            "感情: neutral\n"
            "文字数: 500\n"
            "タグ: 戦国,歴史\n"
            "<|end_of_metadata|>\n"
            "<|start_of_story|>"
            "我が朽木家は、"
        ),
        "target": "戦国歴史物としての語彙（家、領地、武士など）および語り口調の再現"
    }
]


def clean_sentencepiece_spaces(text):
    """SentencePieceのデコード時に生じる不自然な文字間スペースをトリミングする"""
    text = re.sub(r'([ぁ-んァ-ヶー一-龠々])\s+([ぁ-んァ-ヶー一-龠々])', r'\1\2', text)
    text = re.sub(r'([ぁ-んァ-ヶー一-龠々])\s+([「」『』、。！？])', r'\1\2', text)
    text = re.sub(r'([「」『』、。！？])\s+([ぁ-んァ-ヶー一-龠々])', r'\1\2', text)
    text = re.sub(r'\s*([:\n])\s*', r'\1', text)
    return text


def _log_to_mlflow(results: list[dict], report_md: str, model_path: str):
    """Log inference results to MLflow as artifacts and metrics."""
    try:
        mlflow.set_tracking_uri("file:./mlruns")

        # Check if there's an active run; if not, log under the latest run
        active_run = mlflow.active_run()
        if active_run is None:
            # Try to find and log to the most recent run
            experiment = mlflow.get_experiment_by_name("LLM_Training")
            if experiment is None:
                print("[MLflow] No active experiment found. Skipping MLflow logging.")
                return

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=1,
                order_by=["start_time DESC"],
            )
            if runs.empty:
                print("[MLflow] No runs found. Skipping MLflow logging.")
                return
            run_id = runs.iloc[0]["run_id"]
            with mlflow.start_run(run_id=run_id):
                _do_mlflow_logging(results, report_md, model_path)
        else:
            _do_mlflow_logging(results, report_md, model_path)

    except Exception as e:
        print(f"[MLflow] Inference logging failed: {e}")


def _do_mlflow_logging(results: list[dict], report_md: str, model_path: str):
    """Actual MLflow logging logic (must be inside an active run)."""
    # Log each test case output as a separate artifact
    for r in results:
        tc_id = r["test_case_id"]
        output_data = {
            "test_case_id": tc_id,
            "test_case_name": r["test_case_name"],
            "target": r["target"],
            "prompt": r["prompt"],
            "generated_output": r["generated_output"],
            "output_length_chars": len(r["generated_output"]),
        }
        # Save as JSON artifact
        tmp_path = Path(f"logs/_tmp_{tc_id}.json")
        tmp_path.parent.mkdir(exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(str(tmp_path), artifact_path="inference_results")
        tmp_path.unlink(missing_ok=True)

    # Log the full Markdown report
    mlflow.log_artifact(report_md, artifact_path="inference_reports")

    # Log summary metrics
    total_chars = sum(len(r["generated_output"]) for r in results)
    mlflow.log_metrics({
        "inference_total_output_chars": total_chars,
        "inference_test_cases": len(results),
    })

    # Log model path reference
    mlflow.log_param("inference_model_path", model_path)

    print(f"[MLflow] Inference results logged ({len(results)} test cases).")


def run_evaluation(model_path="models/output", max_new_tokens=200, log_mlflow=True):
    """
    Run the full inference evaluation suite.

    Args:
        model_path: Path to the trained model.
        max_new_tokens: Maximum tokens to generate.
        log_mlflow: Whether to log results to MLflow.

    Returns:
        Tuple of (report_path, structured_results)
    """
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
    json_file = report_dir / f"eval_results_{timestamp}.json"

    structured_results = []
    report_content = []
    report_content.append(f"# Model Inference Evaluation Report")
    report_content.append(f"* **Execution Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"* **Model Path**: `{model_path}`")
    report_content.append(f"* **Device**: `{model.device}`")
    report_content.append(f"* **Max New Tokens**: {max_new_tokens}")
    report_content.append(f"\n---\n")
    report_content.append(f"## 1. Evaluation Results Summary\n")

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
                top_p=0.9,
            )

        raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_output = clean_sentencepiece_spaces(raw_output)

        structured_results.append({
            "test_case_id": case["id"],
            "test_case_name": case["name"],
            "target": case["target"],
            "prompt": prompt,
            "raw_output": raw_output,
            "generated_output": cleaned_output,
            "generation_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
        })

        report_content.append(f"### [{case['id']}] {case['name']}")
        report_content.append(f"* **Target Ability**: {case['target']}")
        report_content.append(f"#### Prompt Prefix:")
        report_content.append(f"```text\n{prompt}\n```")
        report_content.append(f"#### Model Generated Output:")
        report_content.append(f"```text\n{cleaned_output}\n```")
        report_content.append(f"\n")

    # Save Markdown report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
    print(f"Evaluation report written to: {report_file}")

    # Save structured JSON (machine-readable)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2, ensure_ascii=False)
    print(f"Structured results written to: {json_file}")

    # Log to MLflow
    if log_mlflow:
        _log_to_mlflow(structured_results, str(report_file), model_path)

    return str(report_file), structured_results


if __name__ == "__main__":
    run_evaluation()
