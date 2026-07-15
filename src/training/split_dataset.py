#!/usr/bin/env python3
"""
データセット分割スクリプト：小説単位分割
======================================
学習データと検証データの間のデータ漏洩を防ぐため、
ユニークな小説タイトル별로データセットを分割。

各小説の全章は学習セットまたは検証セットのいずれかにまとめて配置。
分割比率: 99% 学習 / 1% 検証（設定可能）。
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.common.logger import logger


def split_dataset(
    input_path: Path,
    train_output: Path,
    val_output: Path,
    val_ratio: float = 0.01,
    seed: int = 42,
) -> dict:
    """小説タイトル별로データセットを分割。"""
    try:
        title_to_lines = defaultdict(list)

        logger.info(f"Reading dataset from {input_path}...")
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    title = data.get("metadata", {}).get("title", "unknown")
                    title_to_lines[title].append(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line: {e}")
                    continue

        titles = list(title_to_lines.keys())
        total_chunks = sum(len(v) for v in title_to_lines.values())
        logger.info(f"Found {len(titles)} unique novels, {total_chunks} total chunks")

        random.seed(seed)
        random.shuffle(titles)

        # 検証用小説数
        val_count = max(1, int(len(titles) * val_ratio))
        val_titles = set(titles[:val_count])
        train_titles = set(titles[val_count:])

        logger.info(f"Split: {len(train_titles)} train novels, {len(val_titles)} val novels")

        train_chunks = 0
        val_chunks = 0

        # 出力ファイルに書き出し
        with (
            open(train_output, "w", encoding="utf-8") as f_train,
            open(val_output, "w", encoding="utf-8") as f_val,
        ):
            for title in titles:
                lines = title_to_lines[title]
                if title in val_titles:
                    f_val.writelines(lines)
                    val_chunks += len(lines)
                else:
                    f_train.writelines(lines)
                    train_chunks += len(lines)

        stats = {
            "train_novels": len(train_titles),
            "val_novels": len(val_titles),
            "train_chunks": train_chunks,
            "val_chunks": val_chunks,
            "total_chunks": train_chunks + val_chunks,
        }

        logger.info(f"Split complete: train={train_chunks} chunks, val={val_chunks} chunks")
        return stats

    except Exception as e:
        logger.error(f"Dataset split failed: {e}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="Split dataset by novel title")
    parser.add_argument(
        "--input", default="../DataPreprocessing/data/dataset.jsonl", help="Input dataset path"
    )
    parser.add_argument(
        "--train-output", default="data/train_dataset.jsonl", help="Training output path"
    )
    parser.add_argument(
        "--val-output", default="data/val_dataset.jsonl", help="Validation output path"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.01, help="Validation ratio (default: 0.01)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    train_output = Path(args.train_output).resolve()
    val_output = Path(args.val_output).resolve()

    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    stats = split_dataset(input_path, train_output, val_output, args.val_ratio, args.seed)

    stats_path = train_output.parent / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
