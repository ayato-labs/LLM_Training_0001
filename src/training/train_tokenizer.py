#!/usr/bin/env python3
"""
日本語小説向けSentencePiece BPEトークナイザの学習。
ADR-021: ドメイン固有64k語彙 + end_of_storyトークン。
"""

import argparse
import json
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer

CORPUS_PATH = Path("data/corpus.jsonl")
SPM_MODEL_PREFIX = Path("data/tokenizer_novel64k")
TOKENIZER_PATH = Path("data/tokenizer.json")

SPECIAL_TOKENS = [
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[PAD]",
    "[MASK]",
    "<|start_of_metadata|>",
    "<|end_of_metadata|>",
    "<|start_of_story|>",
    "<|end_of_story|>",
]

VOCAB_SIZE = 64000


def extract_corpus_text(corpus_path: Path, output_path: Path) -> int:
    """SentencePiece学習用にcorpus.jsonlからテキストを抽出。"""
    count = 0
    with (
        open(corpus_path, encoding="utf-8") as f_in,
        open(output_path, "w", encoding="utf-8") as f_out,
    ):
        for line in f_in:
            data = json.loads(line)
            text = data.get("text", "")
            if text:
                f_out.write(text + "\n")
                count += 1
    return count


def train_sentencepiece(corpus_txt: Path, model_prefix: Path):
    """SentencePiece BPEモデルを学習。"""
    special_symbols = ",".join(SPECIAL_TOKENS)

    spm.SentencePieceTrainer.train(
        input=str(corpus_txt),
        model_prefix=str(model_prefix),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9999,
        byte_fallback=True,
        normalization_rule_name="nfkc",
        split_digits=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=special_symbols,
    )
    print(f"SentencePiece model trained: {model_prefix}.model / .vocab")


def convert_to_hf_tokenizer(model_path: Path, output_path: Path):
    """SentencePieceモデルをHuggingFace tokenizer.jsonに変換。

    tokenizersライブラリを使用してtokenizer.jsonを適切に構築：
    - SPモデルからの語彙（ID 0-63999）
    - 特殊トークンはSPの位置に配置（追加ではない）
    - ▁プレフィックスのMetaspace前処理
    - added_tokensなし（すべて基本語彙）
    """
    import sentencepiece as spm
    from tokenizers import decoders, models, pre_tokenizers
    from tokenizers.normalizers import NFKC

    # SP語彙の読み込み
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path) + ".model")

    vocab = {}
    for i in range(sp.get_piece_size()):
        vocab[sp.id_to_piece(i)] = i

    print(f"  Vocab size from SP: {len(vocab)}")
    print(
        f"  Special tokens: unk={vocab.get('<unk>')}, bos={vocab.get('<s>')}, "
        f"eos={vocab.get('</s>')}, pad={vocab.get('<pad>')}"
    )

    # tokenizersライブラリでトークナイザを構築
    tokenizer = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=[],
            unk_token="<unk>",
            continuing_subword_prefix="",
            end_of_word_suffix="",
            fuse_unk=True,
            byte_fallback=True,
            ignore_merges=False,
        )
    )

    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="\u2581",
        prepend_scheme="always",
        split=True,
    )
    tokenizer.decoder = decoders.Metaspace(
        replacement="\u2581",
        prepend_scheme="always",
        split=True,
    )

    # added_tokensなしで保存 - すべて語彙内
    tokenizer.save(str(output_path), pretty=True)
    print(f"HuggingFace tokenizer.json saved to {output_path}")

    # 検証: HFで読み込んで特殊トークンIDを確認
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(output_path))
    hf_tokenizer.unk_token = "<unk>"
    hf_tokenizer.bos_token = "<s>"
    hf_tokenizer.eos_token = "</s>"
    hf_tokenizer.pad_token = "<pad>"

    print("  Verification:")
    print(f"    unk_token_id={hf_tokenizer.unk_token_id} (expected=0)")
    print(f"    bos_token_id={hf_tokenizer.bos_token_id} (expected=1)")
    print(f"    eos_token_id={hf_tokenizer.eos_token_id} (expected=2)")
    print(f"    pad_token_id={hf_tokenizer.pad_token_id} (expected=3)")
    print(f"    vocab_size={len(hf_tokenizer)} (expected={VOCAB_SIZE})")


def main():
    global VOCAB_SIZE
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for novels")
    parser.add_argument("--vocab-size", type=int, default=64000, help="Vocabulary size")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH, help="Corpus JSONL path")
    parser.add_argument("--output", type=Path, default=TOKENIZER_PATH, help="Output tokenizer.json")
    parser.add_argument("--skip-extract", action="store_true", help="Skip corpus extraction")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip SentencePiece training (use existing .model)",
    )
    args = parser.parse_args()

    VOCAB_SIZE = args.vocab_size

    corpus_txt = Path("data/corpus.txt")

    if not args.skip_extract:
        print("Extracting corpus text...")
        count = extract_corpus_text(args.corpus, corpus_txt)
        print(f"Extracted {count} documents")

    if not corpus_txt.exists():
        raise FileNotFoundError(
            f"Corpus text not found: {corpus_txt}. Run without --skip-extract first."
        )

    if not args.skip_training:
        print(f"Training SentencePiece BPE (vocab={VOCAB_SIZE})...")
        train_sentencepiece(corpus_txt, SPM_MODEL_PREFIX)
    else:
        print("Skipping SentencePiece training (using existing model)")

    print("Converting to HuggingFace format...")
    convert_to_hf_tokenizer(SPM_MODEL_PREFIX, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
