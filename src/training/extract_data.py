import json
import re
import sqlite3
from pathlib import Path

# パス設定
DB_PATH = Path(r"C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\Data_Collection\novels.db")
OUTPUT_PATH = Path("data/corpus.jsonl")


def clean_text(text):
    """Web小説のノイズ除去：

    - ルビ文字（例：《読み》）の削除
    - 前付け/後付けの簡易検出
    - 極端な空行の削除
    """
    if not text:
        return ""

    # ルビを削除
    text = re.sub(r"《.*?》", "", text)
    # 明らかなノイズ行を削除（"目次"、"前書き"などを含む行 - 調整が必要）
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line and len(line) > 2]

    return "\n".join(cleaned_lines)


def extract_to_jsonl():
    """SQLiteデータベースからデータを抽出してJSONL形式で保存。"""
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 全章のテキストを取得するクエリ
    query = """
    SELECT c.body_text, n.title
    FROM chapters c
    JOIN novels n ON c.novel_id = n.id
    """

    cursor.execute(query)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in cursor.fetchall():
            body_text, title = row
            text = clean_text(body_text)

            if len(text) > 50:  # 非常に短い章は無視
                data = {"text": text, "metadata": {"title": title}}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    conn.close()
    print(f"Extraction completed. Data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    extract_to_jsonl()
