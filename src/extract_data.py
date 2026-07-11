import json
import re
import sqlite3
from pathlib import Path

# Path configuration
DB_PATH = Path(r"C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\Data_Collection\novels.db")
OUTPUT_PATH = Path("data/corpus.jsonl")


def clean_text(text):
    """
    Remove web novel noise:
    - Ruby characters (e.g., 《読み》)
    - Front/Back matter (simplified detection)
    - Extreme empty lines
    """
    if not text:
        return ""

    # Remove ruby
    text = re.sub(r"《.*?》", "", text)
    # Remove obvious noise lines (often containing "目次", "前書き", etc. - needs careful tuning)
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line and len(line) > 2]

    return "\n".join(cleaned_lines)


def extract_to_jsonl():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Query to fetch all chapter texts
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

            if len(text) > 50:  # Ignore very short chapters
                data = {"text": text, "metadata": {"title": title}}
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    conn.close()
    print(f"Extraction completed. Data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    extract_to_jsonl()
