import sqlite3
import json
from pathlib import Path

# Path configuration
DB_PATH = Path(r"C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\Data_Collection\novels.db")
OUTPUT_PATH = Path("data/corpus.jsonl")

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
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for row in cursor.fetchall():
            body_text, title = row
            # Basic cleaning: remove empty lines
            text = "\n".join([line.strip() for line in body_text.splitlines() if line.strip()])
            if text:
                data = {"text": text, "metadata": {"title": title}}
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    conn.close()
    print(f"Extraction completed. Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    # Ensure data directory exists
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    extract_to_jsonl()
