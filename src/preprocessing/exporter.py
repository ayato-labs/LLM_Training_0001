import sqlite3
import json
from pathlib import Path
import project_config as config

def export_db_to_jsonl():
    """
    DBからデータを読み込み、学習用のJSONLへ変換する前処理タスク
    """
    db_path = Path(r"C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\Novel_Data_Collection\novels.db")
    output_path = config.DATA_PATH
    
    # 出力先ディレクトリの確保
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # データを結合して取得
    cursor.execute("""
        SELECT n.title, n.synopsis, c.chapter_number, c.subtitle, c.body_text
        FROM chapters c
        JOIN novels n ON c.novel_id = n.id
        ORDER BY n.id, c.chapter_number
    """)
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in cursor:
            title, synopsis, number, subtitle, body = row
            entry = {
                "text": body,
                "metadata": {
                    "title": title,
                    "synopsis": synopsis,
                    "chapter": number,
                    "subtitle": subtitle
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
            
    conn.close()
    print(f"Preprocessing completed: {count} chapters exported to {output_path}")

if __name__ == "__main__":
    export_db_to_jsonl()
