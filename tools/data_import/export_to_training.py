import sqlite3
import json
import os
from pathlib import Path

# 設定: 相対パスで指定
DB_PATH = Path("tools/data_import/novels.db")
OUTPUT_FILE = Path("data/dataset.jsonl")

def export():
    # 出力先ディレクトリの確認
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 小説と章を結合して取得
    # 注意: テーブル名やカラム名は以前の調査に基づく
    # novelsテーブル: id, title, synopsis
    # chaptersテーブル: novel_id, chapter_number, subtitle, body_text
    try:
        cursor.execute("""
            SELECT n.title, n.synopsis, c.chapter_number, c.subtitle, c.body_text
            FROM chapters c
            JOIN novels n ON c.novel_id = n.id
            ORDER BY n.id, c.chapter_number
        """)
        
        count = 0
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for row in cursor:
                title, synopsis, number, subtitle, body = row
                
                # 構造化データセット（JSONL）
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
                
        print(f"Successfully exported {count} chapters to {OUTPUT_FILE}")
        
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    export()
