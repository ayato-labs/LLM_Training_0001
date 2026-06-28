import sqlite3
import json
from pathlib import Path
import training_config as config

def calculate_conversation_rate(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines: return 0.0
    # 「で始まる行を会話とみなす単純なヒューリスティック
    conv_lines = sum(1 for line in lines if line.startswith('「'))
    return round(conv_lines / len(lines), 4)

def export_db_to_jsonl():
    """
    DBからデータを読み込み、会話率を算出して学習用のJSONLへ変換する前処理タスク
    """
    db_path = Path(r"C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\Novel_Data_Collection\novels.db")
    output_path = config.DATA_PATH
    
    # 出力先ディレクトリの確保
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # データを全件取得
    cursor.execute("""
        SELECT n.id, n.title, n.synopsis, n.genre, n.tags, c.chapter_number, c.subtitle, c.body_text
        FROM chapters c
        JOIN novels n ON c.novel_id = n.id
        ORDER BY n.id, c.chapter_number
    """)
    all_data = cursor.fetchall()
    
    # 統計計算用の辞書
    novel_stats = {} # {novel_id: {"total_conv_lines": 0, "total_lines": 0}}
    
    # 1パス目: 小説全体の会話率計算用統計を作成
    for row in all_data:
        novel_id = row[0]
        body = row[7]
        lines = [line.strip() for line in body.split('\n') if line.strip()]
        
        if novel_id not in novel_stats:
            novel_stats[novel_id] = {"total_conv_lines": 0, "total_lines": 0}
        
        novel_stats[novel_id]["total_lines"] += len(lines)
        novel_stats[novel_id]["total_conv_lines"] += sum(1 for line in lines if line.startswith('「'))
        
    # 2パス目: JSONL出力
    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_data:
            novel_id, title, synopsis, genre, tags, number, subtitle, body = row
            
            # 会話率計算
            chapter_conv = calculate_conversation_rate(body)
            total_lines = novel_stats[novel_id]["total_lines"]
            novel_conv = (novel_stats[novel_id]["total_conv_lines"] / total_lines) if total_lines > 0 else 0.0
            
            # 条件付き学習用プレフィックスの構築
            metadata_prefix = (
                f"作品名: {title or '不明'}\n"
                f"ジャンル: {genre or '未設定'}\n"
                f"会話率(全体): {novel_conv:.2%}\n"
                f"会話率(章): {chapter_conv:.2%}\n"
                f"タグ: {tags or 'なし'}\n\n"
            )
            formatted_text = metadata_prefix + body
            
            entry = {
                "text": formatted_text,
                "metadata": {
                    "title": title,
                    "synopsis": synopsis,
                    "genre": genre,
                    "tags": tags,
                    "chapter": number,
                    "subtitle": subtitle,
                    "metrics": {
                        "novel_conversation_rate": novel_conv,
                        "chapter_conversation_rate": chapter_conv
                    }
                }
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    conn.close()
    print(f"Preprocessing completed: {len(all_data)} chapters exported to {output_path}")

if __name__ == "__main__":
    export_db_to_jsonl()
