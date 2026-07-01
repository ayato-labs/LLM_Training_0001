import re
import sqlite3
import json
from pathlib import Path
import training_config as config

"""
小説特化型LLM学習用データセット生成パイプライン
--------------------------------------------------
本モジュールは、Novel_Data_Collectionで収集されたDBデータを読み込み、
LLM学習に最適なJSONL形式へ変換する前処理を担当します。

主な機能とメタデータ生成:
1. 構造化: 作品情報、章メタデータ、本文の結合
2. 会話率計算: 
   - 鉤括弧「」内の文字数合計を用いた会話密度の算出
3. 感情分析:
   - 簡易感情辞書を用いた、シーン単位の感情(Positive/Negative/Neutral)判定
4. 文体統計:
   - 文字数および文字種(漢字、ひらがな、カタカナ、英字、記号)の構成比率計算

これらをメタ情報(metrics)として付与し、モデルがコンテキストの
「密度」「トーン」「文体構成」を数値として理解できるように設計されています。
"""

# 簡易的な感情辞書
POSITIVE_WORDS = {'素晴らしい', '楽しい', '嬉しい', '大好き', '成功', '美しい', '希望', '愛', '優しい'}
NEGATIVE_WORDS = {'悲しい', '辛い', '憎い', '失敗', '怖い', '醜い', '絶望', '憎悪', '冷たい', '死'}

def calculate_conversation_rate(text):
    """
    鉤括弧「」で囲まれた文字数の合計を算出し、全体に対する割合を返す
    (改行を含む長台詞にも対応)
    """
    matches = re.findall(r'「(.*?)」', text, re.DOTALL)
    conv_chars = sum(len(match) for match in matches)
    total_chars = len(text)
    
    if total_chars == 0: return 0.0
    return round(conv_chars / total_chars, 4)

def get_sentiment(text):
    pos_count = sum(text.count(word) for word in POSITIVE_WORDS)
    neg_count = sum(text.count(word) for word in NEGATIVE_WORDS)
    
    total = pos_count + neg_count
    if total == 0:
        return {"label": "neutral", "score": 1.0}
    
    score = (pos_count - neg_count) / total
    if abs(score) < 0.2:
        return {"label": "neutral", "score": abs(score)}
    elif score > 0:
        return {"label": "positive", "score": score}
    else:
        return {"label": "negative", "score": abs(score)}

def get_text_stats(text):
    total = len(text)
    if total == 0:
        return {
            "total_chars": 0, "kanji_ratio": 0.0, "hiragana_ratio": 0.0,
            "katakana_ratio": 0.0, "roman_ratio": 0.0, "symbol_ratio": 0.0
        }
    
    kanji = sum(1 for c in text if '一' <= c <= '龯')
    hiragana = sum(1 for c in text if 'ぁ' <= c <= 'ゟ')
    katakana = sum(1 for c in text if 'ァ' <= c <= 'ヿ')
    roman = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    symbols = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    
    return {
        "total_chars": total,
        "kanji_ratio": round(kanji / total, 4),
        "hiragana_ratio": round(hiragana / total, 4),
        "katakana_ratio": round(katakana / total, 4),
        "roman_ratio": round(roman / total, 4),
        "symbol_ratio": round(symbols / total, 4)
    }

def split_text_with_overlap(text, chunk_size=2000, overlap=200):
    """
    文章を重なり（overlap）を持たせて分割する。
    1024トークン(約2000〜2200文字)に収まるようにチャンクサイズを設定。
    """
    if len(text) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # 重なり分を引いて次の開始位置を設定
        start += (chunk_size - overlap)
    return chunks

def export_db_to_jsonl():
    db_path = Path(r"C:\Users\saiha\My_Service\programing\LLM\Novel_LLM\Novel_Data_Collection\novels.db")
    output_path = config.DATA_PATH
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT n.id, n.title, n.synopsis, n.genre, n.tags, c.chapter_number, c.subtitle, c.body_text
        FROM chapters c
        JOIN novels n ON c.novel_id = n.id
        ORDER BY n.id, c.chapter_number
    """)
    all_data = cursor.fetchall()
    
    # 統計計算用の辞書
    novel_stats = {} 
    
    for row in all_data:
        novel_id = row[0]
        body = row[7]
        lines = [line.strip() for line in body.split('\n') if line.strip()]
        
        if novel_id not in novel_stats:
            novel_stats[novel_id] = {"total_conv_lines": 0, "total_lines": 0}
        
        novel_stats[novel_id]["total_lines"] += len(lines)
        novel_stats[novel_id]["total_conv_lines"] += sum(1 for line in lines if '「' in line)
        
    exported_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in all_data:
            novel_id, title, synopsis, genre, tags, number, subtitle, body = row
            
            total_lines = novel_stats[novel_id]["total_lines"]
            novel_conv = (novel_stats[novel_id]["total_conv_lines"] / total_lines) if total_lines > 0 else 0.0
            
            # 本文をオーバーラップ付きで分割
            body_chunks = split_text_with_overlap(body, chunk_size=2000, overlap=200)
            
            for chunk_idx, chunk_body in enumerate(body_chunks):
                # チャンクごとにメトリクスを再計算することでプレフィックスとの整合性を担保
                chapter_conv = calculate_conversation_rate(chunk_body)
                sentiment = get_sentiment(chunk_body)
                stats = get_text_stats(chunk_body)
                
                # 条件付き学習用プレフィックス（ADR-0013: 特殊トークンによる境界強化）
                metadata_prefix = (
                    f"<|start_of_metadata|>\n"
                    f"作品名: {title or '不明'}\n"
                    f"ジャンル: {genre or '未設定'}\n"
                    f"会話率(全体): {novel_conv:.2%}\n"
                    f"会話率(章): {chapter_conv:.2%}\n"
                    f"感情: {sentiment['label']}\n"
                    f"文字数: {stats['total_chars']}\n"
                    f"タグ: {tags or 'なし'}\n"
                    f"<|end_of_metadata|>\n"
                    f"<|start_of_story|>"
                )
                formatted_text = metadata_prefix + chunk_body
                
                entry = {
                    "text": formatted_text,
                    "metadata": {
                        "title": title,
                        "synopsis": synopsis,
                        "genre": genre,
                        "tags": tags,
                        "chapter": number,
                        "subtitle": f"{subtitle or ''} (Part {chunk_idx + 1})" if len(body_chunks) > 1 else (subtitle or ""),
                        "metrics": {
                            "novel_conversation_rate": novel_conv,
                            "chapter_conversation_rate": chapter_conv,
                            "sentiment": sentiment,
                            "text_stats": stats,
                            "chunk_index": chunk_idx,
                            "total_chunks": len(body_chunks)
                        }
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                exported_count += 1
            
    conn.close()
    print(f"Preprocessing completed: {exported_count} chunks exported to {output_path}")

if __name__ == "__main__":
    export_db_to_jsonl()
