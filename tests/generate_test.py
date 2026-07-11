from src.eval.generate import generate_text

prompt = (
    "作品名: 魔女と傭兵\n"
    "ジャンル: ハイファンタジー〔ファンタジー〕\n"
    "会話率(全体): 31.76%\n"
    "会話率(章): 30.00%\n"
    "感情: negative\n"
    "文字数: 1000\n"
    "タグ: R15,残酷な描写あり,男主人公,ダンジョン,傭兵,魔女,現地人主人公,双刃剣,主人公強い\n\n\n\n"
    "「おい、ジグ」"
)

try:
    result = generate_text(prompt)
    with open("generate_result.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
