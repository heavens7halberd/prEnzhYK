from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_tone(text):
    try:
        result = sentiment_analyzer(text)[0]
        return {"label": result['label'], "score": result['score']}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    while True:
        text = input("\nВведите путь к изображению (или введите «exit», чтобы выйти): ")
        if text.lower() == "exit":
            break
        tone = analyze_tone(text)
        if "error" in tone:
            print("Ошибка:", tone["error"])
        else:
            print(f"Настроение: {tone['label']}, Уверенность: {tone['score']:.2f}")
