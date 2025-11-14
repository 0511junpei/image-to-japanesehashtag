import os
from flask import Flask, request, jsonify
from google.cloud import vision, translate_v3 as translate
from google.api_core.exceptions import GoogleAPICallError

app = Flask(__name__)

vision_client = vision.ImageAnnotatorClient()

translate_client = translate.TranslationServiceClient()

def detect_labels(image_base64):
    image = vision.Image(content=image_base64)
    features = [
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=15)
    ]
    request_body = vision.AnnotateImageRequest(image=image, features=features)
    response = vision_client.annotate_image(request=request_body)

    labels = []
    for label in response.label_annotations:
        if label.score >= 0.7:
            labels.append(label.description)
    return labels

def translate_to_japanese(texts):
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
    TRANSLATE_PARENT = f"projects/{PROJECT_ID}/locations/global"

    if not texts:
        return[]
    
    try:
        response = translate_client.translate_text(
            parent=TRANSLATE_PARENT,
            contents=texts,
            target_language_code="ja",
            mime_type="text/plain",
        )
        return [translation.translated_text for translation in response.translations]
    except GoogleAPICallError as e:
        print(f"Translation API Error: {e}")
        return texts

@app.route("/")
def test():
    return "Hello World"

@app.route("/", methods=["POST"])
def hashtag_generator():
    try:
        data = request.get_json(silent=True)
        image_base64 = data.get("image_data")

        if not image_base64:
            return jsonify({"error": "リクエストボディに 'image_data'（Base64）がありません"}), 400 

        # ラベル検出(Vision API)
        english_labels = detect_labels(image_base64)

        # 日本語翻訳(Translation API)
        japanese_labels = translate_to_japanese(english_labels)

        # ハッシュタグ化
        hashtags = []
        for label in japanese_labels:
            hashtags.append(f"#{label}")

        return jsonify({
            "status": "success",
            "hashtags": hashtags
        })
    except Exception as e:
        print(f"処理中に予期せぬエラーが発生しました: {e}")
        return jsonify({"error": f"サーバーエラー：{str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)