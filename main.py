import os
from flask import Flask, request, jsonify
from google.cloud import vision, translate_v3 as translate
from google.api_core.exceptions import GoogleAPICallError

app = Flask(__name__)

def detect_labels(image_base64):
    image = vision.Image(content=image_base64)
    features = [
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=15)
    ]
    request_body = vision.AnnotateImageRequest(image=image, features=features)
    vision_client = vision.ImageAnnotatorClient()
    response = vision_client.annotate_image(request=request_body)

    labels = []
    for label in response.label_annotations:
        if label.score >= 0.7:
            labels.append(label.description)
    return labels

def translate_to_japanese(texts):
    if not texts:
        return[]
    
    # 1. 公式キーで取得を試みる
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

    # 2. 公式キーで取得できなかった場合、古い代替キーで試行
    if PROJECT_ID is None:
        PROJECT_ID = os.environ.get("GCP_PROJECT")
        print(f"DEBUG: GOOGLE_CLOUD_PROJECT was None. Trying GCP_PROJECT: '{PROJECT_ID}'")

    # 3. どちらのキーでも取得できなかった場合の処理
    if PROJECT_ID is None:
        print("FATAL ERROR: Neither GOOGLE_CLOUD_PROJECT nor GCP_PROJECT is set. Translation will fail.")
        # この時点でエラーになることを確定させるため、無効な値を返す
        return []
    
    TRANSLATE_PARENT = f"projects/{PROJECT_ID}/locations/global"

    translate_client = translate.TranslationServiceClient()
    
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