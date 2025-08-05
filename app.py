from flask import Flask, request, jsonify
from flask_cors import CORS
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import base64
import io
import os
import json

app = Flask(__name__)
CORS(app)

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "upbeat-element-468120-b4")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    cred_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    with open("/tmp/credentials.json", "w") as f:
        f.write(cred_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/credentials.json"

vertexai.init(project=project_id, location=location)
generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "Imagen API"})

@app.route("/generate-images", methods=["POST"])
def generate_images():
    try:
        data = request.json
        prompt = data.get("prompt")
        count = data.get("count", 4)
        
        if not prompt:
            return jsonify({"success": False, "error": "Prompt is required"}), 400
            
        if count > 6:
            count = 6
        
        print(f"Generating {count} images for: {prompt}")
        
        images = generation_model.generate_images(
            prompt=prompt,
            number_of_images=count,
            aspect_ratio="1:1",
            safety_filter_level="block_some",
            person_generation="allow_adult"
        )
        
        image_data = []
        for i, image in enumerate(images):
            buffered = io.BytesIO()
            image._pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            image_data.append({
                "id": i + 1,
                "imageData": img_base64,
                "mimeType": "image/png",
                "prompt": prompt
            })
        
        return jsonify({
            "success": True,
            "images": image_data,
            "prompt": prompt
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Failed to generate images",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
