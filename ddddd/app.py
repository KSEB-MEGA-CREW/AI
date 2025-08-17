
import os
from flask import Flask, request, jsonify
from transformers import pipeline
import config

# Import the necessary functions from other modules
from download_from_s3 import download_model_from_s3
from inference import inference
from mapping_point import map_gloss_to_point
from data_load import load_valid_gloss_set

# --- Global Variables ---
# These will be initialized by initialize_app()
app = Flask(__name__)
text_to_gloss_pipeline = None
valid_gloss_set = None

# --- 1. Initialization Function ---
def initialize_app():
    """
    Downloads the model if not present, then loads all assets (pipeline, gloss set).
    This function is called once before the app starts.
    Returns:
        A tuple of (initialized_pipeline, initialized_gloss_set)
    """
    print("--- Initializing Application ---")
    
    # Use absolute paths for reliability
    local_model_path = os.path.abspath(config.LOCAL_MODEL_PATH)
    local_uni_gloss_set_path = os.path.abspath(config.LOCAL_UNI_GLOSS_SET_PATH)

    # Step 1: Ensure model is downloaded BEFORE trying to load it.
    print(f"모델 파일이 경로에 있는지 확인 \n파일 경로: {local_model_path}...")
    download_model_from_s3(
        s3_model_path=config.S3_MODEL_PATH, 
        s3_gloss_set_path=config.S3_UNI_GLOSS_SET, 
        local_model_path=local_model_path, 
        local_gloss_set_path=local_uni_gloss_set_path
    )
    print("모델 다운 완료.")

    # Step 2: Now that files are guaranteed to exist, create the pipeline.
    print("Loading Text-to-Gloss pipeline...")
    try:
        initialized_pipeline = pipeline(
            "translation",
            model=local_model_path,
            tokenizer=local_model_path,
            device_map="auto"
        )
        print("Pipeline 초기화 성공!!!.")
    except Exception as e:
        print(f"FATAL: 파이프라인 초기화 오류: {e}")
        return None, None

    # Step 3: Load the valid gloss set
    print("검증 셋 불러오기...")
    initialized_gloss_set = load_valid_gloss_set()
    print("검증 셋 불러오기.")
    
    print("--- Application Initialized Successfully ---")
    return initialized_pipeline, initialized_gloss_set

# --- 2. API Endpoints ---
@app.route('/', methods=['GET'])
def root_test():
    if text_to_gloss_pipeline is None:
        return jsonify({"error": "Model pipeline is not available. The service is likely initializing or has failed."}), 503
    return jsonify({"status": "Service is running"}), 200

@app.route('/T2G/translate', methods=['GET'])
def translate_text():
    """
    Takes a text parameter via URL and returns the translation result as JSON.
    Example: http://127.0.0.1:1958/T2G/translate?text=Hello
    """
    if text_to_gloss_pipeline is None:
        return jsonify({"error": "Model pipeline is not available."}), 503

    text_to_translate = request.args.get('text')
    if not text_to_translate:
        return jsonify({"error": "The 'text' parameter is required."}), 400

    try:
        # Use the globally loaded pipeline and gloss set
        prediction_gloss_list = inference(text_to_translate, text_to_gloss_pipeline)
        result = map_gloss_to_point(prediction_gloss_list, valid_gloss_set)
        return jsonify(result), 200  # 200 OK for a successful request

    except Exception as e:
        print(f"Error during translation: {e}")
        return jsonify({"error": "An error occurred during translation."}), 500

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    # Initialize the app and load all assets first
    text_to_gloss_pipeline, valid_gloss_set = initialize_app()
    
    # Run the Flask app only if the initialization was successful
    if text_to_gloss_pipeline and valid_gloss_set:
        app.run(host='0.0.0.0', port=1958, debug=True)
    else:
        print("Application failed to initialize. Shutting down.")
