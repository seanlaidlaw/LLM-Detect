from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import socket

app = Flask(__name__)

# Global variables for models
model_pipeline = None
modernbert_tokenizer = None
modernbert_model = None

def find_available_port(start_port=5000, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

def load_models():
    """Load both MAGE and ModernBERT models"""
    global model_pipeline, modernbert_tokenizer, modernbert_model
    
    try:
        # Load MAGE model
        model_pipeline = pipeline("text-classification", model="yaful/MAGE")
        print("MAGE model loaded successfully!")
    except Exception as e:
        print(f"Error loading MAGE model: {e}")
        model_pipeline = None
    
    try:
        # Load ModernBERT model
        modernbert_tokenizer = AutoTokenizer.from_pretrained("GeorgeDrayson/modernbert-ai-detection-raid-mage")
        modernbert_model = AutoModelForSequenceClassification.from_pretrained("GeorgeDrayson/modernbert-ai-detection-raid-mage")
        print("ModernBERT model loaded successfully!")
    except Exception as e:
        print(f"Error loading ModernBERT model: {e}")
        modernbert_tokenizer = None
        modernbert_model = None

def classify_with_mage(text):
    """Classify text using the MAGE model"""
    if not model_pipeline or not text.strip():
        return {"label": "Human", "confidence": 1.0, "model": "MAGE"}
    
    try:
        # Define comprehensive label mapping (same as test detection script)
        label_mapping = {
            "0": "GPT",
            "1": "Human",
            "LABEL_0": "Human",
            "LABEL_1": "GPT",
            0: "GPT",
            1: "Human"
        }
        
        # Perform classification
        result = model_pipeline(text)
        
        # Transform the prediction
        original_label = result[0]['label']
        mapped_label = label_mapping.get(original_label, str(original_label))
        
        return {
            "label": mapped_label,
            "confidence": result[0]['score'],
            "model": "MAGE"
        }
        
    except Exception as e:
        print(f"Error in MAGE classification: {e}")
        return {"label": "Human", "confidence": 1.0, "model": "MAGE"}

def classify_with_modernbert(text):
    """Classify text using the ModernBERT model"""
    if not modernbert_tokenizer or not modernbert_model or not text.strip():
        return {"label": "Human", "confidence": 1.0, "model": "ModernBERT"}
    
    try:
        # Tokenize and classify
        inputs = modernbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = modernbert_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probability of machine-generated text (index 1)
        machine_prob = probabilities[0][1].item()
        human_prob = probabilities[0][0].item()
        
        # Determine label based on higher probability
        if machine_prob > human_prob:
            label = "GPT"
            confidence = machine_prob
        else:
            label = "Human"
            confidence = human_prob
        
        return {
            "label": label,
            "confidence": confidence,
            "model": "ModernBERT"
        }
        
    except Exception as e:
        print(f"Error in ModernBERT classification: {e}")
        return {"label": "Human", "confidence": 1.0, "model": "ModernBERT"}

def classify_text(text):
    """Classify text using both models"""
    if not text.strip():
        return {
            "mage": {"label": "Human", "confidence": 1.0, "model": "MAGE"},
            "modernbert": {"label": "Human", "confidence": 1.0, "model": "ModernBERT"},
            "any_gpt": False
        }
    
    # Get results from both models
    mage_result = classify_with_mage(text)
    modernbert_result = classify_with_modernbert(text)
    
    # Check if either model detected GPT
    any_gpt = mage_result["label"] == "GPT" or modernbert_result["label"] == "GPT"
    
    combined_result = {
        "mage": mage_result,
        "modernbert": modernbert_result,
        "any_gpt": any_gpt
    }
    
    print(f"Combined classification result: {combined_result}")
    return combined_result

@app.route('/')
def index():
    """Main page with text input"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """API endpoint for text classification"""
    data = request.get_json()
    text = data.get('text', '')
    
    # Perform classification with both models
    result = classify_text(text)
    
    return jsonify(result)

@app.route('/status')
def status():
    """Get current classification status"""
    return jsonify({
        "mage": {"label": "Human", "confidence": 1.0, "model": "MAGE"},
        "modernbert": {"label": "Human", "confidence": 1.0, "model": "ModernBERT"},
        "any_gpt": False
    })

if __name__ == '__main__':
    # Load models on startup
    print("Loading models...")
    load_models()
    
    # Find available port
    try:
        port = find_available_port(5000)
        print(f"Starting Flask app on port {port}")
        print(f"Open your browser and go to: http://localhost:{port}")
        app.run(debug=False, host='0.0.0.0', port=port)
    except RuntimeError as e:
        print(f"Error: {e}")
        exit(1) 