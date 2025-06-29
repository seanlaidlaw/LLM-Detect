from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import socket
import re
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np

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

def validate_text_input(text):
    """Validate text input against criteria and return validation results"""
    # Trim leading/trailing whitespace and normalize internal whitespace
    text = ' '.join(text.split())
    
    if not text.strip():
        return {
            "valid": False,
            "warnings": [],
            "errors": ["Text cannot be empty"],
            "metrics": {}
        }
    
    # Calculate text metrics
    metrics = calculate_text_metrics(text)
    
    errors = []
    warnings = []
    
    # HARD REQUIREMENTS (must be met)
    if metrics['token_count'] <= 40:
        errors.append(f"Token count ({metrics['token_count']}) must be > 40")
    if metrics['token_count'] >= 950:
        errors.append(f"Token count ({metrics['token_count']}) must be < 950")
    if metrics['sentence_count'] < 2:
        errors.append(f"Sentence count ({metrics['sentence_count']}) must be at least 2")
    if metrics['paragraph_count'] != 1:
        errors.append(f"Paragraph count ({metrics['paragraph_count']}) must be exactly 1")
    
    # WARNING THRESHOLDS (will show orange warning)
    if metrics['sentence_count'] > 50:
        warnings.append(f"Sentence count ({metrics['sentence_count']}) exceeds recommended maximum of 50")
    if metrics['word_count'] < 30:
        warnings.append(f"Word count ({metrics['word_count']}) is below recommended minimum of 30")
    if metrics['word_count'] > 750:
        warnings.append(f"Word count ({metrics['word_count']}) exceeds recommended maximum of 750")
    if metrics['avg_words_per_sentence'] < 7:
        warnings.append(f"Average words per sentence ({metrics['avg_words_per_sentence']:.1f}) is below recommended range (7-31)")
    elif metrics['avg_words_per_sentence'] > 31:
        warnings.append(f"Average words per sentence ({metrics['avg_words_per_sentence']:.1f}) exceeds recommended range (7-31)")
    if metrics['avg_paragraph_length'] < 150:
        warnings.append(f"Paragraph length ({metrics['avg_paragraph_length']:.0f} chars) is below recommended range (150-4000)")
    elif metrics['avg_paragraph_length'] > 4000:
        warnings.append(f"Paragraph length ({metrics['avg_paragraph_length']:.0f} chars) exceeds recommended range (150-4000)")
    if metrics['punctuation_density'] < 0.01:
        warnings.append(f"Punctuation density ({metrics['punctuation_density']:.3f}) is below recommended range (0.01-0.05)")
    elif metrics['punctuation_density'] > 0.05:
        warnings.append(f"Punctuation density ({metrics['punctuation_density']:.3f}) exceeds recommended range (0.01-0.05)")
    
    return {
        "valid": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
        "metrics": metrics
    }

def calculate_text_metrics(text):
    """Calculate comprehensive text metrics for validation"""
    # Token count
    if modernbert_tokenizer:
        tokens = modernbert_tokenizer.encode(text, add_special_tokens=True)
        token_count = len(tokens)
    else:
        # Fallback: rough estimate
        token_count = len(text.split()) * 1.3
    
    # Word count
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha()]
    word_count = len(words)
    
    # Sentence count
    sentences = sent_tokenize(text)
    sentence_count = len(sentences)
    
    # Paragraph count
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)
    
    # Average words per sentence
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    
    # Average paragraph length
    if paragraph_count > 0:
        avg_paragraph_length = np.mean([len(p) for p in paragraphs])
    else:
        avg_paragraph_length = len(text)
    
    # Punctuation density
    import string
    punctuation_chars = sum(1 for char in text if char in string.punctuation)
    punctuation_density = punctuation_chars / len(text) if len(text) > 0 else 0
    
    return {
        'token_count': token_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': paragraph_count,
        'avg_words_per_sentence': avg_words_per_sentence,
        'avg_paragraph_length': avg_paragraph_length,
        'punctuation_density': punctuation_density
    }

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
    # Trim leading/trailing whitespace and normalize internal whitespace
    text = ' '.join(text.split())
    
    if not text.strip():
        return {
            "mage": {"label": "Human", "confidence": 1.0, "model": "MAGE"},
            "modernbert": {"label": "Human", "confidence": 1.0, "model": "ModernBERT"},
            "any_gpt": False,
            "validation": {"valid": False, "warnings": [], "errors": ["Text is empty"], "metrics": {}}
        }
    
    # Validate text input
    validation = validate_text_input(text)
    
    # Only classify if validation passes
    if not validation["valid"]:
        return {
            "mage": {"label": "Human", "confidence": 1.0, "model": "MAGE"},
            "modernbert": {"label": "Human", "confidence": 1.0, "model": "ModernBERT"},
            "any_gpt": False,
            "validation": validation
        }
    
    # Get results from both models
    mage_result = classify_with_mage(text)
    modernbert_result = classify_with_modernbert(text)
    
    # Check if either model detected GPT
    any_gpt = mage_result["label"] == "GPT" or modernbert_result["label"] == "GPT"
    
    combined_result = {
        "mage": mage_result,
        "modernbert": modernbert_result,
        "any_gpt": any_gpt,
        "validation": validation
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
        "any_gpt": False,
        "validation": {"valid": True, "warnings": [], "errors": [], "metrics": {}}
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