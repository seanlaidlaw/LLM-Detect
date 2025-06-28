from transformers import pipeline
import torch

pipe = pipeline("text-classification", model="yaful/MAGE")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("yaful/MAGE")
model = AutoModelForSequenceClassification.from_pretrained("yaful/MAGE")

# Load text from file
with open('Data/ModelExamples/gpt.txt', 'r') as file:
    text = file.read()

# Define label mapping
label_mapping = {
    "0": "GPT",
    "1": "Human",
    0: "GPT",
    1: "Human"
}

# Perform classification using pipeline
try:
    result = pipe(text)
    # print(f"Original Pipeline Result: {result}")
    
    # Transform the prediction dictionary
    original_label = result[0]['label']
    mapped_label = label_mapping.get(original_label, str(original_label))
    
    transformed_result = {
        "label": mapped_label,  # Map numeric label to name
        "confidence": result[0]['score'],
        "model": "MAGE"
    }
    
    print(f"Transformed Result: {transformed_result}")
    # print(f"Predicted Label: {transformed_result['label']}")
    # print(f"Confidence Score: {transformed_result['confidence']:.4f}")
    # print(f"Detection Model: {transformed_result['model']}")
    
except Exception as e:
    print(f"Error with pipeline: {e}")
