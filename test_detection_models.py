from transformers import pipeline
import torch
from datasets import load_dataset

pipe_mage = pipeline("text-classification", model="yaful/MAGE")

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("yaful/MAGE")
# model = AutoModelForSequenceClassification.from_pretrained("yaful/MAGE")

pipe_bert = pipeline("text-classification", model="GeorgeDrayson/modernbert-ai-detection-raid-mage")

# Define model configurations with their names
model_configs = [
    {"pipeline": pipe_mage, "name": "MAGE"},
    {"pipeline": pipe_bert, "name": "ModernBERT"}
]

# Load text from file
# with open('Data/ModelExamples/gpt.txt', 'r') as file:
with open('Data/ModelExamples/human.txt', 'r') as file:
# with open('Data/example_paragraphs/nature_genetics_style.txt', 'r') as file:
# with open('Data/example_paragraphs/gpt_style.txt', 'r') as file:
    text = file.read()
    
# Load the MAGE dataset
print("Loading MAGE dataset...")
ds = load_dataset("yaful/MAGE")
print(f"Dataset loaded. Train size: {len(ds['train'])}, Test size: {len(ds['test'])}")

# Analyze ground truth labels to verify mapping
print("\n" + "="*50)
print("ANALYZING GROUND TRUTH LABELS")
print("="*50)

# Get unique labels and their counts
train_labels = ds['train']['label']
test_labels = ds['test']['label']

print("Train set label distribution:")
unique_train, counts_train = torch.unique(torch.tensor(train_labels), return_counts=True)
for label, count in zip(unique_train, counts_train):
    print(f"  Label {label}: {count} samples")

print("\nTest set label distribution:")
unique_test, counts_test = torch.unique(torch.tensor(test_labels), return_counts=True)
for label, count in zip(unique_test, counts_test):
    print(f"  Label {label}: {count} samples")

# Show some examples of each label
print("\nSample texts for each label:")
for label in unique_test:
    label_samples = [sample for sample in ds['test'] if sample['label'] == label]
    if label_samples:
        print(f"\nLabel {label} example:")
        print(f"  Text: {label_samples[0]['text'][:200]}...")
        print(f"  Length: {len(label_samples[0]['text'])} characters")

# Define label mapping based on dataset analysis
# We'll update this after seeing the actual labels
label_mapping = {
    "0": "GPT",
    "1": "Human",
    "LABEL_0": "Human",
    "LABEL_1": "GPT",
    0: "GPT",
    1: "Human"
}

def classify_text(text, model_config):
    """Classify text using a specific model"""
    try:
        pipe = model_config["pipeline"]
        model_name = model_config["name"]
        
        result = pipe(text)
        
        # Transform the prediction dictionary
        original_label = result[0]['label']
        mapped_label = label_mapping.get(original_label, str(original_label))
        
        return {
            "label": mapped_label,
            "confidence": result[0]['score'],
            "model": model_name
        }
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        return None

def analyze_agreement(dataset, model_configs, confidence_threshold=0.7, max_samples=None):
    """Analyze agreement between models and compare with ground truth"""
    agreements = 0
    total_processed = 0
    disagreements = []
    
    # Track accuracy for each model
    model_accuracies = {config["name"]: {"correct": 0, "total": 0} for config in model_configs}
    
    # Use test set for evaluation
    test_data = dataset['test']
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    print(f"Analyzing {len(test_data)} samples...")
    
    for i, sample in enumerate(test_data):
        if i % 50 == 0:
            print(f"Processing sample {i}/{len(test_data)}")
        
        text = sample['text']
        ground_truth = sample['label']
        results = []
        
        # Get predictions from both models
        for model_config in model_configs:
            result = classify_text(text, model_config)
            if result:
                results.append(result)
        
        # Check if we have results from both models
        if len(results) == 2:
            mage_result = results[0]
            bert_result = results[1]
            
            # Check agreement conditions
            same_label = mage_result['label'] == bert_result['label']
            high_confidence = (mage_result['confidence'] > confidence_threshold and 
                             bert_result['confidence'] > confidence_threshold)
            
            # Check accuracy against ground truth
            # We need to map ground truth to our label format
            ground_truth_mapped = label_mapping.get(ground_truth, str(ground_truth))
            
            for result in results:
                model_name = result['model']
                predicted_label = result['label']
                
                if predicted_label == ground_truth_mapped:
                    model_accuracies[model_name]["correct"] += 1
                model_accuracies[model_name]["total"] += 1
            
            if same_label and high_confidence:
                agreements += 1
            else:
                disagreements.append({
                    'text': text[:100] + "..." if len(text) > 100 else text,
                    'ground_truth': ground_truth_mapped,
                    'mage': mage_result,
                    'bert': bert_result,
                    'same_label': same_label,
                    'high_confidence': high_confidence
                })
            
            total_processed += 1
    
    agreement_rate = agreements / total_processed if total_processed > 0 else 0
    
    # Calculate accuracy for each model
    model_accuracy_results = {}
    for model_name, stats in model_accuracies.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        model_accuracy_results[model_name] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"]
        }
    
    return {
        'agreements': agreements,
        'total_processed': total_processed,
        'agreement_rate': agreement_rate,
        'model_accuracies': model_accuracy_results,
        'disagreements': disagreements[:10]  # Show first 10 disagreements
    }

# Run the analysis
print("\n" + "="*50)
print("ANALYZING MODEL AGREEMENT")
print("="*50)

results = analyze_agreement(ds, model_configs, confidence_threshold=0.7, max_samples=200)

print(f"\nResults:")
print(f"Total samples processed: {results['total_processed']}")
print(f"Agreements (same label + confidence >70%): {results['agreements']}")
print(f"Agreement rate: {results['agreement_rate']:.2%}")

print(f"\nModel Accuracy:")
for model_name, accuracy_data in results['model_accuracies'].items():
    print(f"  {model_name}: {accuracy_data['accuracy']:.2%} ({accuracy_data['correct']}/{accuracy_data['total']})")

if results['disagreements']:
    print(f"\nFirst 10 disagreements:")
    for i, disagreement in enumerate(results['disagreements'], 1):
        print(f"\n{i}. Text: {disagreement['text']}")
        print(f"   Ground Truth: {disagreement['ground_truth']}")
        print(f"   MAGE: {disagreement['mage']['label']} (confidence: {disagreement['mage']['confidence']:.3f})")
        print(f"   BERT: {disagreement['bert']['label']} (confidence: {disagreement['bert']['confidence']:.3f})")
        print(f"   Same label: {disagreement['same_label']}, High confidence: {disagreement['high_confidence']}")

# Original single text classification (commented out for now)
# # Load text from file
# # with open('Data/ModelExamples/gpt.txt', 'r') as file:
# with open('Data/ModelExamples/human.txt', 'r') as file:
# # with open('Data/example_paragraphs/nature_genetics_style.txt', 'r') as file:
# # with open('Data/example_paragraphs/gpt_style.txt', 'r') as file:
#     text = file.read()

# # Perform classification using pipeline
# try:
#     for model_config in model_configs:
#         pipe = model_config["pipeline"]
#         model_name = model_config["name"]
        
#         result = pipe(text)
#         # print(f"Original Pipeline Result: {result}")
        
#         # Transform the prediction dictionary
#         original_label = result[0]['label']
#         mapped_label = label_mapping.get(original_label, str(original_label))
        
#         transformed_result = {
#             "label": mapped_label,  # Map numeric label to name
#             "confidence": result[0]['score'],
#             "model": model_name
#         }
        
#         print(f"Transformed Result: {transformed_result}")
#         # print(f"Predicted Label: {transformed_result['label']}")
#         # print(f"Confidence Score: {transformed_result['confidence']:.4f}")
#         # print(f"Detection Model: {transformed_result['model']}")
    
# except Exception as e:
#     print(f"Error with pipeline: {e}")
