import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import os

def load_test_data():
    """Load the test dataset"""
    test_data = pd.read_csv('datasets/test_data.csv')
    return test_data

def load_model_components():
    """Load the saved model components"""
    model = load('models/fake_account_model.joblib')
    scaler = load('models/scaler.joblib')
    label_encoder = load('models/label_encoder.joblib')
    features = load('models/features.joblib')
    return model, scaler, label_encoder, features

def evaluate_model_performance():
    # Check if model exists
    if not os.path.exists('models/fake_account_model.joblib'):
        print("Error: Model not found. Please train the model first using fake_account_detector.py")
        return
    
    print("Loading model and test data...")
    
    # Load model components and test data
    model, scaler, label_encoder, features = load_model_components()
    test_data = load_test_data()
    
    # Prepare test data
    X_test = test_data[features]
    y_test = label_encoder.transform(test_data['Label'])
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Convert predictions back to original labels
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print detailed metrics
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(y_test_original, y_pred_original))
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_test_original, y_pred_original)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation_confusion_matrix.png')
    plt.close()
    
    # Calculate and print class-wise accuracy
    print("\nClass-wise Accuracy:")
    print("-" * 50)
    for label in label_encoder.classes_:
        mask = y_test_original == label
        class_accuracy = accuracy_score(y_test[mask], y_pred[mask])
        print(f"{label}: {class_accuracy:.4f}")
    
    # Print total counts
    print("\nTotal Counts:")
    print("-" * 50)
    print(f"Total test samples: {len(y_test)}")
    print(f"Correct predictions: {sum(y_test == y_pred)}")
    print(f"Incorrect predictions: {sum(y_test != y_pred)}")
    
    print("\nEvaluation complete! Results have been saved to 'evaluation_confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model_performance() 