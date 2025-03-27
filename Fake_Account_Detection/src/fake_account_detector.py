import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import dump, load

def load_data():
    # Load the training data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_data = pd.read_csv(os.path.join(base_dir, 'data', 'training_data.csv'))
    # For testing, we'll use 30% of the training data
    train_data, test_data = train_test_split(train_data, test_size=0.3, random_state=42)
    return train_data, test_data

def preprocess_data(train_data, test_data):
    # Select features for the model
    features = [
        'Username Length', 'Contains Numbers', 'Contains Special Characters',
        'Followers', 'Following', 'Posts', 'Account Activity',
        'Account_Type_Business', 'Account_Type_Personal',
        'Platform_instagram', 'Platform_facebook', 'Platform_twitter'
    ]
    
    X_train = train_data[features]
    X_test = test_data[features]
    
    # Convert labels to numeric using LabelEncoder
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data['Is_Fake'])
    y_test = label_encoder.transform(test_data['Is_Fake'])
    
    return X_train, X_test, y_train, y_test, features, label_encoder

def train_model(X_train, y_train):
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define model with better parameters
    model = RandomForestClassifier(
        n_estimators=2000,        # More trees for better generalization
        max_depth=4,              # Slightly deeper trees
        min_samples_split=30,     # Balanced split criteria
        min_samples_leaf=15,      # Balanced leaf size
        class_weight='balanced',  # Better handling of imbalanced data
        max_features='sqrt',      # Standard RF practice
        random_state=42,
        bootstrap=True,           # Enable bootstrapping
        oob_score=True,          # Enable out-of-bag score
        n_jobs=-1                # Use all CPU cores
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print("\nCross-validation scores:", cv_scores)
    print("Average CV score: {:.2f} (+/- {:.2f})".format(cv_scores.mean(), cv_scores.std() * 2))
    
    # Train the final model
    model.fit(X_train_scaled, y_train)
    return model, scaler

def save_model(model, scaler, label_encoder, features):
    # Create models directory if it doesn't exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save model components
    dump(model, os.path.join(models_dir, 'fake_account_model.joblib'))
    dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    dump(label_encoder, os.path.join(models_dir, 'label_encoder.joblib'))
    dump(features, os.path.join(models_dir, 'features.joblib'))
    
    print("\nModel components saved to 'models' directory:")
    print("- models/fake_account_model.joblib")
    print("- models/scaler.joblib")
    print("- models/label_encoder.joblib")
    print("- models/features.joblib")

def load_saved_model():
    # Load model components
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    model = load(os.path.join(models_dir, 'fake_account_model.joblib'))
    scaler = load(os.path.join(models_dir, 'scaler.joblib'))
    label_encoder = load(os.path.join(models_dir, 'label_encoder.joblib'))
    features = load(os.path.join(models_dir, 'features.joblib'))
    
    return model, scaler, label_encoder, features

def evaluate_model(model, scaler, X_test, y_test, features, label_encoder):
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Convert numeric labels back to original labels for reporting
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_original, y_pred_original))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_original, y_pred_original)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.savefig(os.path.join(base_dir, 'output', 'confusion_matrix.png'))
    plt.close()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(base_dir, 'output', 'roc_curve.png'))
    plt.close()
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'output', 'feature_importance.png'))
    plt.close()
    
    # Print feature importance
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Print detailed metrics
    print("\nDetailed Metrics:")
    print(f"ROC AUC Score: {roc_auc:.3f}")
    
    # Calculate and print prediction probabilities distribution
    print("\nPrediction Probability Distribution:")
    proba_bins = pd.cut(y_pred_proba.max(axis=1), bins=10)
    print(pd.value_counts(proba_bins, normalize=True).sort_index())

def predict_account(url, model, scaler, label_encoder):
    # Extract features from URL
    features = {
        'Username Length': len(url.split('/')[-1]),
        'Contains Numbers': int(any(char.isdigit() for char in url.split('/')[-1])),
        'Contains Special Characters': int(any(not char.isalnum() for char in url.split('/')[-1])),
        'Followers': 0,  # These would need to be fetched from the actual account
        'Following': 0,
        'Posts': 0,
        'Account Activity': 0,
        'Account_Type_Business': 0,
        'Account_Type_Personal': 0,
        'Platform_instagram': int('instagram' in url),
        'Platform_facebook': int('facebook' in url),
        'Platform_twitter': int('twitter' in url)
    }
    
    # Create feature vector
    feature_vector = pd.DataFrame([features])
    
    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Make prediction
    prediction = model.predict(feature_vector_scaled)
    prediction_proba = model.predict_proba(feature_vector_scaled)
    
    # Convert prediction to original label
    result = label_encoder.inverse_transform(prediction)[0]
    confidence = prediction_proba[0][prediction[0]] * 100
    
    return result, confidence

def main():
    # Check if model already exists
    if os.path.exists('../models/fake_account_model.joblib'):
        print("Loading existing model...")
        model, scaler, label_encoder, features = load_saved_model()
        
        # Load test data for evaluation
        _, test_data = load_data()
        X_test = test_data[features]
        y_test = label_encoder.transform(test_data['Is_Fake'])
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluate_model(model, scaler, X_test, y_test, features, label_encoder)
    else:
        # Load and preprocess data
        print("Loading data...")
        train_data, test_data = load_data()
        X_train, X_test, y_train, y_test, features, label_encoder = preprocess_data(train_data, test_data)
        
        # Train model
        print("\nTraining model...")
        model, scaler = train_model(X_train, y_train)
        
        # Save model
        save_model(model, scaler, label_encoder, features)
        
        # Evaluate model
        print("\nEvaluating model...")
        evaluate_model(model, scaler, X_test, y_test, features, label_encoder)
    

    
    print("\nModel training and evaluation complete!")

if __name__ == "__main__":
    main()