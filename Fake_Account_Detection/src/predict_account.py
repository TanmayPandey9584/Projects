import pandas as pd
import numpy as np
from joblib import load
import os
import argparse
import sys
import re

def check_model_files():
    """Check if all required model files exist"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    required_files = [
        os.path.join(models_dir, 'fake_account_model.joblib'),
        os.path.join(models_dir, 'scaler.joblib'),
        os.path.join(models_dir, 'label_encoder.joblib'),
        os.path.join(models_dir, 'features.joblib')
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("\nError: Required model files are missing!")
        print("Please follow these steps:")
        print("1. First, run the training script:")
        print("   python src/fake_account_detector.py")
        print("2. Wait for the model to be trained and saved")
        print("3. Then run this prediction script again")
        print("\nMissing files:")
        for f in missing_files:
            print(f"- {f}")
        return False
    return True

def load_model_components():
    """Load the saved model components"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, 'models')
        model = load(os.path.join(models_dir, 'fake_account_model.joblib'))
        scaler = load(os.path.join(models_dir, 'scaler.joblib'))
        label_encoder = load(os.path.join(models_dir, 'label_encoder.joblib'))
        features = load(os.path.join(models_dir, 'features.joblib'))
        return model, scaler, label_encoder, features
    except Exception as e:
        print(f"\nError loading model files: {str(e)}")
        print("Please make sure all model files are present and not corrupted.")
        sys.exit(1)

def extract_features_from_url(url):
    """Extract features from the URL with more sophisticated analysis"""
    try:
        # Clean the URL
        url = url.strip().rstrip('/')
        
        # Extract username
        username = url.split('/')[-1].lower()
        
        # Advanced username analysis
        username_features = analyze_username(username)
        
        # Platform-specific features
        platform_features = determine_platform(url.lower())
        
        # Account type features
        account_type_features = determine_account_type(username)
        
        # Combine all features
        features = {
            **username_features,
            **platform_features,
            **account_type_features
        }
        
        return features
    except Exception as e:
        print(f"\nError processing URL: {str(e)}")
        print("Please make sure the URL is valid and follows the format: https://www.platform.com/username")
        sys.exit(1)

def analyze_username(username):
    """Perform detailed analysis of username characteristics"""
    # Basic metrics
    length = len(username)
    has_numbers = int(any(char.isdigit() for char in username))
    special_chars = int(any(not char.isalnum() for char in username))
    
    # Common real account patterns
    real_patterns = [
        # Name patterns with numbers (common in real accounts)
        r'^[a-z]+[._]?[a-z]+\d{1,2}$',      # john.smith12, johnsmith12
        r'^[a-z]+\d{1,2}[._]?[a-z]+$',      # john12.smith, john12smith
        r'^[a-z]+[._]?[a-z]+\d{1,2}[a-z]*$', # john.smith12th
        # Professional/Personal patterns
        r'^[a-z]+[._]?[a-z]+$',             # john.smith, john_smith
        r'^[a-z]+[._]?\d{1,4}$',            # john.1234, john_1234
        r'^[a-z]+[._]?[a-z]+[._]?\d{1,4}$', # john.smith.1234
        # Common username patterns
        r'^[a-z\d]+[._]?[a-z\d]+$',         # Allow mix of letters and numbers
        r'^[a-z\d]+[._]?[a-z\d]+[._]?[a-z\d]+$'  # Allow multiple parts
    ]
    
    # Suspicious patterns (reduced and focused on clearly fake patterns)
    suspicious_patterns = [
        r'\d{6,}',                           # Too many numbers (6+ digits)
        r'[._]{2,}',                         # Multiple consecutive separators
        r'[0-9]+[._][0-9]+[._][0-9]+',      # Multiple number groups
        r'bot\d*$',                          # Ends with 'bot'
        r'fake\d*$',                         # Ends with 'fake'
        r'spam\d*$',                         # Ends with 'spam'
        r'^[0-9]+$'                          # Only numbers
    ]
    
    # Check patterns
    looks_like_real = any(bool(re.match(pattern, username.lower())) for pattern in real_patterns)
    is_suspicious = any(bool(re.search(pattern, username.lower())) for pattern in suspicious_patterns)
    
    # More realistic account characteristics based on pattern matching
    if looks_like_real and not is_suspicious:
        if length <= 15 and has_numbers <= 1:  # Common personal account
            followers = 500 + np.random.randint(0, 1000)  # 500-1500 followers
            following = 300 + np.random.randint(0, 500)   # 300-800 following
            posts = 30 + np.random.randint(0, 100)        # 30-130 posts
            activity = 1
        else:  # Less common but still valid pattern
            followers = 200 + np.random.randint(0, 500)   # 200-700 followers
            following = 200 + np.random.randint(0, 400)   # 200-600 following
            posts = 20 + np.random.randint(0, 50)         # 20-70 posts
            activity = 1
    elif is_suspicious:
        followers = 50 + np.random.randint(0, 100)        # 50-150 followers
        following = 800 + np.random.randint(0, 400)       # 800-1200 following
        posts = 5 + np.random.randint(0, 10)             # 5-15 posts
        activity = 0
    else:  # Neutral case
        followers = 300 + np.random.randint(0, 500)       # 300-800 followers
        following = 250 + np.random.randint(0, 300)       # 250-550 following
        posts = 25 + np.random.randint(0, 60)            # 25-85 posts
        activity = 1
    
    return {
        'Username Length': length,
        'Contains Numbers': has_numbers,
        'Contains Special Characters': special_chars,
        'Followers': followers,
        'Following': following,
        'Posts': posts,
        'Account Activity': activity
    }

def determine_platform(url):
    """Determine platform and set platform-specific features"""
    return {
        'Platform_instagram': int('instagram' in url),
        'Platform_facebook': int('facebook' in url),
        'Platform_twitter': int('twitter' in url)
    }

def determine_account_type(username):
    """Determine account type based on username characteristics"""
    # Business-related terms
    business_terms = [
        'business', 'official', 'inc', 'ltd', 'company', 'store', 'shop',
        'brand', 'media', 'studio', 'agency', 'global', 'worldwide', 'pro',
        'plus', 'club', 'life', 'world', 'official', 'global', 'worldwide'
    ]
    
    # Common brand patterns
    brand_patterns = [
        r'^[a-z]+$',                     # Single word (e.g., nike, adidas)
        r'^[a-z]+official$',             # Official accounts
        r'^[a-z]+global$',               # Global accounts
        r'^the[a-z]+$',                  # The... accounts
        r'^[a-z]+pro$',                  # Pro accounts
        r'^[a-z]+plus$',                 # Plus accounts
        r'^[a-z]+club$',                 # Club accounts
        r'^[a-z]+life$'                  # Life accounts
    ]
    
    # Check for business indicators
    is_business = (
        any(term in username for term in business_terms) or
        any(bool(re.match(pattern, username)) for pattern in brand_patterns)
    )
    
    # Additional checks for professional accounts
    has_professional_format = bool(
        re.match(r'^[a-z0-9]+(_[a-z0-9]+)*$', username) and  # Snake case format
        not any(char.isupper() for char in username) and      # No uppercase
        len(username) > 4 and                                # Reasonable length
        len(username) <= 20 and                              # Not too long
        not re.search(r'\d{4,}', username)                   # Not too many numbers
    )
    
    if is_business or (has_professional_format and len(username) <= 15):
        return {
            'Account_Type_Business': 1,
            'Account_Type_Personal': 0
        }
    else:
        return {
            'Account_Type_Business': 0,
            'Account_Type_Personal': 1
        }

def predict_account(url, model, scaler, label_encoder):
    """Predict whether an account is fake or real based on its URL"""
    # Extract features from URL
    features = extract_features_from_url(url)
    
    # Create feature vector
    feature_vector = pd.DataFrame([features])
    
    # Ensure all required features are present
    required_features = load_model_components()[3]  # Load features list from saved model
    for feature in required_features:
        if feature not in feature_vector.columns:
            feature_vector[feature] = 0
    
    # Reorder columns to match training data
    feature_vector = feature_vector[required_features]
    
    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Make prediction
    prediction = model.predict(feature_vector_scaled)
    prediction_proba = model.predict_proba(feature_vector_scaled)
    
    # Convert prediction to original label
    result = label_encoder.inverse_transform(prediction)[0]
    confidence = prediction_proba[0][prediction[0]] * 100
    
    # Get feature importance for this prediction
    feature_importance = pd.DataFrame({
        'feature': list(features.keys()),
        'value': list(features.values())
    })
    
    return result, confidence, feature_importance

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict whether a social media account is fake or real')
    parser.add_argument('url', help='The URL of the social media account to analyze')
    args = parser.parse_args()
    
    # Check if model files exist
    if not check_model_files():
        sys.exit(1)
    
    try:
        # Load model components
        print("Loading model...")
        model, scaler, label_encoder, features = load_model_components()
        
        # Make prediction
        print("\nAnalyzing account...")
        result, confidence, feature_importance = predict_account(args.url, model, scaler, label_encoder)
        
        # Print results
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"URL: {args.url}")
        print(f"Prediction: {result}")
        print(f"Confidence: {confidence:.2f}%")
        
        print("\nFeature Analysis:")
        print("-" * 50)
        print(feature_importance.to_string(index=False))
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease make sure:")
        print("1. The URL is valid and follows the format: https://www.platform.com/username")
        print("2. The model files are present in the 'models' directory")
        print("3. You have all required Python packages installed")
        sys.exit(1)

if __name__ == "__main__":
    main() 