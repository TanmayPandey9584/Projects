import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_data():
    # Load real accounts data
    real_instagram = pd.read_csv('Real/instagram_real_accounts.csv')
    real_facebook = pd.read_csv('Real/facebook_real_accounts.csv')
    real_twitter = pd.read_csv('Real/twitter_real_accounts.csv')
    
    # Load fake accounts data
    fake_instagram = pd.read_csv('Fake/instagram_fake_accounts.csv')
    fake_facebook = pd.read_csv('Fake/facebook_fake_accounts.csv')
    fake_twitter = pd.read_csv('Fake/twitter_fake_accounts.csv')
    
    # Combine all data
    real_data = pd.concat([real_instagram, real_facebook, real_twitter], ignore_index=True)
    fake_data = pd.concat([fake_instagram, fake_facebook, fake_twitter], ignore_index=True)
    
    # Add platform information
    real_data['Platform'] = real_data['URL'].apply(lambda x: 'instagram' if 'instagram' in x else 'facebook' if 'facebook' in x else 'twitter')
    fake_data['Platform'] = fake_data['URL'].apply(lambda x: 'instagram' if 'instagram' in x else 'facebook' if 'facebook' in x else 'twitter')
    
    # Combine real and fake data
    data = pd.concat([real_data, fake_data], ignore_index=True)
    
    return data

def preprocess_data(data):
    # Convert boolean columns to integers
    data['Contains Numbers'] = data['Contains Numbers'].astype(int)
    data['Contains Special Characters'] = data['Contains Special Characters'].astype(int)
    
    # Convert Account Type to numeric using one-hot encoding
    account_type_dummies = pd.get_dummies(data['Account Type'], prefix='Account_Type')
    data = pd.concat([data, account_type_dummies], axis=1)
    
    # Convert Platform to numeric using one-hot encoding
    platform_dummies = pd.get_dummies(data['Platform'], prefix='Platform')
    data = pd.concat([data, platform_dummies], axis=1)
    
    return data

def create_train_test_datasets():
    # Create output directory if it doesn't exist
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    
    # Load and preprocess data
    print("Loading data...")
    data = load_data()
    processed_data = preprocess_data(data)
    
    # Split the data into train and test sets
    print("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42, stratify=processed_data['Label'])
    
    # Save the datasets
    print("Saving datasets...")
    train_data.to_csv('datasets/train_data.csv', index=False)
    test_data.to_csv('datasets/test_data.csv', index=False)
    
    # Print dataset information
    print("\nDataset Information:")
    print(f"Total samples: {len(processed_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    print("\nClass distribution in training set:")
    print(train_data['Label'].value_counts(normalize=True))
    
    print("\nClass distribution in testing set:")
    print(test_data['Label'].value_counts(normalize=True))
    
    print("\nDatasets have been saved to the 'datasets' directory:")
    print("- datasets/train_data.csv")
    print("- datasets/test_data.csv")

if __name__ == "__main__":
    create_train_test_datasets() 