import pickle
import pandas as pd

# =============================================================================
# 1. Load the Saved Model
# =============================================================================
with open("Hardness_Prediction_Model.pkl", "rb") as file:
    model = pickle.load(file)

print("✅ Model loaded successfully!")

# =============================================================================
# 2. Load and Prepare the Test Data
# =============================================================================
test_data = pd.read_csv("test_data.csv", encoding="ISO-8859-1")

# Extract the feature names from the trained model
selected_features = model.best_estimator_.get_booster().feature_names

# Ensure test data has only the selected features
x_test = test_data[selected_features]

print("✅ Test data prepared successfully!")

# =============================================================================
# 3. Make Predictions
# =============================================================================
y_pred = model.predict(x_test)

# =============================================================================
# 4. Save Predictions for Analysis
# =============================================================================
predictions_df = pd.DataFrame({"Predicted Hardness (HVN)": y_pred})
predictions_df.to_csv("Predicted_Hardness.csv", index=False)

print("✅ Predictions saved successfully in 'Predicted_Hardness.csv'!")
