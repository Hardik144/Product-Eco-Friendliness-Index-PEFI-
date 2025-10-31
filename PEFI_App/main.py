import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPRegressor # Import MLPRegressor

"""
PEFI Prediction Tool
====================
Use this script to predict the Eco-Friendliness Index for new products
"""

# Load saved models and encoders
print("🔧 Loading trained models and encoders...")

try:
    with open('best_pefi_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('pefi_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    print("✅ Models loaded successfully!\n")
except FileNotFoundError:
    print("❌ Model files not found. Please run the training pipeline first!")
    # Note: In a real application, you might want to handle this more gracefully
    #       than just exiting, but for this script it's fine.
    exit()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_eco_friendliness(product_data):
    """
    Predict eco-friendliness index for a product

    Parameters:
    -----------
    product_data : dict
        Dictionary containing product features

    Returns:
    --------
    dict : Prediction results including score and category
    """
    # Create DataFrame from input
    df = pd.DataFrame([product_data])

    # Drop 'Product_Name' column if it exists, as the model was trained without it
    if 'Product_Name' in df.columns:
        df = df.drop('Product_Name', axis=1)

    # Ensure all expected columns are present, even if with default/placeholder values
    # This is important if the input data doesn't have all columns the model was trained on
    # For this specific dataset, we expect certain columns based on the notebook.
    # You might need a more robust check for real-world applications.

    # Encode categorical variables
    for col in label_encoders.keys():
        if col in df.columns:
            try:
                df[col] = label_encoders[col].transform(df[col])
            except ValueError as e:
                print(f"⚠️ Warning: Unknown value in '{col}'. Using default encoding.")
                # If unknown category, use the most common encoding (0)
                df[col] = 0 # Or potentially handle with a specific 'unknown' value if trained

    # Make prediction
    # Scale numerical features if the best model is Neural Network
    # Check if the loaded model is an instance of MLPRegressor
    if isinstance(model, MLPRegressor):
        # Ensure numerical columns are correctly identified and scaled
        # This assumes the order and number of columns match the training data
        # A safer approach would be to use a Pipeline saved from training
        # For this script, we'll rely on the scaler expecting the same column order/structure as X_train

        # Identify numerical columns based on the original X used for training (if available)
        # Or based on expected columns minus categorical ones
        # Assuming the order of columns in df is the same as X_train after encoding/dropping
        # This is a potential point of failure if input column order changes
        # A robust solution would use a saved list of numerical columns or a Pipeline
        # For now, we'll proceed assuming the simple case matches the notebook

        # --- A more robust approach (requires saving more info from training) ---
        # numerical_cols_saved = ['Transport_Distance_km', 'Carbon_Footprint_kgCO2', ...] # Need to save this list
        # df_numerical = df[numerical_cols_saved]
        # df_categorical_encoded = df.drop(numerical_cols_saved, axis=1)
        # df_scaled_numerical = scaler.transform(df_numerical)
        # df_scaled = pd.concat([pd.DataFrame(df_scaled_numerical, columns=numerical_cols_saved), df_categorical_encoded], axis=1)
        # # Need to ensure column order is correct before prediction

        # --- Simplified approach based on script's current structure ---
        # Assuming df now contains only the columns X was trained on, in the same order
        # and categorical columns are already encoded.
        # We need to apply scaler only to the numerical columns.
        # This is tricky without knowing which columns are numerical after encoding.
        # A safer way is to save the list of numerical columns from training.

        # Let's assume for this script's structure that the scaler was fitted on the *entire* X_train after encoding
        # This is not ideal but matches how the training script used it for the NN.
        # If your training scaled only numerical columns, you'd need to adjust this.
        # *** Relying on the structure from the training script (cell 5qQC3tOIxt8u) ***
        # The training script scaled the entire X_train after LabelEncoding, which is incorrect for non-NN models
        # but is how the NN was trained. For the NN prediction here, we must scale the whole dataframe.
        # For other models (RF, XGBoost), scaling the entire dataframe (including encoded categoricals) is generally not needed and can hurt performance.
        # We need to apply scaling conditionally *and correctly* based on the model type and how it was trained.

        # The original training script scaled X_train *after* encoding categorical columns.
        # This means the scaler expects the encoded categorical columns + numerical columns.
        # When predicting with the NN, we need to scale the input df *after* encoding it.

        # Identify numerical columns based on dtypes after encoding
        numerical_cols_after_encoding = df.select_dtypes(include=np.number).columns.tolist()

        # Create a copy to scale
        df_to_scale = df[numerical_cols_after_encoding].copy() # Only select columns that should be numeric

        # Apply scaler
        # The scaler was fit on X_train (which included encoded categoricals and numericals)
        # This means the order of columns matters significantly here.
        # The safest way is to get the column order from the trained model or training data.
        # Assuming df's column order matches X_train after dropping Product_Name and encoding:
        try:
             # Transform the dataframe. This requires the columns to be in the same order as X_train
            df_scaled_values = scaler.transform(df.values)
            df_scaled = pd.DataFrame(df_scaled_values, columns=df.columns) # Recreate DataFrame with scaled values
            eco_score = model.predict(df_scaled)[0]
        except Exception as e:
             print(f"❌ Error during scaling or NN prediction: {e}")
             # Fallback or re-raise
             raise e # Re-raise the exception after printing


    else:
        # For models like RandomForest or XGBoost, scaling the entire encoded dataframe is usually not needed.
        # Predict directly on the encoded dataframe.
        try:
            eco_score = model.predict(df)[0]
        except Exception as e:
            print(f"❌ Error during model prediction: {e}")
            # Fallback or re-raise
            raise e # Re-raise the exception after printing


    # Categorize
    if eco_score <= 40:
        category = 'Poor'
        emoji = '🔴'
    elif eco_score <= 60:
        category = 'Fair'
        emoji = '🟡'
    elif eco_score <= 80:
        category = 'Good'
        emoji = '🟢'
    else:
        category = 'Excellent'
        emoji = '🌟'

    # Get category prediction from classifier
    # Check if the classifier needs scaling (e.g., if it's an MLPClassifier)
    # The training script used RandomForestClassifier, which doesn't need scaling.
    # If you change the classifier, adjust this.
    # For now, assume the classifier doesn't need scaling based on the training script.
    try:
        # Ensure the input to the classifier has the correct columns and encoding
        # The classifier was trained on X_train (encoded categoricals + numericals)
        # Use the same df that was prepared for the regression model prediction
        category_pred = classifier.predict(df)[0]
    except Exception as e:
        print(f"❌ Error during classifier prediction: {e}")
        # Fallback or re-raise
        raise e # Re-raise the exception after printing


    return {
        'eco_score': round(eco_score, 2),
        'category': category,
        'emoji': emoji,
        'classifier_prediction': category_pred
    }

# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================

print("="*70)
print("🔮 EXAMPLE PREDICTIONS")
print("="*70)

# Example 1: Eco-friendly product (Cotton clothing from India)
example1 = {
    'Category': 'Clothing',
    'Material_Type': 'Cotton',
    'Packaging_Type': 'Biodegradable Paper',
    'Energy_Consumption_Manufacturing': 'Low',
    'Source_Country': 'India',
    'Transport_Distance_km': 200,
    'Carbon_Footprint_kgCO2': 5.5,
    'Recyclability_Percent': 90,
    'Durability_Years': 3.0,
    'Eco_Certification': 'GOTS'
}

result1 = predict_eco_friendliness(example1)
print(f"\n🌱 Example 1: Fabindia Organic Cotton Kurta")
print(f"   Material: Cotton | Packaging: Biodegradable")
print(f"   Source: India (200 km)")
print(f"   {result1['emoji']} Eco-Friendliness Score: {result1['eco_score']}/100")
print(f"   Category: {result1['category']}")
print(f"   Classifier Predicted Category: {result1['classifier_prediction']}")


# Example 2: Less eco-friendly product (Imported electronics)
example2 = {
    'Category': 'Electronics',
    'Material_Type': 'Metal+Plastic',
    'Packaging_Type': 'Multi-layer Plastic',
    'Energy_Consumption_Manufacturing': 'Very High',
    'Source_Country': 'China',
    'Transport_Distance_km': 4500,
    'Carbon_Footprint_kgCO2': 75.0,
    'Recyclability_Percent': 30,
    'Durability_Years': 3.0,
    'Eco_Certification': 'None'
}

result2 = predict_eco_friendliness(example2)
print(f"\n📱 Example 2: Imported Smartphone")
print(f"   Material: Metal+Plastic | Packaging: Multi-layer Plastic")
print(f"   Source: China (4500 km)")
print(f"   {result2['emoji']} Eco-Friendliness Score: {result2['eco_score']}/100")
print(f"   Category: {result2['category']}")
print(f"   Classifier Predicted Category: {result2['classifier_prediction']}")


# Example 3: Moderately eco-friendly (Indian FMCG)
example3 = {
    'Category': 'FMCG_Food',
    'Material_Type': 'Glass',
    'Packaging_Type': 'Recyclable Glass',
    'Energy_Consumption_Manufacturing': 'Low',
    'Source_Country': 'India',
    'Transport_Distance_km': 350,
    'Carbon_Footprint_kgCO2': 3.2,
    'Recyclability_Percent': 95,
    'Durability_Years': 1.0,
    'Eco_Certification': 'FSSAI'
}

result3 = predict_eco_friendliness(example3)
print(f"\n🍯 Example 3: Dabur Honey (Glass Bottle)")
print(f"   Material: Glass | Packaging: Recyclable Glass")
print(f"   Source: India (350 km)")
print(f"   {result3['emoji']} Eco-Friendliness Score: {result3['eco_score']}/100")
print(f"   Category: {result3['category']}")
print(f"   Classifier Predicted Category: {result3['classifier_prediction']}")


# Example 4: Poor eco-friendliness (Plastic heavy product)
example4 = {
    'Category': 'Homecare',
    'Material_Type': 'Plastic',
    'Packaging_Type': 'Multi-layer Plastic',
    'Energy_Consumption_Manufacturing': 'High',
    'Source_Country': 'China',
    'Transport_Distance_km': 4000,
    'Carbon_Footprint_kgCO2': 8.5,
    'Recyclability_Percent': 15,
    'Durability_Years': 0.5,
    'Eco_Certification': 'None'
}

result4 = predict_eco_friendliness(example4)
print(f"\n🧴 Example 4: Imported Plastic Cleaning Product")
print(f"   Material: Plastic | Packaging: Multi-layer Plastic")
print(f"   Source: China (4000 km)")
print(f"   {result4['emoji']} Eco-Friendliness Score: {result4['eco_score']}/100")
print(f"   Category: {result4['category']}")
print(f"   Classifier Predicted Category: {result4['classifier_prediction']}")


# ============================================================================
# INTERACTIVE PREDICTION (CUSTOM INPUT)
# ============================================================================

print("\n" + "="*70)
print("🎯 MAKE YOUR OWN PREDICTION")
print("="*70)

def interactive_prediction():
    """
    Interactive function to get user input and make predictions
    """
    print("\nEnter product details (or press Enter to skip interactive mode):\n")

    try:
        # Get user input
        category = input("Category (FMCG_Food/Electronics/Clothing/Personal_Care/Homecare/Kitchenware/Stationery/Sports): ").strip()
        if not category:
            print("Skipping interactive mode...")
            return

        material = input("Material Type (Plastic/Paper/Metal/Glass/Cotton/Aluminum/Metal+Plastic/etc.): ").strip()
        packaging = input("Packaging Type (Recyclable Cardboard/Recyclable Plastic/Multi-layer Plastic/etc.): ").strip()
        energy = input("Energy Consumption (Very Low/Low/Medium/High/Very High): ").strip()
        source = input("Source Country (India/China/Vietnam/USA/etc.): ").strip()
        distance = int(input("Transport Distance (km): ").strip())
        carbon = float(input("Carbon Footprint (kg CO2): ").strip())
        recyclability = int(input("Recyclability Percent (0-100): ").strip())
        durability = float(input("Durability (years): ").strip())
        certification = input("Eco Certification (None/BIS/FSSAI/Energy Star/GOTS/etc.): ").strip()

        # Create product data dictionary
        custom_product = {
            'Category': category,
            'Material_Type': material,
            'Packaging_Type': packaging,
            'Energy_Consumption_Manufacturing': energy,
            'Source_Country': source,
            'Transport_Distance_km': distance,
            'Carbon_Footprint_kgCO2': carbon,
            'Recyclability_Percent': recyclability,
            'Durability_Years': durability,
            'Eco_Certification': certification
        }

        # Make prediction
        result = predict_eco_friendliness(custom_product)

        print("\n" + "="*70)
        print("📊 PREDICTION RESULTS")
        print("="*70)
        print(f"\n{result['emoji']} Eco-Friendliness Score: {result['eco_score']}/100")
        print(f"Category: {result['category']}")
        print(f"Classifier Prediction: {result['classifier_prediction']}")

        # Recommendations
        print(f"\n💡 Sustainability Recommendations:")
        if result['eco_score'] < 40:
            print("   🔴 This product has low eco-friendliness. Consider:")
            print("   - Using more recyclable materials")
            print("   - Reducing packaging waste")
            print("   - Sourcing locally to reduce transportation")
            print("   - Improving product durability")
        elif result['eco_score'] < 60:
            print("   🟡 This product has moderate eco-friendliness. To improve:")
            print("   - Consider biodegradable packaging options")
            print("   - Explore renewable energy in manufacturing")
            print("   - Obtain eco-certifications")
        elif result['eco_score'] < 80:
            print("   🟢 This product has good eco-friendliness. Minor improvements:")
            print("   - Further optimize transportation routes")
            print("   - Explore carbon offset programs")
            print("   - Consider circular economy principles")
        else:
            print("   🌟 Excellent! This product is highly eco-friendly!")
            print("   - Maintain current sustainable practices")
            print("   - Share your sustainability story with consumers")
            ("   - Consider pursuing premium eco-certifications")

    except (ValueError, KeyboardInterrupt):
        print("\n❌ Invalid input or operation cancelled.")
        return

# Uncomment the line below to enable interactive mode
# interactive_prediction()

# ============================================================================
# BATCH PREDICTION FROM CSV
# ============================================================================

def predict_from_csv(input_file, output_file='predictions.csv'):
    """
    Make predictions for multiple products from a CSV file

    Parameters:
    -----------
    input_file : str
        Path to input CSV file with product data
    output_file : str
        Path to save predictions
    """
    print(f"\n📂 Reading products from: {input_file}")

    try:
        # Read input file
        df = pd.read_csv(input_file)
        print(f"✅ Loaded {len(df)} products")

        # Make predictions
        predictions = []
        categories = []
        classifier_predictions = [] # Capture classifier predictions

        for idx, row in df.iterrows():
            product_dict = row.to_dict()
            result = predict_eco_friendliness(product_dict)
            predictions.append(result['eco_score'])
            categories.append(result['category'])
            classifier_predictions.append(result['classifier_prediction']) # Store classifier prediction

        # Add predictions to dataframe
        df['Predicted_Eco_Score'] = predictions
        df['Predicted_Category_Regression'] = categories # Rename for clarity
        df['Predicted_Category_Classification'] = classifier_predictions # Add classifier prediction

        # Save results
        df.to_csv(output_file, index=False)
        print(f"✅ Predictions saved to: {output_file}")

        # Show summary
        print(f"\n📊 Prediction Summary:")
        print(f"   - Average Eco-Score (Regression): {np.mean(predictions):.2f}")
        print(f"   - Category Distribution (Regression):")
        category_dist_reg = pd.Series(categories).value_counts()
        for cat, count in category_dist_reg.items():
            print(f"     • {cat}: {count} ({count/len(categories)*100:.1f}%)")

        print(f"\n   - Category Distribution (Classification):")
        category_dist_clf = pd.Series(classifier_predictions).value_counts()
        for cat, count in category_dist_clf.items():
            print(f"     • {cat}: {count} ({count/len(classifier_predictions)*100:.1f}%)")


        return df

    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

# Example: Predict from CSV (uncomment to use)
# predict_from_csv('new_products.csv', 'eco_predictions.csv')

# ============================================================================
# COMPARISON TOOL
# ============================================================================

def compare_products(products_list):
    """
    Compare eco-friendliness of multiple products

    Parameters:
    -----------
    products_list : list of dict
        List of product data dictionaries
    """
    print("\n" + "="*70)
    print("🔄 PRODUCT COMPARISON")
    print("="*70)

    results = []
    for i, product in enumerate(products_list, 1):
        product_name = product.get('Product_Name', f'Product {i}') # Get product name if exists
        result = predict_eco_friendliness(product)
        results.append({
            'Product': i,
            'Name': product_name, # Use fetched name
            'Score': result['eco_score'],
            'Category_Reg': result['category'],
            'Category_Clf': result['classifier_prediction'], # Add classifier category
            'Emoji': result['emoji']
        })

    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['Score'], reverse=True)

    print(f"\n🏆 Ranking (Best to Worst):\n")
    for rank, item in enumerate(results_sorted, 1):
        print(f"   {rank}. {item['Emoji']} {item['Name']}")
        print(f"      Score: {item['Score']}/100")
        print(f"      Category (Regression): {item['Category_Reg']}")
        print(f"      Category (Classification): {item['Category_Clf']}\n")


    return results_sorted

# Example comparison
print("\n" + "="*70)
print("🔄 PRODUCT COMPARISON EXAMPLE")
print("="*70)

products_to_compare = [
    {**example1, 'Product_Name': 'Fabindia Cotton Kurta'},
    {**example2, 'Product_Name': 'Imported Smartphone'},
    {**example3, 'Product_Name': 'Dabur Honey'},
    {**example4, 'Product_Name': 'Plastic Cleaner'}
]

comparison_results = compare_products(products_to_compare)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_improvement_suggestions(product_data, current_score):
    """
    Suggest improvements to increase eco-friendliness score
    """
    suggestions = []

    # Check recyclability
    if product_data['Recyclability_Percent'] < 50:
        suggestions.append("✓ Improve recyclability by using mono-material packaging")

    # Check packaging
    if 'Multi-layer' in product_data['Packaging_Type']:
        suggestions.append("✓ Switch to recyclable or biodegradable packaging")

    # Check source
    if product_data['Source_Country'] != 'India':
        suggestions.append("✓ Consider local sourcing to reduce transportation impact")

    # Check energy
    if product_data['Energy_Consumption_Manufacturing'] in ['High', 'Very High']:
        suggestions.append("✓ Explore renewable energy options for manufacturing")

    # Check certification
    if product_data['Eco_Certification'] == 'None':
        suggestions.append("✓ Obtain eco-certifications (GOTS, FSC, Ecomark)")

    # Check material
    if product_data['Material_Type'] == 'Plastic':
        suggestions.append("✓ Explore alternative materials (bamboo, recycled plastic, paper)")

    # Estimate potential improvement
    # This is a simplified estimation, not based on model re-prediction
    potential_score = current_score + len(suggestions) * 8
    potential_score = min(95, potential_score)

    return suggestions, potential_score

# Example improvement analysis
print("\n" + "="*70)
print("📈 IMPROVEMENT ANALYSIS")
print("="*70)

print(f"\n🔍 Analyzing: Imported Smartphone")
print(f"Current Score: {result2['eco_score']}/100 ({result2['category']})")

suggestions, potential = get_improvement_suggestions(example2, result2['eco_score'])

print(f"\n💡 Suggested Improvements:")
for suggestion in suggestions:
    print(f"   {suggestion}")

print(f"\n🎯 Potential Score with Improvements: {potential}/100")
print(f"   Improvement: +{potential - result2['eco_score']:.1f} points")

# ============================================================================
# FINAL NOTES
# ============================================================================

print("\n" + "="*70)
print("✨ PEFI PREDICTION TOOL READY!")
print("="*70)

print("""
📝 HOW TO USE THIS TOOL:

1. SINGLE PREDICTION:
   result = predict_eco_friendliness(product_data)

2. BATCH PREDICTIONS FROM CSV:
   predict_from_csv('input.csv', 'output.csv')

3. COMPARE PRODUCTS:
   compare_products([product1, product2, product3])

4. INTERACTIVE MODE:
   Uncomment: interactive_prediction()

5. IMPROVEMENT SUGGESTIONS:
   suggestions, potential = get_improvement_suggestions(product_data, current_score)

📊 Available Categories:
   FMCG_Food, Electronics, Clothing, Personal_Care,
   Homecare, Kitchenware, Stationery, Sports

🎯 Eco-Rating Scale:
   🔴 Poor (0-40) | 🟡 Fair (41-60) | 🟢 Good (61-80) | 🌟 Excellent (81-100)
""")