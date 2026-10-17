import streamlit as st
import pandas as pd
import joblib

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('titanic_decision_tree_model.pkl')

model = load_model()

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details below to predict their survival probability based on our Decision Tree model.")

# Create input widgets for user features
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else (f"{x}nd Class" if x==2 else f"{x}rd Class"))
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0.1, 80.0, 25.0)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.2)

with col2:
    sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["S", "C", "Q"], format_func=lambda x: "Southampton (S)" if x=="S" else ("Cherbourg (C)" if x=="C" else "Queenstown (Q)"))

# Predict button
if st.button("Predict Survival"):
    # Prepare the input DataFrame matching the exact feature structure used during training
    # Features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    # One-hot encode categorical variables identically to training (drop_first=True)
    # Note: 'Sex' becomes Sex_male or Sex_female depending on encoding; let's handle via pandas get_dummies safely
    # For a robust match, we recreate the same dummy columns expected by model
    input_encoded = pd.get_dummies(input_data, drop_first=True)
    
    # Align columns with the model's training features (adds missing columns as 0 if any category is missing)
    # The DecisionTree expects specific columns after get_dummies. Let's ensure standard alignment:
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[expected_cols]

    # Make prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"🎉 **Survived!** (Estimated Survival Probability: {probability[1]*100:.2f}%)")
    else:
        st.error(f"⚠️ **Deceased.** (Estimated Survival Probability: {probability[1]*100:.2f}%)")