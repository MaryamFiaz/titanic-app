import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Titanic ML Analytics App", page_icon="🚢", layout="wide")

st.title("🚢 Advanced Titanic Survival Prediction & Analytics")
st.write("Predict passenger survival and explore dataset insights using machine learning models.")

# Tabs for Organization
tab1, tab2 = st.tabs(["🔮 Model Prediction", "📊 Exploratory Data Analysis"])

with tab1:
    st.header("Passenger Details Input")
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else f"{x}nd Class" if x==2 else "3rd Class")
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 25)
        fare = st.number_input("Ticket Fare ($)", min_value=0.0, value=32.2)

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
        parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
        embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

    st.markdown("---")
    
    # Model Selection Option
    model_choice = st.radio("Choose Model for Prediction:", ["Decision Tree (Default)", "Random Forest Simulation"])

    if st.button("Predict Survival"):
        # Preprocessing input
        sex_val = 1 if sex == "female" else 0
        emb_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
        emb_val = emb_map[embarked]

        input_data = [[pclass, sex_val, age, sibsp, parch, fare, emb_val]]

        try:
            # Load model
            with open("titanic_decision_tree_model.pkl", "rb") as f:
                model = pickle.load(f)
            
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1] * 100 if hasattr(model, "predict_proba") else None

            if prediction == 1:
                st.success(f"🎉 **Passenger Survived!** (Survival Probability: {prob:.1f}%)" if prob else "🎉 Passenger Survived!")
            else:
                st.error(f"⚠️ **Passenger Did Not Survive.** (Survival Probability: {prob:.1f}%)" if prob else "⚠️ Passenger Did Not Survive.")
        except Exception as e:
            st.error(f"Error loading model: {e}")

with tab2:
    st.header("Visual Analytics & Insights")
    # Sample visualization data
    df_sample = pd.DataFrame({
        "Sex": ["female", "male", "female", "male"],
        "Survived": ["Yes", "No", "Yes", "Yes"],
        "Count": [233, 109, 81, 46]
    })
    
    fig = px.bar(df_sample, x="Sex", y="Count", color="Survived", title="Survival Rate by Gender", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
