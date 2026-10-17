import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Titanic ML Analytics App",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .status-survived {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 18px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    .status-died {
        background: linear-gradient(135deg, #CB2D3E 0%, #EF473A 100%);
        padding: 18px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚢 Titanic AI Survival Predictor & Analytics")
st.caption("Interactive Machine Learning Portfolio App | Powered by Scikit-Learn")

# Sidebar Configuration
st.sidebar.header("⚙️ Passenger Inputs")

pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else f"{x}nd Class" if x==2 else "3rd Class")
sex = st.sidebar.radio("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age (Years)", 0, 80, 25)
fare = st.sidebar.number_input("Ticket Fare ($)", min_value=0.0, value=32.2)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

tab1, tab2 = st.tabs(["🔮 AI Survival Prediction", "📊 Dataset Analytics"])

def load_model():
    model_filename = "titanic_decision_tree_model.pkl"
    try:
        return joblib.load(model_filename)
    except Exception:
        with open(model_filename, "rb") as f:
            return pickle.load(f)

with tab1:
    st.subheader("Prediction Dashboard")
    
    if st.button("🚀 Predict Survival Status", use_container_width=True):
        sex_val = 1 if sex.lower() == "female" else 0
        emb_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}
        emb_val = emb_map[embarked]
        
        # Calculate Family Size as the 8th feature
        family_size = sibsp + parch + 1

        # 8 features total matching the trained model
        input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, emb_val, family_size]])

        try:
            model = load_model()
            prediction = model.predict(input_data)[0]
            
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_data)[0][1] * 100

            col1, col2 = st.columns(2)

            with col1:
                if prob is not None:
                    st.metric(label="Calculated Survival Probability", value=f"{prob:.1f}%")
                else:
                    st.metric(label="Prediction Result", value="Calculated")

            with col2:
                if prediction == 1:
                    st.markdown('<div class="status-survived">🎉 Result: Passenger Survived</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-died">⚠️ Result: Did Not Survive</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error loading prediction model: {e}")

with tab2:
    st.subheader("Exploratory Visual Analytics")
    
    df_sample = pd.DataFrame({
        "Sex": ["Female", "Male", "Female", "Male"],
        "Status": ["Survived", "Did Not Survive", "Survived", "Survived"],
        "Count": [233, 109, 81, 46]
    })
    
    fig = px.bar(
        df_sample, 
        x="Sex", 
        y="Count", 
        color="Status", 
        title="Survival Rate Distribution by Gender",
        barmode="group",
        color_discrete_map={"Survived": "#2ecc71", "Did Not Survive": "#e74c3c"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
