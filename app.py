"""
CardioX - Heart Disease Prediction System
A comprehensive Streamlit application for cardiovascular risk assessment using Machine Learning
Author: AI-Powered Medical Screening Tool
Version: 1.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="CardioX - Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.02);
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF4B4B;
        color: #262730;
    }
    .feature-box h4 {
        color: #FF4B4B;
        margin-top: 0;
    }
    .feature-box p {
        color: #31333F;
        line-height: 1.6;
    }
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        font-size: 24px;
        font-weight: bold;
    }
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF4B4B;
        color: #262730;
    }
    .info-box h4 {
        color: #FF4B4B;
        margin-top: 0;
    }
    .info-box ul {
        color: #31333F;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== LOAD MODEL FUNCTION ====================
@st.cache_resource
def load_model_files():
    """Load the trained model, scaler, and feature columns"""
    try:
        # Try multiple possible paths
        possible_paths = [
            './',
            '../',
            '/mnt/user-data/uploads/'
        ]
        
        for path in possible_paths:
            try:
                # Try loading with pickle first
                try:
                    with open(path + 'KNN_heart.pkl', 'rb') as f:
                        model = pickle.load(f)
                except:
                    # Try with joblib
                    import joblib
                    model = joblib.load(path + 'KNN_heart.pkl')
                
                # Load scaler
                try:
                    with open(path + 'scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                except:
                    import joblib
                    scaler = joblib.load(path + 'scaler.pkl')
                
                # Load columns
                try:
                    with open(path + 'columns.pkl', 'rb') as f:
                        columns = pickle.load(f)
                except:
                    import joblib
                    columns = joblib.load(path + 'columns.pkl')
                
                return model, scaler, columns
            except (FileNotFoundError, Exception):
                continue
        
        # If files not found, return None
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None

@st.cache_resource
def create_demo_model():
    """Create a demo model for testing purposes"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Feature columns in the correct order
    columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
               'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
               'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 
               'ST_Slope_Flat', 'ST_Slope_Up']
    
    # Create dummy training data (minimal example)
    X_train = np.array([
        [65, 160, 280, 1, 110, 2.5, 1, 0, 0, 1, 0, 1, 1, 1, 0],
        [45, 120, 180, 0, 170, 0.0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [58, 140, 240, 1, 130, 1.5, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        [35, 110, 160, 0, 180, 0.0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    ])
    y_train = np.array([1, 0, 1, 0])
    
    # Train scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = KNeighborsClassifier()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, columns

# ==================== SIDEBAR NAVIGATION ====================
st.sidebar.title("❤️ CardioX")
st.sidebar.markdown("**AI-Powered Heart Disease Prediction**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🔮 Predict", "ℹ️ About"]
)
st.sidebar.markdown("---")
st.sidebar.info("""
**Quick Tips:**
- Start with the Home page to learn about the app
- Use the Predict page for risk assessment
- Check the About page for technical details
""")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>❤️ CardioX</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #666;'>AI-Powered Heart Disease Prediction System</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Introduction
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🩺 Welcome to CardioX
        
        CardioX is an advanced **Machine Learning** powered application that predicts the likelihood of heart disease 
        based on various health parameters. Our model has been trained on a comprehensive dataset of cardiovascular 
        health indicators and achieves high accuracy in risk assessment.
        
        ### 🎯 Why Use This Tool?
        
        Cardiovascular diseases are the **leading cause of death globally**, accounting for millions of deaths each year. 
        Early detection and risk assessment can significantly improve treatment outcomes and save lives. This tool 
        helps you:
        
        - **Assess cardiovascular risk** quickly and accurately
        - **Understand key health indicators** related to heart disease
        - **Make informed decisions** about seeking medical consultation
        - **Track health parameters** that matter for heart health
        """)
    
    with col2:
        st.image("https://img.icons8.com/3d-fluency/400/heart-with-pulse.png", width=300)
    
    st.markdown("---")
    
    # Key Features
    st.markdown("## 🌟 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-box'>
        <h4>🤖 AI-Powered Analysis</h4>
        <p>Uses K-Nearest Neighbors (KNN) algorithm with 88.6% accuracy to predict heart disease risk.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-box'>
        <h4>📊 Comprehensive Assessment</h4>
        <p>Analyzes 15+ health parameters including age, blood pressure, cholesterol, and ECG results.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-box'>
        <h4>⚡ Instant Results</h4>
        <p>Get immediate risk assessment with confidence scores and actionable insights.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How to Use
    st.markdown("## 📖 How to Use This Application")
    
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Navigate to the Prediction Page** 📝
       - Click on "🔮 Predict" in the sidebar menu
       - You'll see a comprehensive form with various health parameters
    
    2. **Enter Your Health Information** 📋
       - Fill in all required fields with accurate information
       - Each field has helpful descriptions and valid ranges
       - Don't worry if you're unsure about some values - consult your recent medical reports
    
    3. **Submit for Analysis** 🔍
       - Click the "Predict Heart Disease Risk" button
       - Our AI model will analyze your data instantly
    
    4. **Review Your Results** 📊
       - View your risk assessment (Low/High Risk)
       - Check the confidence score of the prediction
       - Read personalized recommendations based on your results
    
    5. **Take Action** 💪
       - If high risk is detected, consult a cardiologist immediately
       - Follow the recommended lifestyle modifications
       - Schedule regular health check-ups
    
    ### ⚠️ Important Notes:
    
    > **Medical Disclaimer:** This tool is designed for educational and screening purposes only. 
    > It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. 
    > Always consult with qualified healthcare professionals for medical concerns.
    """)
    
    st.markdown("---")
    
    # Health Parameters Info
    st.markdown("## 🔬 Health Parameters Analyzed")
    
    with st.expander("📊 View All Parameters"):
        st.markdown("""
        Our model analyzes the following health indicators:
        
        - **Age**: Risk increases with age
        - **Sex**: Gender-specific risk factors
        - **Chest Pain Type**: Different types indicate varying risk levels
        - **Resting Blood Pressure**: Indicator of cardiovascular health
        - **Cholesterol**: High levels increase heart disease risk
        - **Fasting Blood Sugar**: Diabetes indicator
        - **Resting ECG Results**: Heart's electrical activity at rest
        - **Maximum Heart Rate**: Achieved during exercise
        - **Exercise-Induced Angina**: Chest pain during physical activity
        - **ST Depression (Oldpeak)**: ECG measurement during exercise
        - **ST Slope**: Pattern in ECG readings during exercise
        """)
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("## 📈 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='stat-box'>
        <h2>88.6%</h2>
        <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stat-box'>
        <h2>918</h2>
        <p>Training Samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stat-box'>
        <h2>15</h2>
        <p>Key Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='stat-box'>
        <h2>5</h2>
        <p>Models Tested</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💡 <strong>Remember:</strong> Prevention is better than cure. Regular health check-ups can save lives!</p>
        <p>Made with ❤️ using Streamlit and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== PREDICT PAGE ====================
elif page == "🔮 Predict":
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🔮 CardioX Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Enter your health parameters below for AI-powered risk assessment</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, scaler, columns = load_model_files()
    
    if model is None or scaler is None or columns is None:
        st.warning("⚠️ Model files not found. Using demo model for demonstration purposes.")
        st.info("""
        **For Best Results:**
        - Upload KNN_heart.pkl (trained model)
        - Upload scaler.pkl (feature scaler)  
        - Upload columns.pkl (feature names)
        
        **Currently using:** Demo model with limited training data.
        Predictions may not be as accurate as the full trained model.
        """)
        model, scaler, columns = create_demo_model()
    
    # Create input form
    st.markdown("## 📋 Patient Information")
    
    # Personal Information
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("👤 Age", min_value=1, max_value=120, value=50, 
                              help="Patient's age in years")
        
        sex = st.selectbox("⚥ Sex", options=["Male", "Female"], 
                           help="Biological sex of the patient")
        
        chest_pain_type = st.selectbox(
            "💔 Chest Pain Type",
            options=["Typical Angina (TA)", "Atypical Angina (ATA)", 
                     "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"],
            help="Type of chest pain experienced:\n"
                 "- TA: Typical Angina\n"
                 "- ATA: Atypical Angina\n"
                 "- NAP: Non-Anginal Pain\n"
                 "- ASY: Asymptomatic"
        )
        
        resting_bp = st.number_input("🩸 Resting Blood Pressure (mm Hg)", 
                                      min_value=80, max_value=200, value=120,
                                      help="Blood pressure at rest (normal: 120/80)")
    
    with col2:
        cholesterol = st.number_input("🧪 Serum Cholesterol (mg/dl)", 
                                       min_value=100, max_value=600, value=200,
                                       help="Total cholesterol level (normal: <200 mg/dl)")
        
        fasting_bs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl",
                                  options=["No", "Yes"],
                                  help="Is fasting blood sugar greater than 120 mg/dl?")
        
        resting_ecg = st.selectbox(
            "📈 Resting ECG Results",
            options=["Normal", "ST-T Wave Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"],
            help="Results of electrocardiography at rest:\n"
                 "- Normal: Normal ECG\n"
                 "- ST: ST-T wave abnormality\n"
                 "- LVH: Left ventricular hypertrophy"
        )
        
        max_hr = st.number_input("💓 Maximum Heart Rate Achieved", 
                                 min_value=60, max_value=220, value=150,
                                 help="Maximum heart rate during exercise test")
    
    # Exercise Test Results
    st.markdown("---")
    st.markdown("## 🏃 Exercise Test Results")
    
    col3, col4 = st.columns(2)
    
    with col3:
        exercise_angina = st.selectbox("😰 Exercise-Induced Angina",
                                       options=["No", "Yes"],
                                       help="Chest pain induced by exercise?")
        
        oldpeak = st.number_input("📉 ST Depression (Oldpeak)", 
                                  min_value=0.0, max_value=10.0, value=0.0, step=0.1,
                                  help="ST depression induced by exercise relative to rest")
    
    with col4:
        st_slope = st.selectbox(
            "📊 ST Slope",
            options=["Upsloping (Up)", "Flat", "Downsloping (Down)"],
            help="Slope of the peak exercise ST segment:\n"
                 "- Up: Upsloping\n"
                 "- Flat: Flat\n"
                 "- Down: Downsloping"
        )
    
    st.markdown("---")
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button("🔮 Predict Heart Disease Risk", type="primary")
    
    if predict_button:
        # Process inputs and create feature vector
        try:
            # Map categorical inputs
            sex_m = 1 if sex == "Male" else 0
            
            # Chest Pain Type mapping
            chest_pain_map = {
                "Typical Angina (TA)": "TA",
                "Atypical Angina (ATA)": "ATA",
                "Non-Anginal Pain (NAP)": "NAP",
                "Asymptomatic (ASY)": "ASY"
            }
            chest_pain = chest_pain_map[chest_pain_type]
            chest_pain_ata = 1 if chest_pain == "ATA" else 0
            chest_pain_nap = 1 if chest_pain == "NAP" else 0
            chest_pain_ta = 1 if chest_pain == "TA" else 0
            
            # Resting ECG mapping
            resting_ecg_normal = 1 if resting_ecg == "Normal" else 0
            resting_ecg_st = 1 if "ST" in resting_ecg else 0
            
            # Exercise Angina
            exercise_angina_y = 1 if exercise_angina == "Yes" else 0
            
            # ST Slope mapping
            st_slope_flat = 1 if st_slope == "Flat" else 0
            st_slope_up = 1 if "Up" in st_slope else 0
            
            # Fasting BS
            fasting_bs_value = 1 if fasting_bs == "Yes" else 0
            
            # Create feature dictionary
            features = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs_value,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'Sex_M': sex_m,
                'ChestPainType_ATA': chest_pain_ata,
                'ChestPainType_NAP': chest_pain_nap,
                'ChestPainType_TA': chest_pain_ta,
                'RestingECG_Normal': resting_ecg_normal,
                'RestingECG_ST': resting_ecg_st,
                'ExerciseAngina_Y': exercise_angina_y,
                'ST_Slope_Flat': st_slope_flat,
                'ST_Slope_Up': st_slope_up
            }
            
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([features])
            input_df = input_df[columns]  # Ensure correct column order
            
            # Scale the features
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class='prediction-box high-risk'>
                        ⚠️ HIGH RISK<br>
                        Heart Disease Detected
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.error("### ⚠️ Recommendation")
                    st.markdown("""
                    Based on the analysis, there are significant indicators of heart disease risk. 
                    **Please consult a cardiologist immediately** for a comprehensive evaluation.
                    
                    **Immediate Actions:**
                    - 📞 Schedule an appointment with a cardiologist
                    - 🏥 Get a complete cardiac work-up
                    - 💊 Discuss preventive medications if needed
                    - 🥗 Consider lifestyle modifications
                    - 🚭 Quit smoking if applicable
                    - 🏃 Start a supervised exercise program
                    """)
                else:
                    st.markdown("""
                    <div class='prediction-box low-risk'>
                        ✅ LOW RISK<br>
                        No Significant Heart Disease Indicators
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.success("### ✅ Recommendation")
                    st.markdown("""
                    The analysis shows low risk for heart disease. However, maintaining heart health is important.
                    
                    **Preventive Measures:**
                    - 💚 Maintain a healthy lifestyle
                    - 🥗 Eat a balanced diet
                    - 🏃 Exercise regularly (150 min/week)
                    - 😴 Get adequate sleep
                    - 😌 Manage stress effectively
                    - 🩺 Regular health check-ups annually
                    """)
            
            with col2:
                st.markdown("### 🎯 Confidence Scores")
                
                # Create a nice visualization of probabilities
                low_risk_pct = prediction_proba[0] * 100
                high_risk_pct = prediction_proba[1] * 100
                
                st.markdown(f"""
<div style='padding: 20px; background-color: #ffffff; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
    <h4 style='color: #262730; margin-top: 0; margin-bottom: 15px; font-size: 16px;'>Low Risk Probability</h4>
    <div style='background-color: #e8e8e8; border-radius: 10px; overflow: hidden; margin-bottom: 25px;'>
        <div style='background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); width: {low_risk_pct}%; padding: 12px 10px; color: white; font-weight: bold; text-align: center; min-width: 70px; font-size: 15px;'>
            {low_risk_pct:.1f}%
        </div>
    </div>
    <h4 style='color: #262730; margin-top: 0; margin-bottom: 15px; font-size: 16px;'>High Risk Probability</h4>
    <div style='background-color: #e8e8e8; border-radius: 10px; overflow: hidden;'>
        <div style='background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); width: {high_risk_pct}%; padding: 12px 10px; color: white; font-weight: bold; text-align: center; min-width: 70px; font-size: 15px;'>
            {high_risk_pct:.1f}%
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
                
                st.info("""
                **📌 Note:** The confidence score indicates how certain the model is about its prediction. 
                Higher scores mean higher confidence.
                """)
            
            # Feature importance visualization
            st.markdown("---")
            st.markdown("### 📈 Your Key Health Indicators")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Age", f"{age} years", 
                         delta="Risk increases with age" if age > 55 else "Normal range")
            
            with col2:
                st.metric("Blood Pressure", f"{resting_bp} mm Hg",
                         delta="High" if resting_bp > 140 else "Normal")
            
            with col3:
                st.metric("Cholesterol", f"{cholesterol} mg/dl",
                         delta="High" if cholesterol > 240 else "Normal")
            
            st.markdown("---")
            
            st.warning("""
            **⚕️ Medical Disclaimer:**  
            This prediction is based on machine learning algorithms and should NOT replace professional medical advice. 
            Always consult with qualified healthcare professionals for accurate diagnosis and treatment.
            """)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that all fields are filled correctly and try again.")
    
    # Information section
    with st.expander("ℹ️ How Does the Prediction Work?"):
        st.markdown("""
        ### 🤖 Machine Learning Model
        
        Our prediction system uses a **K-Nearest Neighbors (KNN)** algorithm trained on 918 patient records 
        with various cardiovascular health indicators.
        
        **Model Performance:**
        - **Accuracy:** 88.59%
        - **F1 Score:** 89.86%
        - **Algorithm:** K-Nearest Neighbors
        
        **Features Analyzed:**
        The model considers 15 key features including demographic information, vital signs, 
        ECG results, and exercise test outcomes to predict heart disease risk.
        
        **Data Processing:**
        1. Input data is standardized using the same scaler used during training
        2. Features are arranged in the exact order expected by the model
        3. The KNN algorithm compares your data with trained patterns
        4. A probability score is generated for risk assessment
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>💡 <strong>Remember:</strong> Early detection can save lives. If in doubt, always consult a healthcare professional!</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== ABOUT PAGE ====================
elif page == "ℹ️ About":
    st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>ℹ️ About CardioX</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Learn about our AI model, methodology, and the science behind predictions</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Project Overview
    st.markdown("## 🎯 Project Overview")
    
    st.markdown("""
    This Heart Disease Prediction System, **CardioX**, is an advanced machine learning application designed to assess 
    cardiovascular disease risk using clinical and diagnostic parameters. The system was developed to provide 
    a quick, accessible, and accurate screening tool that can help identify individuals who may benefit from 
    further cardiac evaluation.
    
    ### 🌟 Key Objectives:
    - Provide early screening for heart disease risk
    - Assist healthcare professionals in risk stratification
    - Educate users about cardiovascular health indicators
    - Democratize access to AI-powered health assessment tools
    """)
    
    st.markdown("---")
    
    # Dataset Information
    st.markdown("## 📊 Dataset Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
        <h4>📈 Dataset Characteristics</h4>
        <ul>
            <li><strong>Total Samples:</strong> 918 patient records</li>
            <li><strong>Features:</strong> 11 clinical features</li>
            <li><strong>Target Variable:</strong> Heart Disease (Binary: 0/1)</li>
            <li><strong>Data Source:</strong> UCI Machine Learning Repository</li>
            <li><strong>Feature Engineering:</strong> One-hot encoding for categorical variables</li>
            <li><strong>Final Feature Count:</strong> 15 features after encoding</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-box'>
        <h4>🔍 Clinical Features</h4>
        <ul>
            <li><strong>Demographic:</strong> Age, Sex</li>
            <li><strong>Clinical:</strong> Chest Pain Type, Blood Pressure, Cholesterol</li>
            <li><strong>Diagnostic:</strong> ECG Results, Fasting Blood Sugar</li>
            <li><strong>Exercise Test:</strong> Max Heart Rate, Exercise Angina, ST Depression, ST Slope</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Information
    st.markdown("## 🤖 Machine Learning Models Evaluated")
    
    st.markdown("""
    We tested multiple machine learning algorithms to find the best performer for heart disease prediction. 
    Here's a comparison of all models evaluated:
    """)
    
    # Model comparison table
    model_data = {
        'Model': ['K-Nearest Neighbors (KNN)', 'Logistic Regression', 'Support Vector Machine (SVM)', 
                  'Naive Bayes', 'Decision Tree'],
        'Accuracy': ['88.59%', '87.50%', '86.41%', '86.96%', '76.63%'],
        'F1 Score': ['89.86%', '88.78%', '88.04%', '87.88%', '77.95%'],
        'Status': ['✅ Selected', '❌', '❌', '❌', '❌']
    }
    
    df_models = pd.DataFrame(model_data)
    
    st.dataframe(df_models, use_container_width=True, hide_index=True)
    
    st.success("""
    **🏆 Selected Model: K-Nearest Neighbors (KNN)**
    
    KNN was chosen as the final model due to its superior performance metrics and robust prediction capabilities.
    """)
    
    st.markdown("---")
    
    # Selected Model Details
    st.markdown("## 🎓 Selected Model: K-Nearest Neighbors (KNN)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### How KNN Works
        
        K-Nearest Neighbors is a **non-parametric, instance-based learning algorithm** that makes predictions 
        based on similarity to training examples.
        
        **Algorithm Steps:**
        1. **Distance Calculation:** Calculate the distance between the input data point and all training samples
        2. **Neighbor Selection:** Identify the K nearest neighbors based on distance metrics
        3. **Majority Voting:** Predict the class based on the majority vote of K neighbors
        4. **Confidence Score:** Calculate probability based on the proportion of neighbors in each class
        
        ### Why KNN for Heart Disease?
        
        - **Pattern Recognition:** Excellent at identifying similar patient profiles
        - **Non-Linear Relationships:** Captures complex relationships between features
        - **Interpretability:** Easy to explain predictions based on similar cases
        - **Robust Performance:** Consistently high accuracy across different patient demographics
        """)
    
    with col2:
        st.markdown("""
        <div class='model-card'>
        <h3 style='text-align: center;'>📊 Model Metrics</h3>
        <hr style='border-color: white;'>
        <h4>Accuracy</h4>
        <h2 style='text-align: center;'>88.59%</h2>
        <h4>F1 Score</h4>
        <h2 style='text-align: center;'>89.86%</h2>
        <h4>Training Samples</h4>
        <h2 style='text-align: center;'>734</h2>
        <h4>Test Samples</h4>
        <h2 style='text-align: center;'>184</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature List
    st.markdown("## 🔬 Complete Feature List")
    
    st.markdown("""
    The model uses the following **15 engineered features** for prediction:
    """)
    
    features_info = {
        'Feature Name': [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
            'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 
            'ST_Slope_Flat', 'ST_Slope_Up'
        ],
        'Type': [
            'Numerical', 'Numerical', 'Numerical', 'Binary', 'Numerical', 'Numerical',
            'Binary', 'Binary', 'Binary', 'Binary', 'Binary', 'Binary', 'Binary', 
            'Binary', 'Binary'
        ],
        'Description': [
            'Patient age in years',
            'Resting blood pressure (mm Hg)',
            'Serum cholesterol (mg/dl)',
            'Fasting blood sugar > 120 mg/dl (1=Yes, 0=No)',
            'Maximum heart rate achieved',
            'ST depression induced by exercise',
            'Male sex (1=Male, 0=Female)',
            'Atypical Angina chest pain',
            'Non-Anginal Pain chest pain',
            'Typical Angina chest pain',
            'Normal resting ECG',
            'ST-T wave abnormality',
            'Exercise-induced angina (1=Yes, 0=No)',
            'Flat ST slope',
            'Upsloping ST slope'
        ]
    }
    
    df_features = pd.DataFrame(features_info)
    st.dataframe(df_features, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Technical Implementation
    st.markdown("## 💻 Technical Implementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🛠️ Technologies Used
        
        **Frontend & UI:**
        - Streamlit (Interactive web interface)
        - Custom CSS for styling
        - Responsive design
        
        **Machine Learning:**
        - scikit-learn (Model training & evaluation)
        - pandas (Data manipulation)
        - numpy (Numerical operations)
        
        **Model Persistence:**
        - joblib/pickle (Model serialization)
        - Trained model files: KNN_heart.pkl
        - Scaler: scaler.pkl
        - Feature names: columns.pkl
        """)
    
    with col2:
        st.markdown("""
        ### 📦 Model Deployment
        
        **Architecture:**
        - Single-page Streamlit application
        - Cached model loading for performance
        - Real-time prediction inference
        
        **Required Files:**
        ```
        app.py              # Main application
        KNN_heart.pkl       # Trained KNN model
        scaler.pkl          # Feature scaler
        columns.pkl         # Feature names
        ```
        
        **To Run:**
        ```bash
        streamlit run app.py
        ```
        """)
    
    st.markdown("---")
    
    # Limitations and Future Work
    st.markdown("## ⚠️ Limitations & Disclaimer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.warning("""
        ### 🚨 Known Limitations
        
        1. **Sample Size:** Model trained on 918 samples - larger datasets could improve accuracy
        2. **Population Bias:** Dataset may not represent all demographic groups equally
        3. **Feature Limitation:** Limited to 11 clinical features; more comprehensive data could enhance predictions
        4. **Temporal Validity:** Medical knowledge evolves; model should be periodically retrained
        5. **Not Diagnostic:** This is a screening tool, not a replacement for clinical diagnosis
        """)
    
    with col2:
        st.info("""
        ### 🔮 Future Enhancements
        
        1. **Deep Learning:** Explore neural networks for improved accuracy
        2. **Explainable AI:** Add SHAP/LIME for better interpretability
        3. **More Features:** Include genetic markers, lifestyle factors
        4. **Real-time Learning:** Continuous model updates with new data
        5. **Mobile App:** Native mobile application for broader access
        6. **Integration:** Connect with electronic health records (EHR)
        """)
    
    st.markdown("---")
    
    # Medical Disclaimer
    st.markdown("## ⚕️ Medical Disclaimer")
    
    st.error("""
    **IMPORTANT MEDICAL DISCLAIMER**
    
    CardioX (Heart Disease Prediction System) is designed for **educational and informational purposes only**. 
    It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment.
    
    **Important Points:**
    - This tool provides risk assessment, not diagnosis
    - Always seek the advice of qualified healthcare providers
    - Never disregard professional medical advice based on this tool
    - In case of emergency, contact emergency services immediately
    - Results should be discussed with a cardiologist or primary care physician
    - The model's accuracy of 88.59% means it can still make errors
    
    **If you experience chest pain, shortness of breath, or other cardiac symptoms, seek immediate medical attention.**
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>CardioX v1.0</strong> - Heart Disease Prediction System</p>
        <p>Built with ❤️ using Streamlit and Machine Learning</p>
        <p>© 2024 - For Educational and Research Purposes</p>
        <hr>
        <p><em>"Prevention is better than cure. Take care of your heart!"</em> 💚</p>
    </div>
    """, unsafe_allow_html=True)