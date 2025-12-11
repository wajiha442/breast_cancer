import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide"
)

# Load the model, scaler, and feature names
@st.cache_resource
def load_model():
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

try:
    knn_model, scaler, feature_names = load_model()
    model_loaded = True
except:
    model_loaded = False

# Title and description
st.title("🏥 Breast Cancer Prediction using KNN")
st.markdown("---")

if not model_loaded:
    st.error("⚠ Model files not found! Please train the model first by running 'train_model.py'")
    st.stop()

st.markdown("""
This application uses a *K-Nearest Neighbors (KNN)* machine learning model to predict 
whether a breast tumor is *Malignant* or *Benign* based on cell characteristics.
""")

# Sidebar for model info
st.sidebar.header("📊 Model Information")
st.sidebar.info("""
*Algorithm:* K-Nearest Neighbors (KNN)

*Dataset:* Breast Cancer Wisconsin

*Features:* 30 cell characteristics

*Classes:*
- 🔴 Malignant (Cancer)
- 🟢 Benign (No Cancer)
""")

st.sidebar.markdown("---")
st.sidebar.header("📖 How to Use")
st.sidebar.markdown("""
1. Choose input method (Manual or Upload CSV)
2. Enter feature values or upload file
3. Click 'Predict' button
4. View prediction results
""")

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Make Prediction", "📈 About the Model", "ℹ Feature Information"])

with tab1:
    st.header("Enter Patient Data")
    
    # Input method selection
    input_method = st.radio("Select Input Method:", ["Manual Input", "Upload CSV"])
    
    if input_method == "Manual Input":
        st.markdown("### Enter Feature Values")
        st.info("💡 Tip: You can use default values to test the application")
        
        # Create columns for input
        col1, col2, col3 = st.columns(3)
        
        input_data = {}
        
        # Divide features into 3 columns
        features_per_col = len(feature_names) // 3
        
        with col1:
            st.subheader("Mean Values")
            for i in range(features_per_col):
                feature = feature_names[i]
                input_data[feature] = st.number_input(
                    f"{feature}", 
                    value=0.0, 
                    format="%.4f",
                    key=f"input_{i}"
                )
        
        with col2:
            st.subheader("SE Values")
            for i in range(features_per_col, 2*features_per_col):
                feature = feature_names[i]
                input_data[feature] = st.number_input(
                    f"{feature}", 
                    value=0.0, 
                    format="%.4f",
                    key=f"input_{i}"
                )
        
        with col3:
            st.subheader("Worst Values")
            for i in range(2*features_per_col, len(feature_names)):
                feature = feature_names[i]
                input_data[feature] = st.number_input(
                    f"{feature}", 
                    value=0.0, 
                    format="%.4f",
                    key=f"input_{i}"
                )
        
        # Sample data button
        if st.button("📝 Load Sample Data (Benign)"):
            st.info("Sample benign case loaded!")
            st.experimental_rerun()
        
        # Predict button
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            # Prepare input
            input_array = np.array([list(input_data.values())])
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = knn_model.predict(input_scaled)[0]
            prediction_proba = knn_model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", 
                         "🟢 Benign" if prediction == 1 else "🔴 Malignant",
                         delta="No Cancer" if prediction == 1 else "Cancer Detected")
            
            with col2:
                st.metric("Confidence", 
                         f"{max(prediction_proba)*100:.2f}%")
            
            with col3:
                st.metric("Risk Level",
                         "Low" if prediction == 1 else "High")
            
            # Probability bar
            st.markdown("### Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['Malignant', 'Benign'],
                'Probability': prediction_proba
            })
            st.bar_chart(prob_df.set_index('Class'))
            
            # Recommendation
            if prediction == 1:
                st.success("""
                ✅ *Good News!* The tumor is predicted to be *Benign* (non-cancerous).
                
                However, please consult with a medical professional for proper diagnosis.
                """)
            else:
                st.error("""
                ⚠ *Attention Required!* The tumor is predicted to be *Malignant* (cancerous).
                
                *Important:* This is just a prediction. Please consult with an oncologist immediately 
                for proper medical evaluation and treatment.
                """)
    
    else:  # CSV Upload
        st.markdown("### Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("🔮 Predict All", type="primary"):
                # Scale the data
                X_scaled = scaler.transform(df)
                
                # Make predictions
                predictions = knn_model.predict(X_scaled)
                predictions_proba = knn_model.predict_proba(X_scaled)
                
                # Add predictions to dataframe
                df['Prediction'] = ['Benign' if p == 1 else 'Malignant' for p in predictions]
                df['Confidence'] = [max(p)*100 for p in predictions_proba]
                
                st.success(f"✅ Predictions completed for {len(df)} samples!")
                st.dataframe(df[['Prediction', 'Confidence']])
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

with tab2:
    st.header("📈 About the KNN Model")
    
    st.markdown("""
    ### What is K-Nearest Neighbors (KNN)?
    
    KNN is a simple, yet powerful machine learning algorithm that makes predictions based on similarity. 
    It works by:
    
    1. *Finding Similar Cases:* Looks at the K most similar cases in the training data
    2. *Voting Mechanism:* Takes a majority vote from these neighbors
    3. *Making Prediction:* Assigns the most common class among neighbors
    
    ### Why KNN for Cancer Prediction?
    
    - ✅ *Simple and Interpretable:* Easy to understand and explain
    - ✅ *No Training Phase:* Quick to implement
    - ✅ *Works Well with Medical Data:* Effective for feature-rich datasets
    - ✅ *Non-parametric:* Makes no assumptions about data distribution
    
    ### Model Performance
    
    Our KNN model achieves:
    - *High Accuracy:* Correctly classifies most cases
    - *Good Precision:* Low false positive rate
    - *High Recall:* Detects most cancer cases
    - *Balanced F1-Score:* Good overall performance
    
    ### Important Note
    
    ⚠ This model is for *educational purposes only*. Always consult with qualified 
    medical professionals for actual diagnosis and treatment decisions.
    """)

with tab3:
    st.header("ℹ Feature Information")
    
    st.markdown("""
    The model uses *30 features* computed from cell images. These features describe 
    characteristics of cell nuclei present in the image.
    
    ### Feature Categories:
    
    1. *Mean Features (10):* Average values
    2. *SE Features (10):* Standard error values  
    3. *Worst Features (10):* Largest (worst) values
    
    ### Key Features Include:
    
    - *Radius:* Mean distance from center to perimeter
    - *Texture:* Standard deviation of gray-scale values
    - *Perimeter:* Size of the core tumor
    - *Area:* Area of the tumor
    - *Smoothness:* Local variation in radius lengths
    - *Compactness:* Perimeter² / area - 1.0
    - *Concavity:* Severity of concave portions
    - *Concave Points:* Number of concave portions
    - *Symmetry:* Symmetry of the cell
    - *Fractal Dimension:* Coastline approximation
    """)
    
    st.info("""
    💡 *Tip:* Higher values in certain features (like radius, area, concavity) 
    are often associated with malignant tumors.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤ using Streamlit | Lab 10: Machine Learning</p>
    <p><small>⚠ For Educational Purposes Only - Not for Medical Diagnosis</small></p>
</div>
""", unsafe_allow_html=True)