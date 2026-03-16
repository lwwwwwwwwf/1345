import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="Academic Warning Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model = joblib.load("academic_warning_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'academic_warning_model.pkl' not found!")
        st.stop()

# Load model
model = load_model()

st.title("🎓 Academic Warning Prediction")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("📊 Student Information")
st.sidebar.markdown("Enter the student's details:")

# Input fields with better validation and defaults
col1, col2 = st.columns(2)

with col1:
    gpa = st.number_input(
        "📈 **GPA**", 
        min_value=0.0, 
        max_value=4.3, 
        value=2.5, 
        step=0.1,
        help="Cumulative GPA (0.0 - 4.3)"
    )
    
    credits = st.number_input(
        "📚 **Credits Registered**", 
        min_value=0, 
        max_value=30, 
        value=15, 
        step=1,
        help="Total credits registered this semester"
    )

with col2:
    absences = st.number_input(
        "🚫 **Absences**", 
        min_value=0, 
        max_value=50, 
        value=5, 
        step=1,
        help="Number of absences this semester"
    )
    
    major = st.selectbox(
        "🎯 **Major**",
        ["IT", "Business", "Economics", "Engineering"],
        help="Student's major field of study"
    )

# Prepare data - Handle categorical encoding if needed
data_dict = {
    "gpa": [gpa],
    "credits": [credits],
    "absences": [absences],
    "major": [major]
}

data = pd.DataFrame(data_dict)

# Display input summary
st.markdown("### 📋 Input Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("GPA", f"{gpa:.2f}")
col2.metric("Credits", credits)
col3.metric("Absences", absences)
col4.metric("Major", major)

# Prediction button
if st.button("🔮 **Predict Academic Status**", type="primary", use_container_width=True):
    
    try:
        # Make prediction
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0] if hasattr(model, 'predict_proba') else None
        
        # Results section
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if prediction == 1:
                st.error("⚠️ **ACADEMIC WARNING RISK**")
                st.markdown("**Student may receive Academic Warning**")
                
                if probability is not None:
                    risk_prob = probability[1] * 100
                    st.metric("Risk Probability", f"{risk_prob:.1f}%")
                    
            else:
                st.success("✅ **SAFE STATUS**")
                st.markdown("**Student is in good academic standing**")
                
                if probability is not None:
                    safe_prob = probability[0] * 100
                    st.metric("Safety Probability", f"{safe_prob:.1f}%")
        
        with col2:
            st.markdown("### 📊")
            if probability is not None:
                st.markdown(f"""
                **Risk**: {probability[1]:.1%}
                **Safe**: {probability[0]:.1%}
                """)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 1:
            st.warning("""
            **Immediate Actions:**
            - Meet with academic advisor
            - Review study habits
            - Consider reducing course load
            - Attend tutoring sessions
            """)
        else:
            st.info("""
            **Good Practices:**
            - Maintain current performance
            - Stay consistent with attendance
            - Plan ahead for next semester
            """)
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please check your inputs and model compatibility.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #778899;'>
        Developed with ❤️ for academic success | 🎓
    </div>
    """, 
    unsafe_allow_html=True
)
