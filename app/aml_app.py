"""
Anti Money Laundering Detection System - Streamlit App
A professional dashboard for detecting suspicious transactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AML Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark green theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #00cc66;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px 30px;
        font-size: 18px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00b359;
        border: 2px solid #00ff7f;
    }
    .suspicious-box {
        background-color: #ff4444;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(255, 68, 68, 0.3);
    }
    .normal-box {
        background-color: #00cc66;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 204, 102, 0.3);
    }
    .metric-card {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00cc66;
    }
    .model-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
        font-size: 14px;
    }
    h1 {
        color: #00ff7f;
    }
    h2, h3 {
        color: #00cc66;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('model/aml_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please run aml_train_model.py first.")
        st.stop()

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
feature_columns = model_data['feature_columns']
model_accuracy = model_data['accuracy']

# Get model name (with fallback for older saved models)
model_name = model_data.get('model_name', 'Random Forest')

# Header
st.title("💵 Anti Money Laundering Detection System")
st.markdown("### Detect Suspicious Banking Transactions with Machine Learning")

# Display active model badge
st.markdown(f"""
    <div style='text-align: center;'>
        <span class='model-badge'>🤖 Active Model: {model_name}</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910791.png", width=100)
    st.markdown("## 🔒 System Info")
    
    # Display current model being used
    st.markdown(f"**🤖 Model:** `{model_name}`")
    st.markdown(f"**Model Accuracy:** `{model_accuracy*100:.2f}%`")
    if 'precision' in model_data:
        st.markdown(f"**Precision:** `{model_data['precision']*100:.2f}%`")
        st.markdown(f"**Recall:** `{model_data['recall']*100:.2f}%`")
        st.markdown(f"**F1-Score:** `{model_data['f1_score']*100:.2f}%`")
    
    st.markdown(f"**Features:** {len(feature_columns)}")
    
    # Show model comparison if available
    if 'all_model_results' in model_data and model_data['all_model_results']:
        st.markdown("---")
        st.markdown("### 📊 Model Comparison")
        
        # Create expandable section for model comparison
        with st.expander("View All Models Performance", expanded=False):
            all_results = model_data['all_model_results']
            
            # Display each model's metrics
            for m_name, metrics in all_results.items():
                winner_badge = "🏆 " if m_name == model_name else ""
                st.markdown(f"**{winner_badge}{m_name}**")
                st.markdown(f"- Accuracy: `{metrics['accuracy']*100:.2f}%`")
                st.markdown(f"- Precision: `{metrics['precision']*100:.2f}%`")
                st.markdown(f"- Recall: `{metrics['recall']*100:.2f}%`")
                st.markdown(f"- F1: `{metrics['f1_score']*100:.2f}%`")
                st.markdown("---")
            
            st.info(f"✅ {model_name} was selected for having the highest accuracy.")
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("This system uses machine learning to identify potentially suspicious transactions based on various risk factors.")
    st.markdown("---")
    st.markdown("### ⚠️ Risk Indicators")
    st.markdown("""
    - High transaction amounts
    - Foreign transactions
    - High country risk level
    - Previous suspicious activity
    - Unusual velocity score
    - Location mismatches
    - Late night transactions
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 📝 Transaction Details")
    
    # Input fields in organized columns
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=5000.0,
            step=100.0,
            help="The amount of money that the individual has transfered"
        )
        
        transaction_type = st.selectbox(
            "Transaction Type",
            options=list(label_encoders['transaction_type'].classes_),
            help="Type of transaction being performed"
        )
        
        account_age_years = st.number_input(
            "Account Age (years)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="How long the account has been active"
        )
        
        country_risk_level = st.selectbox(
            "Country Risk Level",
            options=list(label_encoders['country_risk_level'].classes_),
            help="Risk level of the particular country range"
        )
        
        is_foreign_transaction = st.selectbox(
            "Foreign Transaction",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Is this a foreign transaction?"
        )
    
    with input_col2:
        num_prev_transactions = st.number_input(
            "Previous Transactions",
            min_value=0,
            max_value=1000,
            value=25,
            step=1,
            help="Number of previous transactions"
        )
        
        previous_suspicious_activity = st.selectbox(
            "Previous Suspicious Activity",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Has there been previous suspicious activity?"
        )
        
        avg_transaction_amount = st.number_input(
            "Avg Transaction Amount ($)",
            min_value=0.0,
            max_value=1000000.0,
            value=2000.0,
            step=100.0,
            help="Average transaction amount by the holder had processed"
        )
        
        time_of_day = st.selectbox(
            "Time of Day",
            options=list(label_encoders['time_of_day'].classes_),
            help="Time when transaction occurred and duration of it"
        )
        
        device_type = st.selectbox(
            "Device Type",
            options=list(label_encoders['device_type'].classes_),
            help="Device used for the transaction"
        )
    
    with input_col3:
        customer_type = st.selectbox(
            "Customer Type",
            options=list(label_encoders['customer_type'].classes_),
            help="Type of account category"
        )
        
        location_match = st.selectbox(
            "Location Match",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Does location match usual pattern?"
        )
        
        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=15000.0,
            step=500.0,
            help=" account balance"
        )
        
        txn_velocity_score = st.slider(
            "Transaction Velocity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Transaction frequency score (0-1)"
        )
        
        unusual_activity_flag = st.selectbox(
            "Unusual Activity Flag",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Has unusual activity been detected?"
        )

with col2:
    st.markdown("## 🎯 Model Performance")
    
    # Accuracy gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=model_accuracy * 100,
        title={'text': f"{model_name} Accuracy"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00cc66"},
            'steps': [
                {'range': [0, 50], 'color': "#ff4444"},
                {'range': [50, 75], 'color': "#ffaa00"},
                {'range': [75, 100], 'color': "#00cc66"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'size': 14}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# Prediction button
st.markdown("---")
predict_button = st.button("🔍 ANALYZE TRANSACTION")

if predict_button:
    # Prepare input data
    input_data = {
        'amount': amount,
        'transaction_type': label_encoders['transaction_type'].transform([transaction_type])[0],
        'account_age_years': account_age_years,
        'country_risk_level': label_encoders['country_risk_level'].transform([country_risk_level])[0],
        'is_foreign_transaction': is_foreign_transaction,
        'num_prev_transactions': num_prev_transactions,
        'previous_suspicious_activity': previous_suspicious_activity,
        'avg_transaction_amount': avg_transaction_amount,
        'time_of_day': label_encoders['time_of_day'].transform([time_of_day])[0],
        'device_type': label_encoders['device_type'].transform([device_type])[0],
        'customer_type': label_encoders['customer_type'].transform([customer_type])[0],
        'location_match': location_match,
        'account_balance': account_balance,
        'txn_velocity_score': txn_velocity_score,
        'unusual_activity_flag': unusual_activity_flag
    }
    
    # Create DataFrame with correct column order
    input_df = pd.DataFrame([input_data])[feature_columns]
    
    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.markdown(f"## 🎯 Analysis Results (Predicted by {model_name})")
    
    result_col1, result_col2 = st.columns([1, 1])
    
    with result_col1:
        if prediction == 1:
            st.markdown(f"""
                <div class="suspicious-box">
                    ⚠️ SUSPICIOUS TRANSACTION<br>
                    <span style="font-size: 16px;">Confidence: {prediction_proba[1]*100:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.error("🚨 This transaction shows suspicious patterns and requires further investigation.")
        else:
            st.markdown(f"""
                <div class="normal-box">
                    ✓ NORMAL TRANSACTION<br>
                    <span style="font-size: 16px;">Confidence: {prediction_proba[0]*100:.1f}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.success("✅ This transaction appears to be legitimate.")
    
    with result_col2:
        # Probability distribution pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Normal', 'Suspicious'],
            values=[prediction_proba[0], prediction_proba[1]],
            hole=.4,
            marker_colors=['#00cc66', '#ff4444'],
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig_pie.update_layout(
            title="Prediction Probability",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 12},
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed metrics
    st.markdown("### 📊 Detailed Analysis")
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        st.metric("Transaction Risk", f"{prediction_proba[1]*100:.1f}%", 
                 delta=f"{(prediction_proba[1]-0.5)*100:.1f}%")
    
    with metric_col2:
        risk_level = "HIGH" if prediction_proba[1] > 0.7 else "MEDIUM" if prediction_proba[1] > 0.4 else "LOW"
        st.metric("Risk Level", risk_level)
    
    with metric_col3:
        ratio = (amount / account_balance * 100) if account_balance > 0 else 0
        st.metric("Amount/Balance", f"{ratio:.1f}%")
    
    with metric_col4:
        st.metric("Model Confidence", f"{max(prediction_proba)*100:.1f}%")
    
    with metric_col5:
        vs_avg = ((amount - avg_transaction_amount) / avg_transaction_amount * 100) if avg_transaction_amount > 0 else 0
        st.metric("vs Avg Amount", f"{vs_avg:+.1f}%")
    
    # Risk factors summary
    st.markdown("### ⚠️ Risk Factors Detected")
    risk_factors = []
    
    if amount > avg_transaction_amount * 3:
        risk_factors.append("⚠️ Transaction amount is significantly higher than average")
    if is_foreign_transaction == 1:
        risk_factors.append("🌍 Foreign transaction detected")
    if previous_suspicious_activity == 1:
        risk_factors.append("🚩 Previous suspicious activity on record")
    if country_risk_level in ['high']:
        risk_factors.append("🏴 High-risk country transaction")
    if location_match == 0:
        risk_factors.append("📍 Location mismatch detected")
    if txn_velocity_score > 0.7:
        risk_factors.append("⚡ High transaction velocity")
    if unusual_activity_flag == 1:
        risk_factors.append("🔔 Unusual activity flag triggered")
    if time_of_day in ['night']:
        risk_factors.append("🌙 Late night transaction")
    if account_balance > 0 and amount > account_balance * 0.5:
        risk_factors.append("💰 Large portion of account balance")
    
    if risk_factors:
        for factor in risk_factors:
            st.warning(factor)
    else:
        st.info("✅ No significant risk factors detected")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🔒 Anti Money Laundering Detection System | Powered by Varun & Mughul </p>
        <p style='font-size: 12px;'>this system  predicts the suspicious and normal trascation </p>
    </div>
""", unsafe_allow_html=True)