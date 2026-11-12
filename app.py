import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier

st.set_page_config(page_title="Motor Diagnostics", layout="centered")
st.title("🏭 Motor Temperature Diagnostics")
st.write("Enter the morning temperature readings below to check component health.")

# --- YOUR DATA LOADING AND SAMPLING CODE ---
df = pd.read_csv('Raw data.csv')
coolant = df['coolant'].to_numpy()
stator_winding = df['stator_winding'].to_numpy()
ambient = df['ambient'].to_numpy()

def representative_sample(series, n=20, random_state=42):
    bins = pd.qcut(series, q=n, duplicates='drop')
    sampled = (
        series.groupby(bins, observed=True)
              .apply(lambda x: x.sample(1, random_state=random_state))
    )
    return sampled.to_numpy()

coolant_array = representative_sample(df['coolant'])
stator_winding_array = representative_sample(df['stator_winding'])
ambient_array = representative_sample(df['ambient'])

# --- YOUR COOLANT MODEL ---
y_coolant = np.array(['Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Critical', 'Critical', 'Critical'])
scaler = StandardScaler()
X_reshaped = coolant_array.reshape(-1,1)
X_scaled = scaler.fit_transform(X_reshaped)

lrmodel = LogisticRegression()
lrmodel.fit(X_reshaped,y_coolant)

# --- YOUR STATOR MODEL ---
model_stator = LogisticRegression()
stator_winding_reshaped = stator_winding_array.reshape(-1,1)
Y_stator_winding = np.array(['Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Optimal', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Critical', 'Critical', 'Critical', 'Critical'])
model_stator.fit(stator_winding_reshaped,Y_stator_winding)

# --- YOUR AMBIENT MODEL ---
model_ambient = LogisticRegression()
ambient_reshaped = ambient_array.reshape(-1,1)
Y_amb = np.array(['Optimal', 'Optimal', 'Optimal', 'Optimal', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Check', 'Critical'])
model_ambient.fit(ambient_reshaped,Y_amb)

st.divider()

# --- CLEAN INPUT UI ---
col1, col2, col3 = st.columns(3)

with col1:
    amb = st.number_input("Ambient Temp (°C)", value=0.0)
with col2:
    stator = st.number_input("Stator Temp (°C)", value=0.0)
with col3:
    cool = st.number_input("Coolant Temp (°C)", value=0.0)

# --- YOUR PREDICTION LOGIC WITH CLEAN OUTPUT UI ---
if st.button("Submit Temps", use_container_width=True):
    # Your prediction engine
    w_reshape = np.array([cool]).reshape(-1,1)
    ypred = lrmodel.predict(w_reshape)

    stator_winding_r = np.array([stator]).reshape(-1,1)
    spred = model_stator.predict(stator_winding_r)

    amb_r = np.array([amb]).reshape(-1,1)
    apred = model_ambient.predict(amb_r)

    # Clean Output UI
    st.subheader("📋 Diagnostic Results")
    
    # Helper function to display colors based on your classes
    def display_status(component, status):
        # We use [0] to extract the string from your prediction array (e.g., 'Optimal' instead of ['Optimal'])
        if status[0] == 'Critical':
            st.error(f"**{component}:** CRITICAL 🔴")
        elif status[0] == 'Check':
            st.warning(f"**{component}:** CHECK 🟡")
        else:
            st.success(f"**{component}:** OPTIMAL 🟢")

    display_status("Ambient Air", apred)
    display_status("Stator Winding", spred)
    display_status("Coolant", ypred)