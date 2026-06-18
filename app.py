# =====================================================================
# THE OBJECTIVE: This file builds a beautiful web interface.
# It gives non-technical users sliders and buttons to input farmer data,
# feeds those inputs to our frozen ML brain, and prints the answer.
# =====================================================================

import streamlit as st
import joblib
import pandas as pd

# -----------------------------------------------------------------
# 1. SETTING UP THE WEB PAGE TITLE
# -----------------------------------------------------------------
st.set_page_config(page_title="JPIL Farms Yield Predictor", layout="centered")

st.title("🌾 JPIL Farms Smallholder Yield Prediction App")
st.write("Move the sliders below to predict how a farmer's crop yield percentage will change based on resource inputs and extension training.")

# -----------------------------------------------------------------
# 2. BUILDING THE INPUT INTERFACE (SLIDERS & TOGGLES)
# -----------------------------------------------------------------
st.subheader("👨‍🌾Input Farmer Characteristics")

# Create a slider for Farm Size (Minimum: 0.5 ha, Maximum: 10.0 ha, Default: 2.0 ha)
farm_size = st.slider("Select Farm Size (Hectares):", min_value=0.5, max_value=10.0, value=2.0, step=0.1)

# Create a slider for Fertilizer Usage (Minimum: 0 kg, Maximum: 300 kg, Default: 50 kg)
fertilizer = st.slider("Select Fertilizer Used (kg):", min_value=0, max_value=300, value=50, step=5)

# Create a drop-down selector box for Training Attendance
training_attendance = st.selectbox("Did the farmer attend training?", options=["No", "Yes"])

# -----------------------------------------------------------------
# 3. BACKGROUND DATA CONVERSION (THE CONVEYOR BELT)
# -----------------------------------------------------------------
# Convert the user's "Yes" or "No" choice into 1 or 0 so the model understands it
training_encoded = 1 if training_attendance == "Yes" else 0

# Calculate our engineered interaction feature (Fertilizer multiplied by Training)
fertilizer_x_training = fertilizer * training_encoded

# Pack these inputs into a tidy single-row DataFrame with exact column names expected by the robot
input_data = pd.DataFrame({
    'Farm Size (ha)': [farm_size],
    'Fertilizer Used (kg)': [fertilizer],
    'Training_Encoded': [training_encoded],
    'Fertilizer_x_Training': [fertilizer_x_training]
})

# -----------------------------------------------------------------
# 4. LOAD THE MACHINE LEARNING ROBOT
# -----------------------------------------------------------------
# Wake up our frozen Random Forest model from its permanent folder asset
frozen_robot = joblib.load('machine_learning_model/jpil_random_forest_model.pkl')

# -----------------------------------------------------------------
# 5. GENERATE AND SHOW PREDICTIONS LIVE
# -----------------------------------------------------------------
st.write("---")
st.subheader("🔮Live Predictive Result")

# When the web user clicks this big button, run the model
if st.button("Calculate Predicted Yield Change"):
    
    # Run the single-row input table through the live math engine
    prediction = frozen_robot.predict(input_data)
    
    # Grab the single numerical answer out of the array
    final_score = prediction[0]
    
    # Display the result to the user in a beautiful highlighted box
    st.success(f"📈 **Predicted Crop Yield Change:** {final_score:.2f}%")
    
    # Provide deep analytical context based on their inputs
    if training_attendance == "Yes":
        st.info("💡 **Insight:** Notice how the yield prediction jumps higher! This reflects the high feature importance of the combined interaction term.")
    else:
        st.warning("💡 **Insight:** Without training, the return on fertilizer usage remains lower due to unoptimized factor efficiency.")