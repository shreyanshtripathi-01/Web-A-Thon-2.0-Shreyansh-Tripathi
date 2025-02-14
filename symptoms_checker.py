import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the datasets
symptoms_df = pd.read_csv('dataset/symptoms_dataset.csv')
drugs_df = pd.read_csv('dataset/drugs_side_effects.csv')
precautions_df = pd.read_csv('dataset/disease_precaution.csv')

# Preprocess and map diseases
symptoms_df['TYPE'] = symptoms_df['TYPE'].replace({
    'COLD': 'Colds & Flu',
    'FLU': 'Colds & Flu',
    'COVID': 'Covid 19',
    'ALLERGY': 'Allergies'
})

# Prepare features and target
X = symptoms_df.drop('TYPE', axis=1)
y = symptoms_df['TYPE']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.set_page_config(page_title="Symptom Checker", page_icon="🩺")

st.title("🩺 Symptom Checker with Drug Info")
st.markdown("### Check your symptoms and get a diagnosis with drug info!")

# Add a sidebar
st.sidebar.header("Select Your Symptoms")

# Create checkboxes for each symptom
symptoms = X.columns
user_input = []

for symptom in symptoms:
    user_input.append(st.sidebar.checkbox(symptom.replace("_", " ").title()))

# Convert user input to numpy array
user_input = np.array(user_input).reshape(1, -1)

# Predict the condition
if user_input.any():  # Proceed only if at least one symptom is selected
    prediction = model.predict(user_input)[0]
    st.markdown(f"## Predicted Condition: **:red[{prediction}]**")

    # Find and display precaution information
    st.markdown("### Precaution Information:")
    precaution_info = precautions_df[precautions_df['Disease'].str.contains(prediction, case=False, na=False)]
    
    if not precaution_info.empty:
        st.markdown("#### Precautions:")
        for i in range(1, 5):  # Assuming there are four precaution columns
            precaution_column = f'Precaution_{i}'
            if precaution_column in precaution_info.columns:
                st.write(f"- {precaution_info.iloc[0][precaution_column]}")
    else:
        st.markdown("No precaution information available for this condition.")

    # Find and display drug information
    drug_info = drugs_df[drugs_df['medical_condition'].str.contains(prediction, case=False, na=False)]
    
    if not drug_info.empty:
        st.markdown("### Related Drugs:")
        
        # Create a selectbox for drug names
        selected_drug_name = st.selectbox("Select a drug to see more details:", drug_info['drug_name'].unique())

        if selected_drug_name:
            st.markdown(f"## Drug Information: **{selected_drug_name}**")
            selected_drug = drug_info[drug_info['drug_name'] == selected_drug_name].iloc[0]
            for col in drug_info.columns:
                if col != 'drug_name':
                    st.write(f"**{col.replace('_', ' ').title()}:** {selected_drug[col]}")
    else:
        st.markdown("No drug information available for this condition.")
else:
    st.error("Please select at least one symptom to get a prediction.")

st.text("")
st.markdown("""---""")
st.markdown("### Usage")
st.write("""
1. Select your symptoms from the sidebar.
2. The predicted condition and relevant drugs will automatically update.
3. Select a drug name to view its detailed information.
""")
st.markdown("""---""")
st.write("""
    ### Important Notes
    - The app is **`NOT`** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    - The content provided is for educational and informational purposes only!
    """)
st.markdown("""---""")
st.markdown("### About the Application")
st.write("""
This application is designed to predict minor illnesses based on selected symptoms and provide relevant drug information. 
The model is trained on a dataset of symptoms and conditions and cross-referenced with drug data for comprehensive insights.

### Team Name
Robo Giga Boys - 1 Member:
by SHREYANSH TRIPATHI.
""")

# Add some style
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #dfe6f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
