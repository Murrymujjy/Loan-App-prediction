import joblib
import pandas as pd
import streamlit as st


models = joblib.load('loan_models.pkl')


st.title("Loan Prediction App")

# Input sections
st.header("User Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age")
    years_lived_in_community = st.number_input("years_lived_in_community")
    phone_access = st.selectbox("Phone Access", [0, 1])
    
with col2:
    education_map = {'SENIOR SECONDARY': 0, 'PRIMARY': 1, 'JUNIOR SECONDARY': 2, 'NONE': 3,
       'MODERN SCHOOL': 4, '1st DEGREE': 5, 'QUARANIC/INTEGRATED QUARANIC': 6,
       'NATIONAL CERTIFICATE OF EDUCATION (NCE)': 7,
       'HIGHER DEGREE (POST-GRADUATE)': 8, 'POLYTECHNIC/PROF': 9,
       'TERTIARY VOCATIONAL/TECHNICAL/COMMERCIAL': 10, 'TEACHER TRAINING': 11,
       'OTHER': 12, 'OTHER RELIGIOUS': 13, 'LOWER/UPPER 6': 14, 'ADULT EDUCATION': 15,
       'SECONDARY VOCATIONAL/TECHNICAL/COMMERCIAL': 16, 'NURSERY': 17}
    level_of_education = st.selectbox("Level of Education", list(education_map.keys()))
    level_of_education_value = education_map[level_of_education]

    sector_map = {'rural': 0, 'urban': 1}
    sector = st.selectbox("sector", list(sector_map.keys()))
    sector_value = sector_map[sector]

    women_access = st.selectbox("women_access", [0, 1])

# Model Selection
st.header("Model Selection")
selected_model = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
if st.button("Predict"):
    user_data = pd.DataFrame({"age": [age], "years_lived_in_community": [years_lived_in_community], "level_of_education": [level_of_education_value], 
                              "phone_access": [phone_access], "sector": [sector_value], "women_access": [women_access]})
    if selected_model == "Logistic Regression":
        prediction = models['logistic_regression'].predict(user_data)
    elif selected_model == "Decision Tree":
        prediction = models['decision_tree'].predict(user_data)
    else:
        prediction = models['random_forest'].predict(user_data)
    st.write(f"{selected_model}Loan Approval Prediction:", prediction)
    
st.markdown("---")
st.markdown("<center>Made by Team Numerixa</center>", unsafe_allow_html=True)