# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 14:12:57 2024

@author: Yash
"""

import numpy as np
import pickle
import streamlit as st
st.set_page_config(layout="wide")

loaded_model = pickle.load(open("trained_model.sav","rb"))
loaded_encoder = pickle.load(open('encoder.sav',"rb"))

def disease_prediction(input_data):
    input_data_reshaped = input_data.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    disease = loaded_encoder.inverse_transform(prediction.reshape(1,-1))
    return disease[0][0]

def about_section():
    st.markdown("## About Symptom Predictor")
    st.write("Welcome to Symptom Predictor, a web application designed to provide a preliminary prediction of potential diseases based on user-provided symptoms. This project leverages machine learning techniques to analyze symptom data and predict likely diseases. It's important to note that Symptom Predictor is a demonstration project and should not be considered a substitute for professional medical advice.")

    st.markdown("### Data Source")
    st.write("The model underlying Symptom Predictor is trained on a dataset available on Kaggle, specifically the 'Disease Prediction Data' dataset by Marslino Edward. You can find the dataset [here](https://www.kaggle.com/datasets/marslinoedward/disease-prediction-data?resource=download).")

    st.markdown("### Model")
    st.write("Symptom Predictor utilizes an Extra Trees Regressor from the scikit-learn library for disease prediction. The model is trained on a diverse set of symptom patterns from the provided dataset, aiming to make predictions based on user-input symptoms.The model has an F1 score of 97% on test data")

    st.markdown("### Disclaimer")
    st.write("**Important: Symptom Predictor is not a substitute for professional medical advice, diagnosis, or treatment.** This web application is designed for educational and illustrative purposes only. It does not replace the expertise and judgment of healthcare professionals. Always consult with a qualified healthcare provider for accurate medical advice and diagnosis.")
    st.write("Remember that health conditions can be complex, and accurate diagnosis requires a thorough examination by a trained medical professional. Symptom Predictor serves as a demonstration of machine learning capabilities and is not intended for real-world medical decision-making.")


def main():
    st.title('Prognosis Prediction App')
    st.write('---')
    about_section()
    
    st.write('---')
    st.subheader('Symptoms that you are showiing...')
    
    #gettiong the data
    symptoms = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing",
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity",
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition",
    "spotting_urination", "fatigue", "weight_gain", "anxiety",
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough",
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration",
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea",
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation",
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload",
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation",
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
    "chest_pain", "weakness_in_limbs", "fast_heart_rate",
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool",
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising",
    "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes",
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips",
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness",
    "stiff_neck", "swelling_joints", "movement_stiffness",
    "spinning_movements", "loss_of_balance", "unsteadiness",
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort",
    "foul_smell_of_urine", "continuous_feel_of_urine", "passage_of_gases",
    "internal_itching", "toxic_look_(typhos)", "depression", "irritability",
    "muscle_pain", "altered_sensorium", "red_spots_over_body", "belly_pain",
    "abnormal_menstruation", "dischromic_patches", "watering_from_eyes",
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum",
    "rusty_sputum", "lack_of_concentration", "visual_disturbances",
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma",
    "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption",
    "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf", "palpitations",
    "painful_walking", "pus_filled_pimples", "blackheads", "scurring",
    "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
    ]

    symptom_values = np.zeros(len(symptoms), dtype=int)

    # Number of checkboxes per line
    checkboxes_per_line = 8

    # Calculate the number of rows
    num_rows = len(symptoms) // checkboxes_per_line + (len(symptoms) % checkboxes_per_line > 0)

    # Display checkboxes in columns
    for row in range(num_rows):
        cols = st.columns(checkboxes_per_line)
        for col, symptom_idx in zip(cols, range(row * checkboxes_per_line, (row + 1) * checkboxes_per_line)):
            if symptom_idx < len(symptoms):
                checkbox_key = f"{symptoms[symptom_idx]}_checkbox_{symptom_idx}"  # Create a unique key for each checkbox
                symptom_values[symptom_idx] = col.checkbox(symptoms[symptom_idx], key=checkbox_key)

    # Display the NumPy array
    #st.write("Symptom Values:", symptom_values)

    
    prog = ""
    if st.button("Get Prognosis"):
        prog = disease_prediction(symptom_values)
        
    st.success(prog)
    
    
if __name__ == '__main__':
    main()
