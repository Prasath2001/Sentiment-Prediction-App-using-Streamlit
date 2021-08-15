# Packages
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

pipeline_lr = joblib.load(open("models/emotion_classifier_pipeline_lr_15_aug_2021.pkl","rb"))

# Functions
def predict_emotions(docx):
    results = pipeline_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipeline_lr.predict_proba([docx])
    return results





def main():

    st.title("Emotion Classifier App")
    menu = ['Home','Monitor','About']
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home-Emotion in Text")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label="Submit")
        
        if submit_text:
            col1,col2 = st.columns(2)
            # Apply function here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence: {}".format(np.max(probability)))
            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability,columns=pipeline_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)
                #st.write(proba_df.T)



    
    elif choice == "Monitor":
        st.subheader("Monitor App")
    
    else:
        st.subheader("About")
    



if __name__ == '__main__':
    main()


