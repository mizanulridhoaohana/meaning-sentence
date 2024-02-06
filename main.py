import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer

def load_and_predict_model(sentence, selected_model):
    # Load the selected model
    model_path = './model_1.joblib' if selected_model == 'Model 1 (balanced data 200)' else './model_2.joblib'
    clf = joblib.load(model_path)

    # Load IndoBERT model
    model = SentenceTransformer('indobenchmark/indobert-base-p1')

    # Encode the input sentence
    encoded_sentence = model.encode(sentence)

    # Predict the probability of being logical and meaningful
    prob_logical_and_meaningful = clf.predict_proba([encoded_sentence])[:, 1]

    return prob_logical_and_meaningful[0]

# Streamlit UI
st.title("Logical and Meaningful Sentence Checker")

# Radio button for selecting the model
selected_model = st.radio("Select Model:", ('Model 1 (balanced data 200)', 'Model 2 (imbalance, mon-logic only 2 data)'))

# Input text box for user input
user_input = st.text_area("Enter a sentence:")

if st.button("Check"):
    if user_input:
        result = load_and_predict_model(user_input, selected_model)
        st.write(f"Similarity Percentage: {result*100:.2f}%")

        if result>0.5:
            st.success("The sentence is logical and meaningful.")
        else:
            st.error("The sentence is not logical or not meaningful.")
    else:
        st.warning("Please enter a sentence for checking.")
