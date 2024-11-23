import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Common medical entities
DISEASES =['diabetes', 'cancer' ,'asthma', 'heart disease' , 'covid', 'flu', 'arthritis', 'stroke', 'hypertension','obesity', 'tuberculosis','pneumonia', 'stroke', 'cholesterol', 'parkinsons']
SYMPTOMS = ['fever' , 'headache', 'nausea', 'fatigue', 'cough', 'chills', 'dizziness', 'pain', 'vomiting','sore throat', 'shortness of breath','swelling', 'weight loss', 'itching', 'loss of appetite' ]
TREATMENTS =['medication', 'therapy', 'surgery', 'vaccine', 'radiation', 'insulin', 'chemotherapy', 'dialysis', 'therapy', 'antibiotics', 'painkillers' , 'surgery', 'treatment', 'vaccines']


# Load MedQuAD dataset
file_path = ("notebook\Processed_MedQuAD.csv")

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


#prepare TF-IDF for answer retrieval
@st.cache_resource
def prepare_data(data):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["Question"])
    return vectorizer, tfidf_matrix

# Retrieve the best matching answer
def retrieve_answer(user_query, vectorizer, tfidf_matix, data):
    query_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vec, tfidf_matix)
    best_match_idx = similarity.argmax()
    return data.iloc[best_match_idx]["Answers"]

# Extract entities dynamically from the text

def extract_entities(text, user_query):
    entities = {'diseases': [], 'symptoms': [], 'treatments': []}


    # remove words from the user query itself to avoid self-matching
    query_words = set(re.findall(r'\b\w+\b', user_query.lower()))


    ## Match diseases, symptoms, and treatments in the text using the extended lists
    for word in re.findall(r'\b\w+\b', text.lower()):
        if word in DISEASES and word not in query_words:
            entities['diseases'].append(word)
        elif word in SYMPTOMS and word not in query_words:
            entities['symptoms'].append(word)
        elif word in TREATMENTS and word not in query_words:
            entities['treatments'].append(word)        
    
    return entities






# Streamlit UI

def main():
    st.title("Medical Q&A Chatbot")
    st.write("Ask medical questions, and the bot will provide answer from the MedQuAD dataset.")


    # file path
    file_upload = ("notebook\Processed_MedQuAD.csv")

    if file_upload is not None:
        data = load_data(file_upload)
        if {"Question", "Answers"}.issubset(data.columns):
            vectorizer, tfidf_matrix = prepare_data(data)

            # user input
            user_question = st.text_input("Enter your medical question:")
            if user_question:
                # Retrieve answer from the dataset
                st.subheader("Retrieved Answer from MedQuAD")
                retrieved_answer = retrieve_answer(user_question, vectorizer, tfidf_matrix, data)
                st.write(retrieved_answer)

                st.subheader("Recognized Entities")
                entities_from_answer = extract_entities(retrieved_answer, user_question)
                entities_from_question = extract_entities(user_question, user_question)  


                if any(entities_from_answer.values()) or any(entities_from_question.values()):
                    st.write("### From Retrieved Answer:")
                    for category, items in entities_from_answer.items():
                        if items:
                            st.write(f"- {category.capitalize()}: {', '.join(set(items))}")

                    st.write("### From User Query:")
                    for category, items in entities_from_question.items():
                        if items:
                            st.write(f"- {category.capitalize()}:{', '.join(set(items))}")
                else:
                    st.write("No entities recognized.")

        else:
            st.error("Invalid dataset format. Ensure the dataset has 'question' and 'answer' columns.")  

if __name__ == "__main__":
    main()                                                    

