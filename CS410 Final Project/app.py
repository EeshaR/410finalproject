from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import openai
import os

app = Flask(__name__)

# Load and preprocess data
def preprocess_data(data):
    data = data.fillna("")
    data["Combined_Info"] = (
        data["Name"] + " " + data["Description"] + " " + data["Schedule Information"]
    ).str.lower()
    return data

def compute_tfidf(data):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["Combined_Info"])
    return vectorizer, tfidf_matrix

def compute_bm25(data):
    tokenized_corpus = [doc.split() for doc in data["Combined_Info"]]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

def get_bm25_results(bm25, tokenized_corpus, query, data):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    return data.iloc[top_indices]

def get_tfidf_results(vectorizer, tfidf_matrix, query, data):
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray().ravel()
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    return data.iloc[top_indices]

def query_openai(prompt, bm25_context, tfidf_context, api_key):
    openai.api_key = api_key
    combined_context = f"BM25 Results:\n{bm25_context}\nTF-IDF Results:\n{tfidf_context}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a course navigator. Use the BM25 and TF-IDF context and the user query to answer accurately. If you need additional information, validate with your knowledge. If you cannot answer, say 'I don't have enough information.' Prioritize results based on relevance and provide a complete answer."},
            {"role": "assistant", "content": f"Relevant course context is as follows:\n{combined_context}"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=250,
    )
    return response["choices"][0]["message"]["content"]

# Load data and initialize models
file_path = "course-catalog.csv"
data = pd.read_csv(file_path)
data = preprocess_data(data)
vectorizer, tfidf_matrix = compute_tfidf(data)
bm25 = compute_bm25(data)
tokenized_corpus = [doc.split() for doc in data["Combined_Info"]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    # Retrieve results
    bm25_results = get_bm25_results(bm25, tokenized_corpus, user_query, data)
    tfidf_results = get_tfidf_results(vectorizer, tfidf_matrix, user_query, data)
    bm25_context = bm25_results.to_csv(index=False, header=False)
    tfidf_context = tfidf_results.to_csv(index=False, header=False)
    
    openai_api_key = "OUR_API_KEY"

    # Query OpenAI
    openai_response = query_openai(user_query, bm25_context, tfidf_context, openai_api_key)
    
    return jsonify({'response': openai_response})


if __name__ == "__main__":
    app.run(debug=True)





