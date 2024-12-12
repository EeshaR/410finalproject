# 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
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import openai

# Function to read CSV
def read_csv(file_path):
    return pd.read_csv(file_path)

# Function to preprocess data
def preprocess_data(data):
    # Cleaning data and filling missing values
    data = data.fillna("")
    data["Combined_Info"] = (
        data["Name"] + " " + data["Description"] + " " + data["Schedule Information"]
    ).str.lower()
    return data

# Function to apply TF-IDF vectorization
def compute_tfidf(data):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(data["Combined_Info"])
    return vectorizer, tfidf_matrix

# Function to apply BM25 encoding
def compute_bm25(data):
    tokenized_corpus = [doc.split() for doc in data["Combined_Info"]]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# Function to get BM25 results
def get_bm25_results(bm25, tokenized_corpus, query, data):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    return data.iloc[top_indices]

# Function to get TF-IDF results
def get_tfidf_results(vectorizer, tfidf_matrix, query, data):
    query_vec = vectorizer.transform([query])
    scores = (tfidf_matrix * query_vec.T).toarray().ravel()
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    return data.iloc[top_indices]

# Function to query OpenAI
def query_openai(prompt, bm25_context, tfidf_context, api_key):
    openai.api_key = api_key
    combined_context = f"BM25 Results:\n{bm25_context}\nTF-IDF Results:\n{tfidf_context}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a course navigator. Use the BM25 and TF-IDF context and the user query to answer accurately. If you need additional information, validate with your knowledge. If you cannot answer, say 'I don't have enough information. Make sure you prioritize the results of your answer based on how relevant it is - if a course is more relevant to the query, put it first in the ranking. dont tell me to refer to anything. this should be a comprehensive and complete answer'"},
            {"role": "assistant", "content": f"Relevant course context is as follows:\n{combined_context}"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=250,
    )
    return response["choices"][0]["message"]["content"]

# Interactive query function
def interactive_query(data, vectorizer, tfidf_matrix, bm25, openai_api_key):
    print("\nYou can ask questions about the course catalog.")
    print('Type "done" to exit.\n')
    
    tokenized_corpus = [doc.split() for doc in data["Combined_Info"]]
    
    while True:
        user_query = input("Enter your query: ")
        if user_query.lower() == "done":
            print("Exiting. Goodbye!")
            break
        
        # Get BM25 results
        bm25_results = get_bm25_results(bm25, tokenized_corpus, user_query, data)
        bm25_context = bm25_results.to_csv(index=False, header=False)
        
        # Get TF-IDF results
        tfidf_results = get_tfidf_results(vectorizer, tfidf_matrix, user_query, data)
        tfidf_context = tfidf_results.to_csv(index=False, header=False)
        
        # Query OpenAI
        print("\nProcessing your query with OpenAI...")
        openai_response = query_openai(user_query, bm25_context, tfidf_context, openai_api_key)
        
        # Print response
        print("\nOpenAI response:")
        print(openai_response)

if __name__ == "__main__":
    import sys

    file_path = "course-catalog.csv"
    openai_api_key = "YOUR_API_KEY"
    
    if len(sys.argv) != 1:
        print("Usage: python main.py")
        sys.exit(1)
    
    # file_path = sys.argv[1]
    # openai_api_key = sys.argv[2]
    
    # Process the CSV data
    data = preprocess_data(read_csv(file_path))
    vectorizer, tfidf_matrix = compute_tfidf(data)
    bm25 = compute_bm25(data)
    
    # Start interactive querying
    interactive_query(data, vectorizer, tfidf_matrix, bm25, openai_api_key)