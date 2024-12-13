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
    data["Prerequisites"] = data["Description"].str.extract(r"Prerequisite: (.+?)[\.|,]")  # Extract prerequisites
    data["Prerequisites"] = data["Prerequisites"].fillna("").str.lower()
    return data

# referenced - https://dev.to/dcs_ink/how-to-set-up-the-openai-api-with-python-and-flask-2120?utm_source=chatgpt.com
def filter_by_prerequisites(data, prerequisite_query):
    prerequisite_query = prerequisite_query.lower()
    filtered_data = data[data["Prerequisites"].str.contains(prerequisite_query, na=False)]
    return filtered_data

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
# referenced - https://cookbook.openai.com/examples/how_to_stream_completions?utm_source=chatgpt.com
file_path = "course-catalog.csv"
data = pd.read_csv(file_path)
data = preprocess_data(data)
vectorizer, tfidf_matrix = compute_tfidf(data)
bm25 = compute_bm25(data)
tokenized_corpus = [doc.split() for doc in data["Combined_Info"]]

@app.route('/')
def index():
    return render_template('index.html')

# referenced - https://community.openai.com/t/chat-history-with-functions-enabled-example/322210?utm_source=chatgpt.com
@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    prerequisite_input = ""
    openai_api_key = "OUR_API_KEY"

    # Check if "prerequisite:" is included
    if "prerequisite:" in user_query.lower():
        parts = user_query.lower().split("prerequisite:")
        user_query = parts[0].strip()  # Extract query portion
        prerequisite_input = parts[1].strip()  # Extract prerequisite portion

    # If only a prerequisite is given
    if not user_query and prerequisite_input:
        filtered_data = filter_by_prerequisites(data, prerequisite_input)
        if filtered_data.empty:
            return jsonify({
                'response': f"No courses found with prerequisite '{prerequisite_input}'.",
                'graphData': {'name': '', 'description': '', 'prerequisites': '', 'related_courses': []}
            })
        
        # Limit response to one or two course names
        courses = filtered_data["Name"].head(2).tolist()
        return jsonify({
            'response': f"Courses requiring '{prerequisite_input}' as a prerequisite: {', '.join(courses)}.",
            'graphData': {'name': '', 'description': '', 'prerequisites': '', 'related_courses': []}
        })

    # If both query and prerequisite are provided
    if user_query and prerequisite_input:
        bm25_results = get_bm25_results(bm25, tokenized_corpus, user_query, data)
        tfidf_results = get_tfidf_results(vectorizer, tfidf_matrix, user_query, data)
        combined_results = pd.concat([bm25_results, tfidf_results]).drop_duplicates()

        if combined_results.empty:
            return jsonify({
                'response': f"No courses found matching preference '{user_query}'.",
                'graphData': {'name': '', 'description': '', 'prerequisites': '', 'related_courses': []}
            })
        
        # Check if any of the courses have the given prerequisite
        filtered_courses = combined_results[combined_results["Prerequisites"].str.contains(prerequisite_input, na=False)]
        if filtered_courses.empty:
            return jsonify({
                'response': f"None of the courses matching '{user_query}' have '{prerequisite_input}' as a prerequisite.",
                'graphData': {'name': '', 'description': '', 'prerequisites': '', 'related_courses': []}
            })

        # Limit response to one or two course names
        matching_courses = filtered_courses["Name"].head(2).tolist()
        return jsonify({
            'response': f"Courses matching '{user_query}' with '{prerequisite_input}' as a prerequisite: {', '.join(matching_courses)}.",
            'graphData': {'name': '', 'description': '', 'prerequisites': '', 'related_courses': []}
        })

    # If only a query is given
    if user_query and not prerequisite_input:
        bm25_results = get_bm25_results(bm25, tokenized_corpus, user_query, data)
        tfidf_results = get_tfidf_results(vectorizer, tfidf_matrix, user_query, data)
        combined_results = pd.concat([bm25_results, tfidf_results]).drop_duplicates()

        # Handle no results
        if combined_results.empty:
            return jsonify({
                'response': "No courses found matching your query.",
                'graphData': {'name': '', 'description': '', 'prerequisites': '', 'related_courses': []}
            })

        # Select the top course for graph visualization
        top_course = combined_results.iloc[0]
        top_course_name = top_course["Name"]

        # Extract explicitly related courses (e.g., via exact prerequisites)
        related_courses = data[
            data["Prerequisites"].str.contains(rf"\b{top_course_name}\b", na=False)  # Match exact course name
        ]["Name"].tolist()

        course_metadata = {
            "name": top_course_name,
            "description": top_course["Description"],
            "prerequisites": top_course["Prerequisites"],
            "related_courses": related_courses  # Only include explicitly related courses
        }

        # Convert BM25 and TF-IDF results to OpenAI context
        bm25_context = bm25_results.head(3).to_csv(index=False, header=False)
        tfidf_context = tfidf_results.head(3).to_csv(index=False, header=False)

        # Query OpenAI
        openai_response = query_openai(user_query, bm25_context, tfidf_context, openai_api_key)

        # Return response with graph data
        return jsonify({
            'response': openai_response,
            'data': combined_results.head(3).to_dict('records'),
            'graphData': course_metadata  # Add metadata for graph
        })

    # If neither query nor prerequisite is provided
    return jsonify({
        'response': "How can I assist you today?",
        'graphData': {'name': '', 'description': '', 'prerequisites': '', 'related_courses': []}
    })

if __name__ == "__main__":
    app.run(debug=True)

