# CS 410: Final Project Overview

## How to run the project: 

To create the virtual environment:

`python3 -m venv venv`

To activate the virtual environment:

`source venv/bin/activate`

Then install the required dependencies:
`pip install flask`
`pip install pandas`
`pip install scikit-learn`
`pip install rank_bm25`
`pip install openai==0.28`

To remove cache if you're running into any problems, you can run: `rm -rf __pycache__`.
Then run `python app.py` to run the application.

Make sure to replace `openai_api_key` with the actual OpenAI API key so that the user can query the LLM. 

## Documentation: 
User standpoint: The user opens our easy-to-use html site, types in a query of their choice regarding uiuc course - about what to take, pre-reqs, credits, relevancy. 
The response is then generated for them in a timely fashion. They are free to make as many queries as they’d like. 

Backend standpoint: We begin scraping the course explorer page for current, up to date information about courses offered. Then we save and parse this data into a csv file. 
Then we use a series of different classification and retrieval methods (bm25, open ai, tdidf) to understand the content of the csv. 
This way when a query is asked, we can retrieve the current and correct information. Finally, we created a user-friendly screen to help enhance the experience. 
We tested this tool on many different classmates, teachers, and faculty - especially during course selection time, they mentioned this tool would be useful for sorting and understanding information.
Here is a walk through of the functionality and purpose of each file 

- app.py: This code creates a Flask web application that allows users to search a course catalog using text-based queries and prerequisites. It employs text preprocessing, TF-IDF, and BM25 for ranking courses by relevance to the user's query, integrates OpenAI's API for refined responses, and provides a web interface for user interaction.
  
- main.py: This script implements a command-line tool for searching a course catalog using natural language queries. It preprocesses the course data, applies TF-IDF and BM25 models for ranking course information, and integrates OpenAI's API to generate comprehensive answers based on the most relevant course context. Users can interactively query the catalog, and the system provides ranked results prioritized by relevance.

- scrape_data.py: This script scrapes course details from the UIUC Course Explorer website and saves the data into a CSV file for easy access and analysis.

- testing.py: This code defines unit tests using Python's unittest framework to validate the functionality of the main module, including reading and preprocessing data, computing TF-IDF and BM25 relevance scores, and querying the OpenAI API, using a sample in-memory CSV dataset for testing. 
