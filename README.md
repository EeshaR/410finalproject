# 410finalproject

To activate the virtual environment:

`python3 -m venv venv`
`source venv/bin/activate`

Then install the required dependencies:
`pip install flask`
`pip install pandas`
`pip install scikit-learn`
`pip install rank_bm25`
`pip install openai==0.28`

To remove cache if you're running into any problems, you can run: `rm -rf __pycache__`
Then run `python app.py` to run the application.

MAKE SURE TO REPLACE openai_api_key WITH THE OPENAI API KEY. 