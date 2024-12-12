import unittest
import main
import pandas as pd
from io import StringIO

class TestMainFunctions(unittest.TestCase):
    def setUp(self):
        # In-memory CSV for testing
        self.sample_data = StringIO("""Year,Term,YearTerm,Subject,Number,Name,Description,Credit Hours,Section Info,Degree Attributes,Schedule Information,CRN,Section,Status Code,Part of Term,Section Title,Section Credit Hours,Section Status,Enrollment Status,Type,Type Code,Start Time,End Time,Days of Week,Room,Building,Instructors
2024,Fall,2024FA,CS,101,Introduction to Programming,Learn programming basics.,3,Section 1,,,12345,001,A,Full,,3,Open,Available,Lecture,LEC,10:00 AM,11:00 AM,MWF,101,Engineering,Dr. Smith
""")
        self.openai_api_key = "OUR_API_KEY"  # Replace with a valid key if testing OpenAI API

    def test_read_csv(self):
        data = pd.read_csv(self.sample_data)
        self.assertEqual(len(data), 1)
        print("test reading csv")
        self.assertEqual(data.iloc[0]["Name"], "Introduction to Programming")

    def test_preprocess_data(self):
        data = pd.read_csv(self.sample_data)
        processed_data = main.preprocess_data(data)
        self.assertIn("Combined_Info", processed_data.columns)
        print("test preprocess data)")
        self.assertTrue(len(processed_data["Combined_Info"][0]) > 0)

    def test_compute_tfidf(self):
        data = pd.read_csv(self.sample_data)
        processed_data = main.preprocess_data(data)
        vectorizer, tfidf_matrix = main.compute_tfidf(processed_data)
        print("test tfidf")
        self.assertEqual(tfidf_matrix.shape[0], len(processed_data))

    def test_compute_bm25(self):
        data = pd.read_csv(self.sample_data)
        processed_data = main.preprocess_data(data)
        bm25 = main.compute_bm25(processed_data)

        # Tokenize query to match corpus tokenization
        query_tokens = "programming".lower().split()
        tokenized_corpus = [doc.split() for doc in processed_data["Combined_Info"]]

        print("BM25 tokenized query:", query_tokens)
        print("BM25 tokenized corpus:", tokenized_corpus)

        scores = bm25.get_scores(query_tokens)
        print("BM25 scores:", scores)

        self.assertTrue(len(scores) > 0)
        self.assertGreaterEqual(scores[0], 0, "BM25 score should not be negative")

    def test_openai_query(self):
        # This test assumes the OpenAI API key and response
        if self.openai_api_key != "test_api_key":
            data = pd.read_csv(self.sample_data)
            processed_data = main.preprocess_data(data)
            bm25 = main.compute_bm25(processed_data)
            vectorizer, tfidf_matrix = main.compute_tfidf(processed_data)
            
            bm25_results = main.get_bm25_results(bm25, [doc.split() for doc in processed_data["Combined_Info"]], "Test prompt", processed_data)
            tfidf_results = main.get_tfidf_results(vectorizer, tfidf_matrix, "Test prompt", processed_data)

            bm25_context = bm25_results.to_csv(index=False, header=False)
            tfidf_context = tfidf_results.to_csv(index=False, header=False)
            
            response = main.query_openai("Test prompt", bm25_context, tfidf_context, self.openai_api_key)

            print("test openai query")
            self.assertTrue(len(response) > 0)

if __name__ == "__main__":
    unittest.main()