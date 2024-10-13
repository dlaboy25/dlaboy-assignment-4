import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

class LSAModel:
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.vectorizer = None
        self.svd_model = None
        self.normalizer = None
        self.lsa = None
        self.document_vectors = None
        self.documents = None

    def load_data(self):
        print("Loading data...")
        newsgroups = fetch_20newsgroups(subset='all')
        self.documents = newsgroups.data
        print("Data loaded.")

    def fit(self):
        print("Fitting the model...")
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X = self.vectorizer.fit_transform(self.documents)
        self.svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.normalizer = Normalizer(copy=False)
        self.lsa = make_pipeline(self.svd_model, self.normalizer)
        self.document_vectors = self.lsa.fit_transform(X)
        print("Model fitted.")

    def process_query(self, query):
        query_vec = self.vectorizer.transform([query])
        query_vec_lsa = self.lsa.transform(query_vec)
        cosine_similarities = np.dot(self.document_vectors, query_vec_lsa.T).flatten()
        top_indices = cosine_similarities.argsort()[::-1][:5]
        top_docs = [(self.documents[i], cosine_similarities[i]) for i in top_indices]
        return top_docs
