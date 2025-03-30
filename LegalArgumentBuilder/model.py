from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load BERT-based legal NER model
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Load Sentence-BERT for case similarity retrieval
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example legal case database
case_texts = [
    "Landlord filed an eviction case due to non-payment.",
    "Tenant violated the lease agreement.",
    "Contract breach due to non-performance of payment."
]
case_embeddings = sbert_model.encode(case_texts)

# Create FAISS index for fast retrieval
index = faiss.IndexFlatL2(case_embeddings.shape[1])
index.add(np.array(case_embeddings))

# Load BART for argument generation
bart_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")

def generate_legal_argument(user_text):
    # Extract legal entities
    entities = ner_pipeline(user_text)
    
    # Retrieve similar cases
    query_embedding = sbert_model.encode([user_text])
    D, I = index.search(np.array(query_embedding), k=1)
    similar_case = case_texts[I[0][0]]

    # Generate legal argument
    input_text = f"Case: {user_text} | Related Case: {similar_case}. Generate a strong legal argument:"
    generated_text = bart_pipeline(input_text, max_length=200, num_return_sequences=1)
    
    return generated_text[0]["generated_text"]
