# streamlit_app.py
import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import numpy as np

import os
import importlib

#Load sentences & embeddings from disc
with open('mean_clinical_inno_embeddings_masterid_paraphrase-multilingual-mpnet-base-v2.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_masterid = stored_data['pro_master_id']
    stored_products = stored_data['mean_products']
    stored_embeddings = stored_data['mean_embeddings']

# Initialize the SentenceTransformer model
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def get_similar_products(query, products, mean_embeddings_tensor, top_k=10):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, mean_embeddings_tensor)
    top_results = torch.topk(cos_scores, k=top_k)
    
    similar_products = [(products[idx], score.item()) for score, idx in zip(top_results[0], top_results[1])]
    return similar_products



# Streamlit UI
st.title("Product Similarity Finder")

# User input
user_query = st.text_area("Enter your query:", "SuperSole Bred læst er en utrolig...")

if st.button("Find Similar Products"):
    results = get_similar_products(user_query, stored_products, stored_embeddings)
    for product, score in results:
        st.write(f"Product: {product} (Score: {score:.4f})")
