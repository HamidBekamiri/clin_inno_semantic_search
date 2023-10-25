# streamlit_app.py
import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import pickle
import numpy as np

import os
import importlib

sentence_transformers_version = st.sidebar.text_input("sentence-transformers version:", "2.1.0")
huggingface_hub_version = st.sidebar.text_input("huggingface_hub version:", "0.0.17")

# Button to install selected versions
if st.sidebar.button("Install/Update Libraries"):
    os.system(f"pip install sentence-transformers=={sentence_transformers_version}")
    os.system(f"pip install huggingface_hub=={huggingface_hub_version}")
    # Reload modules if necessary (not always reliable, restarting the app might be needed)
    importlib.reload(importlib.import_module("sentence_transformers"))
    importlib.reload(importlib.import_module("huggingface_hub"))
    st.sidebar.text("Libraries updated!")

# Load sentences & embeddings from disc
with open('clinical_inno_embeddings_masterid_paraphrase-multilingual-mpnet-base-v2.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_masterid = stored_data['pro_master_id']
    stored_products = stored_data['products']
    stored_embeddings = stored_data['embeddings']

# Initialize the SentenceTransformer model
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

def get_similar_products(query, products, mean_embeddings_tensor, top_k=10):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, mean_embeddings_tensor)
    top_results = torch.topk(cos_scores, k=top_k)
    
    similar_products = [(products[idx], score.item()) for score, idx in zip(top_results[0], top_results[1])]
    return similar_products

# Create a DataFrame using the loaded data
df = pd.DataFrame({
    'master_id': stored_masterid,
    'product': stored_products,
    'embeddings': list(stored_embeddings)
})

def compute_mean_embedding(group):
    embeddings = np.array(group['embeddings'].tolist())
    return torch.tensor(np.mean(embeddings, axis=0))

# Group by 'master_id' column and compute the mean embeddings
mean_embeddings = df.groupby('master_id').apply(compute_mean_embedding)

# Convert the Series of tensors to a single tensor
mean_embeddings_tensor = torch.stack(mean_embeddings.tolist())

# Streamlit UI
st.title("Product Similarity Finder")

# User input
user_query = st.text_area("Enter your query:", "SuperSole Bred læst er en utrolig...")

if st.button("Find Similar Products"):
    results = get_similar_products(user_query, stored_products, mean_embeddings_tensor)
    for product, score in results:
        st.write(f"Product: {product} (Score: {score:.4f})")
