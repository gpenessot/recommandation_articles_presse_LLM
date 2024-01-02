#draft #1

import streamlit as st
import transformers
from transformers import AutoModel, AutoTokenizer
from qdrant.client import QdrantClient
import plotly.express as px
from sklearn.manifold import TSNE
import numpy as np

# Load pre-trained transformer model and tokenizer
model_name = "bert-base-uncased"  # Change this to your desired transformer model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Qdrant Client
qdrant_address = "http://localhost:6333"  # Change this to your Qdrant server address
client = QdrantClient(qdrant_address)

# Function to encode text with the transformer model
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Function to calculate distance between vectors
def calculate_distance(vector1, vector2):
    return np.linalg.norm(vector1 - vector2)

# Streamlit App
st.title("Vector Distance Calculator")

# Input text box
query_text = st.text_input("Enter a text:", "your text here")

# Encode the query text
query_vector = encode_text(query_text)

# Get vectors from the Qdrant database
database_vectors = client.get_all_ids_and_vectors()

# Calculate distances and find the closest vectors
distances = [calculate_distance(query_vector, vec) for vec in database_vectors.values()]
closest_indices = np.argsort(distances)[:5]  # Display top 5 closest vectors

# Display closest vectors
st.subheader("Closest Vectors:")
for index in closest_indices:
    st.write(f"Vector ID: {index}, Distance: {distances[index]}")

# Plot vector embedding space using TSNE for 2D representation
vector_array = np.array(list(database_vectors.values()))
tsne_model = TSNE(n_components=2, random_state=42)
vector_2d = tsne_model.fit_transform(vector_array)

# Plotly scatter plot
fig = px.scatter(x=vector_2d[:, 0], y=vector_2d[:, 1], text=list(database_vectors.keys()))
fig.update_traces(textposition='top center')
fig.update_layout(title_text='Vector Embedding Space (2D)')
st.plotly_chart(fig)
