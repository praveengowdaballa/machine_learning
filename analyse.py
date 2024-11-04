"""
This module performs clustering analysis on incident descriptions from an Excel file.
It uses sentence embeddings and DBSCAN clustering to identify similar descriptions and
groups them by cluster. The analysis results are saved to a new Excel file, and pie charts 
are generated for the distribution of Incident IDs within each cluster.

Note : Cosine similarity measures the similarity between two vectors of an inner product space.
It is measured by the cosine of the angle between two vectors and determines 
whether two vectors are pointing in roughly the same direction.
It is often used to measure document similarity in text analysis

DBSCAN:
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 
finds core samples in regions of high density and expands clusters from them. 
This algorithm is good for data which contains clusters of similar density.

SentenceTransformer:
SentenceTransformer is a Python library for state-of-the-art sentence embeddings.
It is designed to work with transformer models, which have shown strong performance on
natural language understanding and generation tasks.
The 'paraphrase-MiniLM-L6-v2' model is used here, which is a sentence embedding model
trained on a large dataset for paraphrase identification.
This model captures the semantic meaning of sentences, making it suitable for tasks
like clustering similar descriptions.

vectorizeing

BERT:
BERT language model is an open source machine learning framework for natural language 
processing (NLP). BERT is designed to help computers understand the meaning of ambiguous
language in text by using surrounding text to establish context. The BERT framework
was pretrained using text from Wikipedia and can be fine-tuned with question-and-answer data sets.

BERT, which stands for Bidirectional Encoder Representations from Transformers,
is based on transformers, a deep learning model in which every output element 
is connected to every input element, and the weightings between them are dynamically 
calculated based upon their connection.

Cosine similarity measures the similarity between two vectors of an inner product space. 
It is measured by the cosine of the angle between two vectors and determines whether 
two vectors are pointing in roughly the same direction. 
It is often used to measure document similarity in text analysis.

"""
import base64
from io import BytesIO
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


FILE_NAME = 'pattern.xlsx'
df = pd.read_excel(FILE_NAME)

# Initialize the sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode descriptions into embeddings
descriptions = df['Activity Description'].astype(str).tolist()
embeddings = model.encode(descriptions, convert_to_tensor=False)

# Apply DBSCAN clustering on embeddings
clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings)

# Add cluster labels to the DataFrame
df['Cluster'] = clustering.labels_

# Sort by cluster for easier analysis
df_sorted = df.sort_values(by=['Cluster'])

EXCEL_OUTPUT_FILE = 'pattern_analysis_results.xlsx'
CSV_OUTPUT_FILE = 'pattern_analysis_results.csv'
df_sorted.to_excel(EXCEL_OUTPUT_FILE, index=False)
df_sorted.to_csv(CSV_OUTPUT_FILE, index=False)
print(f"Analysis complete. Results saved to {EXCEL_OUTPUT_FILE} and {CSV_OUTPUT_FILE}")

# Group by clusters and Incident ID to get counts per cluster
clustered_data = df_sorted.groupby(['Cluster', 'Incident ID']).size().reset_index(name='Count')

# Filter out noise points (DBSCAN labels them as -1)
valid_clusters = clustered_data[clustered_data['Cluster'] != -1]

HTML_OUTPUT = 'cluster_analysis.html'
with open(HTML_OUTPUT, 'w') as f:
    f.write("<html><head><title>Cluster Analysis</title></head><body>")
    f.write("<h1>Cluster Analysis Report</h1>")
    # Generate and embed pie charts for each cluster
    for cluster_id, cluster_df in valid_clusters.groupby('Cluster'):
        # Plot pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(cluster_df['Count'], labels=cluster_df['Incident ID'], autopct='%1.1f%%', startangle=140)
        plt.title(f'Incident ID Distribution for Cluster {cluster_id}')
        # Save chart to a buffer and convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()  # Close the figure to free up memory
        # Embed chart in HTML
        f.write(f"<h2>Cluster {cluster_id}</h2>")
        f.write(f'<img src="data:image/png;base64,{img_base64}"/>')

    f.write("</body></html>")

print(f"HTML report generated: {HTML_OUTPUT}")
