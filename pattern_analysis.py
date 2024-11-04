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

Word2Vec: 

SentenceTransformer:
SentenceTransformer is a Python library for state-of-the-art sentence embeddings.
It is designed to work with transformer models, which have shown strong performance on
natural language understanding and generation tasks.
The 'paraphrase-MiniLM-L6-v2' model is used here, which is a sentence embedding model
trained on a large dataset for paraphrase identification.
This model captures the semantic meaning of sentences, making it suitable for tasks
like clustering similar descriptions.

vectorizing

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
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


# File configuration
FILE_NAME = 'pattern.xlsx'
EXCEL_OUTPUT_FILE = 'pattern_analysis_results.xlsx'
CSV_OUTPUT_FILE = 'pattern_analysis_results.csv'
HTML_OUTPUT = 'cluster_analysis.html'

def load_data(filename: str) -> pd.DataFrame:
    """Load data from an Excel file."""
    try:
        return pd.read_excel(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        exit()

def encode_descriptions(df: pd.DataFrame, model) -> list:
    """Encode descriptions using a sentence transformer model."""
    descriptions = df['Activity Description'].astype(str).tolist()
    return model.encode(descriptions, convert_to_tensor=False)

def cluster_data(embeddings: list, eps: float = 0.5, min_samples: int = 2) -> Tuple[pd.DataFrame, list]:
    """Cluster the embeddings using DBSCAN and return the DataFrame with cluster labels."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    return clustering.labels_

def generate_html_report(df: pd.DataFrame, file_name: str):
    """Generate an HTML report with pie charts for each cluster, using consistent color coding for Incident IDs."""
    # Group by clusters and Incident ID to get counts per cluster
    grouped_data = df.groupby(['Cluster', 'Incident ID']).size().reset_index(name='Count')
    valid_clusters = grouped_data[grouped_data['Cluster'] != -1]

    # Generate a unique color map for each Incident ID
    incident_ids = df['Incident ID'].unique()
    colors = plt.cm.tab20.colors  # Using tab20 colormap for a range of distinct colors
    color_map = {incident_id: colors[i % len(colors)] for i, incident_id in enumerate(incident_ids)}

    with open(file_name, 'w') as f:
        f.write("<html><head><title>Cluster Analysis</title></head><body>")
        f.write("<h1>Cluster Analysis Report</h1>")

        # Generate and embed pie charts for each cluster
        for cluster_id, cluster_df in valid_clusters.groupby('Cluster'):
            # Map colors to Incident IDs
            cluster_colors = [color_map[incident_id] for incident_id in cluster_df['Incident ID']]
            # Plot pie chart for the cluster
            plt.figure(figsize=(8, 6))
            plt.pie(cluster_df['Count'], labels=cluster_df['Incident ID'], autopct='%1.1f%%', startangle=140, colors=cluster_colors)
            plt.title(f'Incident ID Distribution for Cluster {cluster_id}')
            # Convert pie chart to base64 and embed in HTML
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            plt.close()
            f.write(f"<h2>Cluster {cluster_id}</h2>")
            f.write(f'<img src="data:image/png;base64,{img_base64}"/>')
        f.write("</body></html>")
    print(f"HTML report generated: {file_name}")
def main():
    # Load data
    df = load_data(FILE_NAME)

    # Initialize model and encode descriptions
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = encode_descriptions(df, model)

    # Cluster data and add labels
    df['Cluster'] = cluster_data(embeddings)
    df_sorted = df.sort_values(by=['Cluster'])

    # Save to Excel and CSV for analysis
    df_sorted.to_excel(EXCEL_OUTPUT_FILE, index=False)
    df_sorted.to_csv(CSV_OUTPUT_FILE, index=False)
    print(f"Analysis complete. Results saved to {EXCEL_OUTPUT_FILE} and {CSV_OUTPUT_FILE}")

    # Generate HTML report
    generate_html_report(df_sorted, HTML_OUTPUT)

if __name__ == "__main__":
    main()
