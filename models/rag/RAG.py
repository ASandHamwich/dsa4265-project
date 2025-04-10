import os
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import faiss

def load_financial_knowledge(csv_path):
    df = pd.read_csv(csv_path)

    # Group by URL
    grouped_data = df.groupby('url').agg({
        'title': 'first',
        'subtitle': lambda x: list(x),
        'header': lambda x: list(x),
        'text': lambda x: list(x),
        'tag': lambda x: list(x)
    }).reset_index()

    # Now process the data into a list of documents with hierarchical structure
    documents = []
    for _, row in grouped_data.iterrows():
        url = row['url']
        title = row['title']
        subtitle = row['subtitle']
        headers = row['header']
        texts = row['text']
        tags = row['tag']
        
        # Build the structured text document combining title, subtitle, headers, and text
        full_text = f"{title}\n{subtitle}\n"
        
        for header, text in zip(headers, texts):
            full_text += f"{header}\n{text}\n"
        
        # Add to the list of documents
        documents.append({
            "url": url,
            "text": full_text.strip(),
            "tags": tags
        })

    return documents

