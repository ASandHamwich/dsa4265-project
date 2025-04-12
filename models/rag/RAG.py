import os
import json
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import faiss

# 1. LOAD FINANCIAL KNOWLEDGE

def load_financial_knowledge(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna('', inplace=True)

    grouped_data = df.groupby('url').agg({
        'bank': 'first',
        'title': 'first',
        'subtitle': lambda x: list(x),
        'header': lambda x: list(x),
        'text': lambda x: list(x),
        'tag': 'first'
    }).reset_index()

    documents = []

    for _, row in grouped_data.iterrows():
        bank = row['bank']
        url = row['url']
        title = row['title']
        subtitles = row['subtitle']
        headers = row['header']
        texts = row['text']
        tag = row['tag']

        document = {
            "bank": bank,
            "url": url,
            "tag": tag,
            "content": {
                "h1": title,
                "h1_text": "",
                "h2": []
            }
        }

        h1_texts = []
        h2_sections = {}
        h2_order = []

        for h2, h3, text in zip(subtitles, headers, texts):
            h2 = h2.strip()
            h3 = h3.strip()
            text = text.strip()

            # Case 1: no h2 and no h3
            if h2 == "" and h3 == "":
                h1_texts.append(text)
            # Case 2: only h3
            elif h2 == "" and h3 != "":
                placeholder_h2 = ""
                if placeholder_h2 not in h2_sections:
                    h2_sections[placeholder_h2] = {"h2_texts": [], "h3": []}
                    h2_order.append(placeholder_h2)
                h2_sections[placeholder_h2]["h3"].append({
                    "h3": h3,
                    "text": text
                })
            else:
                if h2 not in h2_sections:
                    h2_sections[h2] = {"h2_texts": [], "h3": []}
                    h2_order.append(h2)
                # Case 3: only h2
                if h3 == "":
                    h2_sections[h2]["h2_texts"].append(text)
                # Case 4: h2 with h3
                else:
                    h2_sections[h2]["h3"].append({
                        "h3": h3,
                        "text": text
                    })

        # Combine h1 text
        document["content"]["h1_text"] = "\n".join(h1_texts)

        # Assemble h2 content
        for h2 in h2_order:
            h2_title = "" if h2 == "_no_section_title" else h2
            h2_text = "\n".join(h2_sections[h2]["h2_texts"])
            h2_entry = {
                "h2": h2_title,
                "h2_text": h2_text,
                "h3": h2_sections[h2]["h3"]
            }
            document["content"]["h2"].append(h2_entry)

        documents.append(document)

    return documents

# 2. FLATTEN DOCUMENTS TO CHUNKS

def flatten_documents_to_chunks(documents):
    chunks = []
    for doc in documents:
        bank = doc["bank"]
        url = doc["url"]
        tag = doc["tag"]
        h1 = doc["content"]["h1"]
        h1_text = doc["content"]["h1_text"]

        if h1_text.strip():
            chunks.append({
                "text": f"{h1}\n\n{h1_text}",
                "metadata": {"bank": bank, "url": url, "tag": tag, "section": h1}
            })

        for h2_section in doc["content"]["h2"]:
            h2 = h2_section["h2"]
            h2_text = h2_section["h2_text"]
            h2_full = f"{h1} > {h2}" if h2 else h1

            if h2_text.strip():
                chunks.append({
                    "text": f"{h1}\n{h2}\n\n{h2_text}",
                    "metadata": {"bank": bank, "url": url, "tag": tag, "section": h2_full}
                })

            for h3_entry in h2_section["h3"]:
                h3 = h3_entry["h3"]
                h3_text = h3_entry["text"]
                section = f"{h1} > {h2} > {h3}" if h2 else f"{h1} > {h3}"

                chunks.append({
                    "text": f"{h1}\n{h2}\n{h3}\n\n{h3_text}",
                    "metadata": {"bank": bank, "url": url, "tag": tag, "section": section}
                })

    return chunks

# 3. GENERATE EMBEDDINGS AS NUMPY ARRAY

def load_embedding_model(model_type):
    if model_type == "use":
        print("Loading Universal Sentence Encoder...")
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("USE model loaded")
        return model
    elif model_type == "minilm":
        print("Loading MiniLM SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("MiniLM model loaded")
        return model
    else:
        raise ValueError("Unsupported model type")

def generate_embeddings(chunks, model_type, model, batch_size):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if model_type == "use":
            emb = model(batch)
            embeddings.append(emb.numpy())
        elif model_type == "minilm":
            emb = model.encode(batch, convert_to_numpy=True)
            embeddings.append(emb)
        else:
            raise ValueError("Unsupported model type")

    embeddings = np.vstack(embeddings)
    print(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    return embeddings

# 4. BUILD FAISS INDEX

def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

# 5. SEARCH AND RETRIEVE RELEVANT DOCUMENTS WITH HYBRID SEARCH

def search_faiss(index, query, chunks, embed_model, model_type, top_k):
    # Embed query
    if model_type == "use":
        query_emb = embed_model([query]).numpy()
    elif model_type == "minilm":
        query_emb = embed_model.encode([query], convert_to_numpy=True)
    else:
        raise ValueError("Unsupported model type")
    
    faiss.normalize_L2(query_emb)
    
    # Search in the FAISS index
    distances, indices = index.search(query_emb, top_k)
    preliminary_results = []
    
    # # Copy result and attach the distance as score
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        metadata = chunk.get("metadata", {})
        
        result = {
            "text": chunk["text"],
            "metadata": metadata,
            "score": float(distances[0][i])
        }
        preliminary_results.append(result)
    
    # Rerank results
    results = rerank_results(preliminary_results)
    return results

def rerank_results(results):
    return sorted(results, key=lambda x: x["score"], reverse=True)

# 6. PREPARE THE RELEVANT BANKING INFORMATION FOR THE LLM AGENT

def prepare_relevant_banking_info(search_results):
    relevant_info = []
    for result in search_results:
        info = {
            "bank": result["metadata"].get("bank", ""),
            "url": result["metadata"].get("url", ""),
            "tag": result["metadata"].get("tag", ""),
            "section": result["metadata"].get("section", ""),
            "text_snippet": result["text"],
            "score": result["score"]
        }
        relevant_info.append(info)
    return relevant_info

# 7. MAIN FUNCTION

def main(model_type="minilm", batch_size=64, top_k=10, 
         user_query="I eat out and go overseas a lot. What are the best card options for me?", 
         csv_path="../../bank-data/combined_banks.csv"):
    
    documents = load_financial_knowledge(csv_path)
    with open("financial_knowledge.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print("Chunks saved to financial_knowledge.json")

    chunks = flatten_documents_to_chunks(documents)
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Chunks saved to chunks.json")
    
    embed_model = load_embedding_model(model_type)
    embeddings = generate_embeddings(chunks, model_type, embed_model, batch_size)
    
    index = build_faiss_index(embeddings)
    faiss.write_index(index, "financial_knowledge.index")
    print("FAISS index saved to financial_knowledge.index")
    
    search_results = search_faiss(index, user_query, chunks, embed_model, model_type=model_type, top_k=top_k)
    relevant_info = prepare_relevant_banking_info(search_results)
    with open("relevant_banking_info.json", "w", encoding="utf-8") as f:
        json.dump(relevant_info, f, indent=2, ensure_ascii=False)
    print("Relevant banking information saved to relevant_banking_info.json")
    
    return relevant_info

relevant_info = main()