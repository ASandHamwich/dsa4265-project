import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import minmax_scale

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow_hub as hub

# 1. LOAD FINANCIAL KNOWLEDGE

def load_financial_knowledge(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna('', inplace=True)

    grouped_data = df.groupby('url').agg({
        'bank': 'first',
        'title': 'first',
        'subtitle': lambda x: list(x),
        'text': lambda x: list(x),
        'tag': 'first'
    }).reset_index()

    documents = []

    for _, row in grouped_data.iterrows():
        bank = row['bank']
        url = row['url']
        title = row['title']
        subtitles = row['subtitle']
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

        for h2, text in zip(subtitles, texts):
            h2 = h2.strip()
            text = text.strip()

            if h2 == "":
                h1_texts.append(text)
            else:
                if h2 not in h2_sections:
                    h2_sections[h2] = []
                    h2_order.append(h2)
                h2_sections[h2].append(text)

        document["content"]["h1_text"] = "\n".join(h1_texts)

        for h2 in h2_order:
            h2_entry = {
                "h2": h2,
                "h2_text": "\n\n".join(h2_sections[h2])
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

        for h2_entry in doc["content"]["h2"]:
            h2 = h2_entry["h2"]
            h2_text = h2_entry["h2_text"]
            h2_full = f"{h1} > {h2}" if h2 else h1

            if h2_text.strip():
                chunks.append({
                    "text": f"{h1}\n{h2}\n\n{h2_text}",
                    "metadata": {"bank": bank, "url": url, "tag": tag, "section": h2_full}
                })

    return chunks

# 3. GENERATE EMBEDDINGS AS NUMPY ARRAY

def load_embedding_model(model_type):
    if model_type == "use":
        print("Loading Universal Sentence Encoder...")
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        print("USE model loaded")
        return model
    else:
        print(f"Loading {model_type} SentenceTransformer model...")
        model = SentenceTransformer(model_type)
        print(f"{model_type} model loaded")
        return model

def generate_embeddings(chunks, model_type, model, batch_size):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if model_type == "use":
            emb = model(batch)
            embeddings.append(emb.numpy())
        else:
            emb = model.encode(batch, convert_to_numpy=True)
            embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    print(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    return embeddings

# 4. BUILD INDEXES

def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

def build_bm25_index(chunks):
    return BM25Okapi([chunk["text"].lower().split() for chunk in chunks])

# 5. SEARCH AND RETRIEVE RELEVANT DOCUMENTS WITH HYBRID SEARCH

def search_faiss(index, query, chunks, embed_model, model_type, top_k):
    # Embed query
    if model_type == "use":
        query_emb = embed_model([query]).numpy()
    else:
        query_emb = embed_model.encode([query], convert_to_numpy=True)
    
    faiss.normalize_L2(query_emb)
    
    # Dense semantic search the FAISS index
    distances, indices = index.search(query_emb, top_k)
    preliminary_results = []
    
    # Copy result and attach the distance as score
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

def hybrid_search(faiss_index, bm25_index, query, chunks, embed_model, model_type, top_k=10, alpha=0.5, boost_factor=0.1, predicted_tag=None):
    # Dense embedding of query
    if model_type == "use":
        query_emb = embed_model([query]).numpy()
    else:
        query_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    # Dense semantic search the FAISS index
    dense_scores, dense_indices = faiss_index.search(query_emb, top_k)
    # Sparse BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    # Normalize scores
    dense_scores_scaled = minmax_scale(dense_scores[0])
    bm25_scores_scaled = minmax_scale([bm25_scores[i] for i in top_bm25_indices])
    # Score Fusion with metadata boosting
    hybrid_results = []
    seen_indices = set()
    for i, idx in enumerate(dense_indices[0]):
        chunk = chunks[idx]
        score = alpha * dense_scores_scaled[i]
        # Boost if predicted_tag is found in the chunk's tag (case insensitive)
        if predicted_tag and predicted_tag.lower() in chunk["metadata"]["tag"].lower():
            score += boost_factor
        hybrid_results.append({
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "score": score
        })
        seen_indices.add(idx)
    for i, idx in enumerate(top_bm25_indices):
        if idx in seen_indices:
            continue
        chunk = chunks[idx]
        score = (1 - alpha) * bm25_scores_scaled[i]
        if predicted_tag and predicted_tag.lower() in chunk["metadata"]["tag"].lower():
            score += boost_factor
        hybrid_results.append({
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "score": score
        })
    hybrid_results = sorted(hybrid_results, key=lambda x: x["score"], reverse=True)[:top_k]
    final_results = rerank_results(hybrid_results)
    return final_results

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
            "score": float(result["score"])
        }
        relevant_info.append(info)
    return relevant_info

# 7. METADATA BOOSTING
available_tags = [
    "account promotions", "accounts", "atm and branch services",
    "card promotions", "card rewards and services", "credit cards",
    "debit cards", "digital services", "insurance", "investments",
    "loans", "optimise savings", "payments and transfers"
]

def predict_tag_from_query(query, available_tags, embed_model, model_type):
    # Lower-case versions for comparison
    tags_clean = [tag.lower() for tag in available_tags]
    query_clean = query.lower()
    if model_type == "use":
        tag_embeddings = embed_model(tags_clean).numpy()
        query_embedding = embed_model([query_clean]).numpy()[0]
    else:
        tag_embeddings = embed_model.encode(tags_clean, convert_to_numpy=True)
        query_embedding = embed_model.encode([query_clean], convert_to_numpy=True)[0]
    # Compute cosine similarity
    similarities = (tag_embeddings @ query_embedding) / (
        np.linalg.norm(tag_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    top_index = np.argmax(similarities)
    return available_tags[top_index], similarities[top_index]

# 8. MAIN FUNCTION

# all-MiniLM-L6-v2, multi-qa-mpnet-base-dot-v1, all-mpnet-base-v2, msmarco-MiniLM-L-6-v3
def rag(model_type="multi-qa-mpnet-base-dot-v1", batch_size=64, top_k=30, alpha=0.5, boost_factor=0.2,
         user_query="I eat out and go overseas a lot. What are the best card options for me?", 
         csv_path="../../bank-data/combined_banks.csv"):
    
    documents = load_financial_knowledge(csv_path)
    with open("financial_knowledge.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    print("Documents saved to financial_knowledge.json")

    chunks = flatten_documents_to_chunks(documents)
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Chunks saved to chunks.json")
    # with open("chunks.json", "r", encoding="utf-8") as f:
    #     chunks = json.load(f)
    
    embed_model = load_embedding_model(model_type)
    embeddings = generate_embeddings(chunks, model_type, embed_model, batch_size)
    
    faiss_index = build_faiss_index(embeddings)
    faiss.write_index(faiss_index, "faiss.index")
    print("FAISS index saved to faiss.index")
    # faiss_index = faiss.read_index("multiqa_faiss.index")
    
    bm25_index = build_bm25_index(chunks)

    predicted_tag, sim_score = predict_tag_from_query(user_query, available_tags, embed_model, model_type)
    print(f"Predicted tag: {predicted_tag} (similarity: {sim_score:.4f})")
    
    search_results = hybrid_search(faiss_index, bm25_index, user_query, chunks,
                                   embed_model, model_type, top_k, alpha, boost_factor, predicted_tag)
    
    relevant_info = prepare_relevant_banking_info(search_results)
    with open("relevant_banking_info.json", "w", encoding="utf-8") as f:
        json.dump(relevant_info, f, indent=2, ensure_ascii=False)
    print("Relevant banking information saved to relevant_banking_info.json")
    
    return relevant_info

relevant_info = rag()