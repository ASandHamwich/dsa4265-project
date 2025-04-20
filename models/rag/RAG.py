import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import faiss
from sklearn.preprocessing import minmax_scale
from sentence_transformers import SentenceTransformer, CrossEncoder
import tensorflow_hub as hub
from rank_bm25 import BM25Okapi
from itertools import product

BASE_DIR = os.path.dirname(__file__)

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

def hybrid_search(predicted_tag, faiss_index, bm25_index, query, chunks, embed_model, model_type, cross_encoder_model, top_k, alpha, boost_factor):
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
    
    # Dense search results (with boosting)
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
    
    # BM25 search results (with boosting)
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
    
    reranked_results = rerank_with_cross_encoder(query, hybrid_results, cross_encoder_model)
    final_results = rerank_results(reranked_results)
    
    return final_results

def rerank_with_cross_encoder(query, hybrid_results, cross_encoder_model):
    pairs = [(query, res["text"]) for res in hybrid_results]

    scores = cross_encoder_model.predict(pairs)

    reranked = []
    for res, score in zip(hybrid_results, scores):
        reranked.append({
            "text": res["text"],
            "metadata": res["metadata"],
            "score": float(score)
        })

    return sorted(reranked, key=lambda x: x["score"], reverse=True)

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

# model types = all-MiniLM-L6-v2, multi-qa-mpnet-base-dot-v1, all-mpnet-base-v2, msmarco-MiniLM-L-6-v3
def main(model_type="multi-qa-mpnet-base-dot-v1", batch_size=64, top_k=30, alpha=0.5, boost_factor=0.2,
         user_query="I eat out and go overseas a lot. What are the best card options for me?", 
         csv_path="../../bank-data/combined_banks.csv"):
    
    # documents = load_financial_knowledge(csv_path)
    # with open("financial_knowledge.json", "w", encoding="utf-8") as f:
    #     json.dump(documents, f, ensure_ascii=False, indent=2)
    # print("Documents saved to financial_knowledge.json")

    # chunks = flatten_documents_to_chunks(documents)
    # with open("chunks.json", "w", encoding="utf-8") as f:
    #     json.dump(chunks, f, ensure_ascii=False, indent=2)
    # print("Chunks saved to chunks.json")
    chunks_path = os.path.join(BASE_DIR, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    embed_model = load_embedding_model(model_type)
    embed_model.save('embed_models/multi-qa-mpnet-base-dot-v1')
    # model_path = os.path.join(BASE_DIR, "embed_models/multi-qa-mpnet-base-dot-v1")
    # embed_model = SentenceTransformer(model_path)

    # embeddings = generate_embeddings(chunks, model_type, embed_model, batch_size)
    # faiss_index = build_faiss_index(embeddings)
    # faiss.write_index(faiss_index, "faiss.index")
    # print("FAISS index saved to faiss.index")
    faiss_path = os.path.join(BASE_DIR, "multiqa_faiss.index")
    faiss_index = faiss.read_index(faiss_path)
    
    bm25_index = build_bm25_index(chunks)
    with open("bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    # bm25_path = os.path.join(BASE_DIR, "bm25_index.pkl")
    # with open(bm25_path, "rb") as f:
    #     bm25_index = pickle.load(f)

    predicted_tag, sim_score = predict_tag_from_query(user_query, available_tags, embed_model, model_type)
    print(f"Predicted tag: {predicted_tag} (similarity: {sim_score:.4f})")
    
    search_results = hybrid_search(predicted_tag, faiss_index, bm25_index, user_query, chunks, embed_model, model_type)
    relevant_info = prepare_relevant_banking_info(search_results)
    with open("relevant_banking_info.json", "w", encoding="utf-8") as f:
        json.dump(relevant_info, f, indent=2, ensure_ascii=False)
    print("Relevant banking information saved to relevant_banking_info.json")
    
    return relevant_info

def rag(initial_query, agg_spendings, additional_info=None):
    # Use full paths relative to RAG.py file
    chunks_path = os.path.join(BASE_DIR, "chunks.json")
    faiss_path = os.path.join(BASE_DIR, "multiqa_faiss.index")
    bm25_path = os.path.join(BASE_DIR, "bm25_index.pkl")
    model_path = os.path.join(BASE_DIR, "local_model/multi-qa-mpnet-base-dot-v1")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embed_model = SentenceTransformer(model_path)
    faiss_index = faiss.read_index(faiss_path)

    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)

    predicted_tag, sim_score = predict_tag_from_query(
        user_query=initial_query,
        available_tags=available_tags,
        embed_model=embed_model,
        model_type='multi-qa-mpnet-base-dot-v1'
    )

    print(f"Predicted tag: {predicted_tag} (similarity: {sim_score:.4f})")

    search_results = hybrid_search(
        predicted_tag,
        faiss_index,
        bm25_index,
        initial_query,
        chunks,
        embed_model,
        model_type='multi-qa-mpnet-base-dot-v1'
    )

    relevant_bank_info = prepare_relevant_banking_info(search_results)
    return relevant_bank_info

# 9. MODEL TRAINING

# Helper to run retrieval
def run_retrieval(query, params, pipeline_assets):
    model_type = params['model_type']
    embed_model = pipeline_assets['embed_models'][model_type]
    faiss_index = pipeline_assets['faiss_indexes'][model_type]

    predicted_tag, _ = predict_tag_from_query(
        query, pipeline_assets['available_tags'], embed_model, model_type
    )

    return hybrid_search(
        predicted_tag,
        faiss_index,
        pipeline_assets['bm25_index'],
        query,
        pipeline_assets['chunks'],
        embed_model,
        model_type,
        pipeline_assets['cross_encoder_model'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        boost_factor=params['boost_factor']
    )

# Compute Recall@K and MRR
def evaluate_metrics(params, test_df, pipeline_assets, top_k=10):
    hits = 0
    reciprocals = []

    for _, row in test_df.iterrows():
        results = run_retrieval(row['query'], params, pipeline_assets)

        gold_url     = row['gold_url']
        gold_section = row['gold_section'].lower()

        # Only consider the top K results
        top_k_results = results[:top_k]

        # Find all result positions where:
        #  - URLs match exactly, and
        #  - gold_section is a substring of the returned section (case-insensitive)
        matching_ranks = [
            idx for idx, r in enumerate(top_k_results)
            if (r['metadata']['url'] == gold_url) and
               (gold_section in r['metadata']['section'].lower())
        ]

        # Recall@K: Increment hits if at least one relevant result is found in the top K
        if matching_ranks:
            hits += 1
            # MRR@K: Use the rank of the first match
            rank = matching_ranks[0] + 1
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)

    recall = hits / len(test_df)
    mrr = float(np.mean(reciprocals))
    return recall, mrr

#  First Time Initialization
# documents = load_financial_knowledge("../../bank-data/combined_banks.csv")
# with open("financial_knowledge.json", "w", encoding="utf-8") as f:
#     json.dump(documents, f, ensure_ascii=False, indent=2)
# print("Documents saved to financial_knowledge.json")

# chunks = flatten_documents_to_chunks(documents)
# with open("chunks.json", "w", encoding="utf-8") as f:
#     json.dump(chunks, f, ensure_ascii=False, indent=2)
# print("Chunks saved to chunks.json")

# bm25_index = build_bm25_index(chunks)
# with open("bm25_index.pkl", "wb") as f:
#     pickle.dump(bm25_index, f)
# print("BM25 index saved to bm25_index.pkl")

# cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', token='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')
# cross_encoder.save('cross-encoder/msmarco-MiniLM-L-6-v2')
# print("Cross-encoder model saved.")

# model_types = ['all-MiniLM-L6-v2', 'multi-qa-MiniLM-L6-cos-v1', 'all-mpnet-base-v2', 'msmarco-MiniLM-L-6-v3']
# for model_type in model_types:
#     embed_model = load_embedding_model(model_type)
#     embed_model.save(f'embed_models/{model_type}')
#     print(f"{model_type} model saved.")
    
#     embeddings = generate_embeddings(chunks, model_type, embed_model, batch_size=64)
#     faiss_index = build_faiss_index(embeddings)
#     faiss.write_index(faiss_index, f"faiss_{model_type}.index")
#     print(f"FAISS index saved to faiss_{model_type}.index")

# Subsequent Initialization
chunks = json.load(open('chunks.json', 'r', encoding='utf-8'))
cross_encoder_model = CrossEncoder('cross-encoder/msmarco-MiniLM-L-6-v2')
bm25_index = pickle.load(open('bm25_index.pkl', 'rb'))
embed_models = {
    'all-MiniLM-L6-v2': SentenceTransformer('embed_models/all-MiniLM-L6-v2'),
    'multi-qa-MiniLM-L6-cos-v1': SentenceTransformer('embed_models/multi-qa-MiniLM-L6-cos-v1'),
    'all-mpnet-base-v2': SentenceTransformer('embed_models/all-mpnet-base-v2'),
    'msmarco-MiniLM-L-6-v3': SentenceTransformer('embed_models/msmarco-MiniLM-L-6-v3')
}

faiss_indexes = {}
for model_type in embed_models:
    index_path = f"faiss_{model_type}.index"
    faiss_indexes[model_type] = faiss.read_index(index_path)

# Load test queries + gold annotations
test_df = pd.read_csv('test.csv', encoding='latin1')  # columns: query, gold_url, gold_section

# Define your grid 'all-MiniLM-L6-v2', 
param_grid = {
    'top_k': [5, 10, 20],
    'alpha': [0.3, 0.5, 0.7],
    'boost_factor': [0.0, 0.2],
    'model_type': ['multi-qa-MiniLM-L6-cos-v1', 'all-mpnet-base-v2', 'msmarco-MiniLM-L-6-v3']
}

# Load your pipeline assets once:
pipeline_assets = {
    'chunks': chunks,
    'faiss_indexes': faiss_indexes,
    'bm25_index': bm25_index,
    'cross_encoder_model': cross_encoder_model,
    'available_tags': available_tags,
    'embed_models': embed_models
}

# Grid search
results = []
keys = list(param_grid.keys())
all_combinations = list(product(*[param_grid[k] for k in keys]))
total = len(all_combinations)

for i, values in enumerate(all_combinations, start=1):
    params = dict(zip(keys, values))
    print(f"Evaluating [{i}/{total}]: {params}")
    recall, mrr = evaluate_metrics(params, test_df, pipeline_assets)
    print(f"=> Recall@K={recall:.4f}, MRR={mrr:.4f}")
    results.append({**params, 'Recall@K': recall, 'MRR': mrr})

results_df = pd.DataFrame(results)
results_df.to_csv("retrieval_evaluation_results.csv", index=False)

# Plotting (one plot per metric/param combo, grouped by model_type)
for model_type in param_grid['model_type']:
    # A) Recall@K vs top_k (fixed alpha=0.5, boost=0.1)
    subs = results_df[(results_df.alpha == 0.5) & 
                      (results_df.boost_factor == 0.1) & 
                      (results_df.model_type == model_type)]
    plt.figure()
    plt.plot(subs.top_k, subs['Recall@K'], marker='o')
    plt.title(f'Recall@K vs top_k ({model_type})')
    plt.xlabel('top_k'); plt.ylabel('Recall@K')
    plt.savefig(f'/mnt/data/recall_vs_topk_{model_type}.png', bbox_inches='tight')
    plt.close()

    # B) MRR vs alpha (fixed top_k=20, boost=0.1)
    subs = results_df[(results_df.top_k == 20) & 
                      (results_df.boost_factor == 0.1) & 
                      (results_df.model_type == model_type)]
    plt.figure()
    plt.plot(subs.alpha, subs['MRR'], marker='o')
    plt.title(f'MRR vs alpha ({model_type})')
    plt.xlabel('alpha'); plt.ylabel('MRR')
    plt.savefig(f'/mnt/data/mrr_vs_alpha_{model_type}.png', bbox_inches='tight')
    plt.close()

    # C) Recall@K vs boost_factor (fixed top_k=20, alpha=0.5)
    subs = results_df[(results_df.top_k == 20) & 
                      (results_df.alpha == 0.5) & 
                      (results_df.model_type == model_type)]
    plt.figure()
    plt.plot(subs.boost_factor, subs['Recall@K'], marker='o')
    plt.title(f'Recall@K vs boost_factor ({model_type})')
    plt.xlabel('boost_factor'); plt.ylabel('Recall@K')
    plt.savefig(f'/mnt/data/recall_vs_boost_{model_type}.png', bbox_inches='tight')
    plt.close()

print("All plots saved to /mnt/data/ as PNGs grouped by embedding model.")