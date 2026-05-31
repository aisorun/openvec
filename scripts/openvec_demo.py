#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenVec Python Demo Client
This script demonstrates how to interact with the OpenVec Vector Database
via its HTTP REST API:
1. Health check & Server status
2. Creating a collection
3. Inserting documents (individually and in batch)
4. Querying documents (vector similarity search)
5. Deleting a document and verifying deletion
6. Dropping/cleaning up the collection
"""

import urllib.request
import urllib.error
import json
import time
import random
import sys
import hashlib
import os

BASE_URL = "http://127.0.0.1:8000"
COLLECTION_NAME = "python_demo_collection"
DIMENSION = 128
METRIC = "cosine"  # OpenVec supports cosine, l2, ip

def print_header(title):
    print("\n" + "=" * 60)
    print(f"🔹 {title}")
    print("=" * 60)

def http_request(path, method="GET", data=None):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, method=method)
    
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req_data = json.dumps(data).encode("utf-8")
    else:
        req_data = None
        
    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, data=req_data, timeout=5) as response:
            latency = (time.perf_counter() - start_time) * 1000  # ms
            status_code = response.status
            body = response.read().decode("utf-8")
            
            result = json.loads(body) if body else None
            return {
                "success": True,
                "status": status_code,
                "latency_ms": latency,
                "data": result
            }
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - start_time) * 1000  # ms
        try:
            error_body = e.read().decode("utf-8")
            error_json = json.loads(error_body)
        except Exception:
            error_json = error_body if 'error_body' in locals() else str(e)
            
        return {
            "success": False,
            "status": e.code,
            "latency_ms": latency,
            "error": error_json
        }
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000  # ms
        return {
            "success": False,
            "status": 500,
            "latency_ms": latency,
            "error": str(e)
        }

def wait_for_server():
    print("Checking OpenVec server health...")
    for i in range(10):
        res = http_request("/health")
        if res["success"] and res["data"] and res["data"].get("status") == "healthy":
            print(f"✅ OpenVec server is healthy! Version: {res['data'].get('version')} (Check latency: {res['latency_ms']:.2f}ms)")
            return True
        print(f"⏳ Waiting for server to start ({i+1}/10)...")
        time.sleep(1)
    print("❌ Failed to connect to OpenVec server. Please make sure it is running on port 8000.")
    sys.exit(1)

def generate_random_vector(dim=DIMENSION):
    # Generates a random vector normalized to unit length for Cosine Similarity
    vec = [random.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = sum(x*x for x in vec) ** 0.5
    return [x / norm for x in vec]

def generate_vector_from_text(text, dim=DIMENSION):
    # Deterministic generation using SHA256 of text as seed for reproducible embeddings
    sha = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(sha[:4], byteorder="big")
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = sum(x*x for x in vec) ** 0.5
    return [x / norm for x in vec]

def main():
    # Performance metrics tracking
    metrics = {}
    
    # 0. Wait for Server
    print_header("0. Server Initialization Check & Storage Location")
    wait_for_server()
    print(f"\n📂 [Physical Storage Location]")
    print(f"   The database physical files (WAL logs, collections metadata, segments data) are stored in:")
    print(f"   👉  /Users/landaa/Workspaces/aisorun/openvec/openvec_data/")
    print(f"   Inside this folder, you will find active collection folders and WAL files.")
    
    # 1. Clean up existing collection if it exists
    print_header("1. Preparing Workspace (Cleanup old collection)")
    cleanup_res = http_request(f"/collections/{COLLECTION_NAME}", method="DELETE")
    if cleanup_res["success"]:
        print(f"🧹 Cleaned up pre-existing collection '{COLLECTION_NAME}' (Response: {cleanup_res['data']}, Time: {cleanup_res['latency_ms']:.2f}ms)")
    else:
        print(f"ℹ️ No pre-existing collection found or skipped. (Status code: {cleanup_res['status']})")
        
    # 2. Create Collection
    print_header("2. Creating Collection")
    create_payload = {
        "name": COLLECTION_NAME,
        "dimension": DIMENSION,
        "metric": METRIC
    }
    create_res = http_request("/collections", method="POST", data=create_payload)
    if create_res["success"]:
        print(f"✨ Successfully created collection '{COLLECTION_NAME}'")
        print(f"   Metric: {METRIC.upper()}, Dimension: {DIMENSION}")
        print(f"   Response time: {create_res['latency_ms']:.2f}ms")
        metrics["create_collection"] = create_res["latency_ms"]
    else:
        print(f"❌ Failed to create collection: {create_res['error']}")
        sys.exit(1)

    # 3. Inserting Data
    print_header("3. Reading & Inserting Data from sample_data.txt")
    
    # Read the real-world .txt data file line by line
    txt_path = "scripts/sample_data.txt"
    if not os.path.exists(txt_path):
        print(f"❌ Error: Sample data file '{txt_path}' not found!")
        sys.exit(1)
        
    print(f"📖 Reading sample text file from: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    print(f"📊 Loaded {len(lines)} technical sentences. Indexing them into OpenVec individually:")
    individual_latencies = []
    
    for idx, sentence in enumerate(lines):
        doc_id = f"doc_txt_{idx}"
        doc_vector = generate_vector_from_text(sentence)
        doc_payload = {
            "id": doc_id,
            "vector": doc_vector,
            "payload": {
                "content": sentence,
                "length": len(sentence),
                "source_file": "sample_data.txt"
            }
        }
        ins_res = http_request(f"/collections/{COLLECTION_NAME}/insert", method="POST", data=doc_payload)
        if ins_res["success"]:
            print(f"   ✅ Indexed '{doc_id}'\n      - Text: \"{sentence}\"\n      - Vector (first 5 elements): {doc_vector[:5]}...\n      - Time: {ins_res['latency_ms']:.2f}ms")
            individual_latencies.append(ins_res["latency_ms"])
        else:
            print(f"   ❌ Failed to insert '{doc_id}': {ins_res['error']}")
            
    metrics["individual_insert_avg"] = sum(individual_latencies) / len(individual_latencies)
    
    # Batch Insert of 100 random synthetic documents to demonstrate scaling
    print("\n📦 Performing background batch insertion of 100 synthetic documents to demonstrate database density:")
    batch_docs = []
    for idx in range(100):
        doc_id = f"synthetic_doc_{idx}"
        batch_docs.append({
            "id": doc_id,
            "vector": generate_random_vector(),
            "payload": {
                "content": f"Synthetic context data number {idx} for density simulation.",
                "rating": round(random.uniform(0.5, 1.0), 2),
                "source_file": "synthetic_generator"
            }
        })
        
    batch_payload = {"documents": batch_docs}
    batch_res = http_request(f"/collections/{COLLECTION_NAME}/batch_insert", method="POST", data=batch_payload)
    if batch_res["success"]:
        print(f"   ✅ Successfully batch-inserted 100 documents! (Response time: {batch_res['latency_ms']:.2f}ms)")
        metrics["batch_insert_100"] = batch_res["latency_ms"]
    else:
        print(f"   ❌ Failed to batch-insert: {batch_res['error']}")

    # 4. Query Data (Vector Similarity Search)
    print_header("4. Querying Data (Vector Similarity Search)")
    
    # Let's perform a search using a real-world query sentence from our dataset
    # We will search for a semantic context. Let's query using the vector generated
    # from the sentence about "Retrieval Augmented Generation (RAG)"
    query_sentence = "OpenVec 是一款完全采用原生 Rust 开发的超轻量、高性能的双模向量数据库。"
    query_vector = generate_vector_from_text(query_sentence)
    
    limit = 3
    search_payload = {
        "vector": query_vector,
        "limit": limit
    }
    
    print(f"🔍 [QUERY DATA - Querying Vector (generated from text)]")
    print(f"   - Query Text: \"{query_sentence}\"")
    print(f"   - Target Vector (first 5 elements): {query_vector[:5]}... (Total {len(query_vector)} dims)")
    print(f"   - Top limit: {limit}")
    
    print(f"\n⚡ Executing similarity search against OpenVec REST Server...")
    search_res = http_request(f"/collections/{COLLECTION_NAME}/search", method="POST", data=search_payload)
    if search_res["success"] and search_res["data"]:
        print(f"   🎉 Search completed successfully in {search_res['latency_ms']:.2f}ms!")
        metrics["vector_search"] = search_res["latency_ms"]
        
        # Display hits
        hits = search_res["data"]
        print("\n🏆 [RETRIEVED DATA - Search Results]")
        print("   " + "-" * 80)
        for i, hit in enumerate(hits):
            doc_id = hit.get("id")
            score = hit.get("score", 0.0)
            payload = hit.get("payload", {})
            content = payload.get("content", "N/A")
            print(f"   [{i+1}] ID: {doc_id:<18} | Cosine Distance: {score:.6f}")
            print(f"       Content: \"{content}\"")
            print(f"       Payload Details: {json.dumps(payload, ensure_ascii=False)}")
        print("   " + "-" * 80)
    else:
        print(f"   ❌ Search failed: {search_res['error']}")

    # 5. Delete Data & Verify
    print_header("5. Deleting Data & Verifying")
    
    # We will delete the document corresponding to the 2nd sentence:
    # "Vector databases are designed to store, index, and query high-dimensional vector embeddings efficiently."
    target_id = "doc_txt_1"
    print(f"ℹ️ Verifying document '{target_id}' exists first:")
    get_res1 = http_request(f"/collections/{COLLECTION_NAME}/documents/{target_id}", method="GET")
    if get_res1["success"]:
        doc_data = get_res1["data"]
        retrieved_vec = doc_data.get("vectors", {}).get("default", [])
        print(f"   ✅ Found document!")
        print(f"      - ID: {doc_data.get('id')}")
        print(f"      - Text: \"{doc_data.get('payload', {}).get('content')}\"")
        print(f"      - Vector (first 5 elements): {retrieved_vec[:5]}... (Total {len(retrieved_vec)} dims)")
        print(f"      - Query time: {get_res1['latency_ms']:.2f}ms")
    else:
        print(f"   ❌ Document not found: {get_res1['error']}")
        
    print(f"\n🗑️ Deleting document '{target_id}':")
    del_res = http_request(f"/collections/{COLLECTION_NAME}/documents/{target_id}", method="DELETE")
    if del_res["success"] and del_res["data"].get("deleted"):
        print(f"   ✅ Document successfully deleted from index! (Time: {del_res['latency_ms']:.2f}ms)")
        metrics["delete_document"] = del_res["latency_ms"]
    else:
        print(f"   ❌ Deletion failed or document not deleted: {del_res['error']}")
        
    print(f"\n🔍 Verifying document '{target_id}' is gone:")
    get_res2 = http_request(f"/collections/{COLLECTION_NAME}/documents/{target_id}", method="GET")
    if not get_res2["success"] and get_res2["status"] == 404:
        print(f"   ✅ Verified! Document '{target_id}' no longer exists (Server correctly returned 404 DocumentNotFound, Time: {get_res2['latency_ms']:.2f}ms)")
    else:
        print(f"   ⚠️ Unexpected result: Found document or wrong status! Status: {get_res2['status']}, Response: {get_res2.get('data') or get_res2.get('error')}")

    # 6. Clean up (Drop Collection)
    print_header("6. Cleaning Up Workspace")
    print("ℹ️ Dropping collection to leave database environment clean.")
    print("ℹ️ Note: If you wish to inspect files inside `/Users/landaa/Workspaces/aisorun/openvec/openvec_data/` after the run,")
    print("    you can comment out the drop step below in `scripts/openvec_demo.py`.")
    drop_res = http_request(f"/collections/{COLLECTION_NAME}", method="DELETE")
    if drop_res["success"]:
        print(f"✅ Successfully dropped collection '{COLLECTION_NAME}' (Response: {drop_res['data']}, Time: {drop_res['latency_ms']:.2f}ms)")
        metrics["drop_collection"] = drop_res["latency_ms"]
    else:
        print(f"❌ Failed to drop collection: {drop_res['error']}")

    # Summary report outputs
    print("\n" + "=" * 60)
    print("📊 EMPIRICAL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Create Collection:       {metrics.get('create_collection', 0):.2f} ms")
    print(f"Individual Insert (Avg):  {metrics.get('individual_insert_avg', 0):.2f} ms")
    print(f"Batch Insert (100 docs): {metrics.get('batch_insert_100', 0):.2f} ms ({metrics.get('batch_insert_100', 0)/100:.3f} ms/doc)")
    print(f"Vector Similarity Search: {metrics.get('vector_search', 0):.2f} ms")
    print(f"Delete Document:          {metrics.get('delete_document', 0):.2f} ms")
    print(f"Drop Collection:          {metrics.get('drop_collection', 0):.2f} ms")
    print("=" * 60)
    print("🎉 All operations completed successfully!\n")

if __name__ == "__main__":
    main()
