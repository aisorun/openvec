#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Set search path to pick up the compiled openvec_py.so
sys.path.insert(0, os.path.abspath("scripts"))

try:
    import openvec_py
except ImportError as e:
    print("❌ Failed to import openvec_py directly.")
    print("   Please make sure to run the symlink step first.")
    print(f"   Error details: {e}")
    sys.exit(1)

def test_flow():
    print("🚀 Running PyO3 Binding Validation Tests...")
    
    # 1. Initialize local DB
    db_dir = "scripts/py_test_data"
    if os.path.exists(db_dir):
        import shutil
        shutil.rmtree(db_dir)
        
    db = openvec_py.OpenVecPy(db_dir)
    print("✅ OpenVecPy successfully instantiated locally.")
    
    # 2. Collections management
    coll_name = "py_test_coll"
    if db.collection_exists(coll_name):
        db.drop_collection(coll_name)
        
    coll = db.create_collection(coll_name, 3, "cosine")
    print(f"✅ Collection '{coll_name}' successfully created (Metric: Cosine, Dim: 3).")
    
    assert db.collection_exists(coll_name)
    assert coll_name in db.list_collections()
    
    # 3. Document insertion
    doc1_id = "py_doc_1"
    doc2_id = "py_doc_2"
    doc1_vec = [1.0, 0.0, 0.0]
    doc2_vec = [0.0, 1.0, 0.0]
    doc1_payload = {"name": "Alice", "age": 30, "is_admin": True}
    doc2_payload = {"name": "Bob", "age": 25, "is_admin": False}
    
    coll.insert(doc1_id, doc1_vec, doc1_payload)
    coll.insert(doc2_id, doc2_vec, doc2_payload)
    print("✅ Documents successfully inserted into Collection.")
    
    # Verify count
    assert coll.doc_count() == 2
    assert coll.index_type() == "flat"
    
    # 4. Search
    query = [0.95, 0.05, 0.0]
    results = coll.search(query, limit=1)
    
    print(f"🔎 Search Results: {results}")
    assert len(results) == 1
    hit = results[0]
    assert hit["id"] == doc1_id
    assert hit["payload"]["name"] == "Alice"
    assert hit["payload"]["age"] == 30
    assert hit["payload"]["is_admin"] is True
    print("✅ Search similarity and payload parsing successfully verified.")
    
    # 5. Deletion
    deleted = coll.delete(doc1_id)
    assert deleted
    assert coll.doc_count() == 1
    
    # Verify deleted
    results_after = coll.search(query, limit=1)
    assert results_after[0]["id"] == doc2_id
    print("✅ Document deletion successfully verified.")
    
    # 6. Cleanup
    dropped = db.drop_collection(coll_name)
    assert dropped
    print("🧹 Collection dropped successfully.")
    
    print("🎉 All PyO3 Python binding tests passed successfully!\n")

if __name__ == "__main__":
    test_flow()
