"""
Quick end-to-end test for RAG Engine Phase 2 (Ingestion Pipelines).

Tests:
  1. Structured ingestion — retail_data.xlsx → DuckDB
  2. Unstructured ingestion — chainlit.md → ChromaDB
  3. Query routing after both are loaded
"""

import asyncio
import logging
import os
import sys
import shutil

# Add parent directory to path since script was moved to helper_scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# Ensure .env is loaded
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from llm_factory import setup_global_llm
from rag_engine import RAGEngine

TEST_THREAD_ID = "_test_phase2"


async def test_structured_ingestion(engine: RAGEngine):
    """Test: Excel file → DuckDB tables."""
    print("\n" + "=" * 60)
    print("TEST 1: Structured Ingestion (retail_data.xlsx → DuckDB)")
    print("=" * 60)

    xlsx_path = "./retail_data.xlsx"
    if not os.path.exists(xlsx_path):
        print("❌ SKIP: retail_data.xlsx not found")
        return False

    result = await engine.ingest_file(xlsx_path, "retail_data.xlsx")
    print(result)

    # Verify DuckDB tables exist
    if engine.sql_tables:
        print(f"\n✅ DuckDB tables created: {engine.sql_tables}")
        # Query each table to verify data via SQLAlchemy
        from sqlalchemy import text
        with engine.sql_engine.connect() as conn:
            for table in engine.sql_tables:
                count = conn.execute(
                    text(f'SELECT COUNT(*) FROM "{table}"')
                ).fetchone()[0]
                print(f"   → {table}: {count} rows")
        return True
    else:
        print("❌ No DuckDB tables created!")
        return False


async def test_unstructured_ingestion(engine: RAGEngine):
    """Test: Markdown file → ChromaDB vectors."""
    print("\n" + "=" * 60)
    print("TEST 2: Unstructured Ingestion (chainlit.md → ChromaDB)")
    print("=" * 60)

    md_path = "./chainlit.md"
    if not os.path.exists(md_path):
        print("❌ SKIP: chainlit.md not found")
        return False

    result = await engine.ingest_file(md_path, "chainlit.md")
    print(result)

    # Verify ChromaDB has vectors
    count = engine.chroma_collection.count()
    if count > 0:
        print(f"\n✅ ChromaDB has {count} vectors")
        return True
    else:
        print("❌ ChromaDB is empty!")
        return False


async def test_query_routing(engine: RAGEngine):
    """Test: Query routing after both pipelines are loaded."""
    print("\n" + "=" * 60)
    print("TEST 3: Query Routing")
    print("=" * 60)

    if not engine.has_data():
        print("❌ SKIP: No data loaded")
        return False

    # Test structured query (should route to SQL)
    print("\n--- Structured Query (should use DuckDB) ---")
    q1 = "How many rows are in the data?"
    print(f"Q: {q1}")
    a1 = await engine.query(q1)
    print(f"A: {a1}")

    # Test unstructured query (should use vector search)
    print("\n--- Unstructured Query (should use ChromaDB) ---")
    q2 = "What is Chainlit?"
    print(f"Q: {q2}")
    a2 = await engine.query(q2)
    print(f"A: {a2}")

    return True


async def test_persistence():
    """Test: Close engine → reopen → verify data persists."""
    print("\n" + "=" * 60)
    print("TEST 4: Persistence (close → reopen → verify)")
    print("=" * 60)

    # Attempt to restore from storage (no LLM argument needed)
    engine2 = RAGEngine.load_from_storage(TEST_THREAD_ID)
    if engine2 is None:
        print("❌ load_from_storage returned None")
        return False

    print(f"✅ Engine restored!")
    print(f"   → ChromaDB vectors: {engine2.chroma_collection.count()}")
    print(f"   → DuckDB tables: {engine2.sql_tables}")
    print(f"   → has_data: {engine2.has_data()}")
    print(f"   → Files summary:\n{engine2.get_loaded_files_summary()}")

    # Quick query to verify it works
    q = "How many rows are in the data?"
    print(f"\n   → Test query: '{q}'")
    a = await engine2.query(q)
    print(f"   → Answer: {a}")

    engine2.close()
    return True


async def main():
    print("🧪 RAG Engine Phase 2 — End-to-End Test")
    print("=" * 60)

    # Setup the generic global LLM using factory
    setup_global_llm()

    # Clean up previous test data
    test_storage = f"./storage/rag/{TEST_THREAD_ID}"
    if os.path.exists(test_storage):
        shutil.rmtree(test_storage)
        print(f"🧹 Cleaned previous test storage: {test_storage}")

    # Create engine (does not need LLM argument anymore)
    engine = RAGEngine(thread_id=TEST_THREAD_ID)

    results = {}

    # Test 1: Structured
    results["structured"] = await test_structured_ingestion(engine)

    # Test 2: Unstructured
    results["unstructured"] = await test_unstructured_ingestion(engine)

    # Test 3: Query Routing
    results["query_routing"] = await test_query_routing(engine)

    # Close engine to test persistence
    engine.close()

    # Test 4: Persistence
    results["persistence"] = await test_persistence()

    # Summary
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {test_name}")

    all_pass = all(results.values())
    print(f"\n{'🎉 ALL TESTS PASSED!' if all_pass else '⚠️ SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

