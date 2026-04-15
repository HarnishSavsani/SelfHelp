import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from llm_factory import setup_global_llm
from rag_engine import RAGEngine

async def test():
    setup_global_llm()
    engine = RAGEngine.load_from_storage("_test_phase2")
    if engine is None:
        print("No engine loaded. Run test_rag.py first.")
        return
    res = await engine.query("How many rows are in the data?")
    print("Response:", res)

asyncio.run(test())
