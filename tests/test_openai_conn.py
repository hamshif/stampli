
import os
import asyncio
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.embeddings import OpenAIEmbeddings

import pytest

@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_openai_connection():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in environment.")

    print(f"Key found: {api_key[:5]}...")

    # 1. Test LLM
    print("\nTesting LLM (gpt-4o)...")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key) # Use mini for cheaper test
    res = await llm.ainvoke("Hello, this is a connectivity test.")
    assert res.content, "LLM returned empty response"
    print(f"LLM Response: {res.content}")

    # 2. Test Embeddings
    print("\nTesting Embeddings (text-embedding-3-small)...")
    emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vec = await emb.aembed_query("Test string")
    assert len(vec) > 0, "Embeddings returned empty vector"
    print(f"Embeddings Success. Vector len: {len(vec)}")

if __name__ == "__main__":
    asyncio.run(test_openai_connection())
