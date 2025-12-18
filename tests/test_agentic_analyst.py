
import os
import pytest
import json
import asyncio
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from stampli.biocontext_stampli import (
    GlobalStore, QuantEngine, QualEngine, QuantTask, QualTask, FilterSchema, agentic_stream_generator, AgentPlan
)

# Mock Data
MOCK_CSV_DATA = """Review_ID,Rating,Year_Month,Reviewer_Location,Review_Text,Branch
1,5,2019-4,Australia,Great park!,Disneyland_HongKong
2,3,2019-5,Australia,Lines too long.,Disneyland_California
3,1,2019-12,UK,Too cold.,Disneyland_Paris
"""

@pytest.fixture
def mock_store(tmp_path):
    # reset singleton
    GlobalStore._instance = None
    
    # write mock csv
    p = tmp_path / "reviews.csv"
    p.write_text(MOCK_CSV_DATA, encoding="utf-8")
    
    with patch("stampli.biocontext_stampli.CSV_PATH", str(p)):
         with patch("stampli.biocontext_stampli.OpenAIEmbeddings") as MockEmbed:
             # Mock Embeddings behavior
             instance = MockEmbed.return_value
             # Make async methods return awaitables (MagicMock is overkill, just use AsyncMock)
             instance.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
             instance.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.1, 0.2, 0.3]])
             
             yield GlobalStore.get_instance()

def test_quant_engine_logic(mock_store):
    # 1. Initialize
    mock_store.initialize()
    
    # 2. Test Filters and Metrics (Quant)
    qe = QuantEngine()
    
    # Task: Avg Rating for Australia
    task = QuantTask(
        filters=FilterSchema(reviewer_country="Australia"),
        metrics=["review_count", "avg_rating"]
    )
    
    results = qe.execute([task])
    res = results["task_0"]
    
    assert res["review_count"] == 2 # ID 1 and 2
    assert res["avg_rating"] == 4.0 # (5+3)/2

def test_quant_engine_date_logic(mock_store):
    mock_store.initialize()
    qe = QuantEngine()
    
    # Task: Count in Spring (Month 3,4,5) -> ID 1 (Apr), ID 2 (May)
    task = QuantTask(
        filters=FilterSchema(season="spring"),
        metrics=["review_count"]
    )
    results = qe.execute([task])
    assert results["task_0"]["review_count"] == 2

    # Task: Count in Dec -> ID 3
    task_w = QuantTask(
        filters=FilterSchema(month=12),
        metrics=["review_count"]
    )
    results_w = qe.execute([task_w])
    assert results_w["task_0"]["review_count"] == 1

def test_qual_engine(mock_store):
    async def _run_test():
        mock_store.initialize()
        # Mock Embeddings are setup in fixture
        
        qe = QualEngine()
        task = QualTask(
            filters=FilterSchema(),
            semantic_intent="lines"
        )
        
        # We mocked docs [0.1, ...], [0.9, ...], [0.1, ...]
        # Query is [0.1, ...]
        # Dot prods: 
        # 1: ~1.0 (High Sim) -> ID 1
        # 2: ~Mid
        # 3: ~1.0 -> ID 3
        
        res = await qe.execute([task])
        snippets = res["snippets"]
        
        # Since we strictly take top 3, we should get all 3 in mock
        assert len(snippets) == 3
        # Check structure
        assert "text" in snippets[0]
        assert "rating" in snippets[0]

    asyncio.run(_run_test())

def test_end_to_end_mocked():
    """Test full generator with MOCKED LLM (no API key needed)."""
    async def _run_test():
        # Mock Global Store
        with patch("stampli.biocontext_stampli.GlobalStore") as MockStoreCls:
            store_inst = MagicMock()
            store_inst.df = pd.DataFrame({
                 "Rating": [5], "season": ["spring"], "month": [4], 
                 "Branch": ["DLP"], "Reviewer_Location": ["UK"],
                 "Review_Text": ["Good"], "sentiment_score": [1.0]
            })
            MockStoreCls.get_instance.return_value = store_inst
            
            # Mock LLM
            with patch("stampli.biocontext_stampli.ChatOpenAI") as MockChat:
                llm_inst = MockChat.return_value
                
                # 1. Router Response (Mock ainvoke)
                # It returns an object that can be parsed as AgentPlan
                mock_plan = AgentPlan(
                    quant_tasks=[QuantTask(filters=FilterSchema(), metrics=["review_count"])],
                    qual_tasks=[]
                )
                # ainvoke returns a dict-like or message
                llm_inst.ainvoke = AsyncMock(return_value={"quant_tasks": [{"filters": {}, "metrics": ["review_count"]}], "qual_tasks": []})
                
                # 2. Synthesis Response (Mock astream)
                async def async_gen(*args, **kwargs):
                    yield MagicMock(content="Final Answer")
                llm_inst.astream = async_gen
                
                # Mock Env for API Key check
                with patch.dict(os.environ, {"OPENAI_API_KEY": "fake-key"}):
                    # Run Generator
                    events = []
                    async for event in agentic_stream_generator([{"role": "user", "content": "Limit test"}]):
                        events.append(event)
                    
                    # Verify Flow
                    # 1. Router Log
                    assert any("Orchestrator" in e for e in events)
                    # 2. Quant Log
                    assert any("Quant Engine" in e for e in events)
                    # 3. Synth Delta
                    assert any("Final Answer" in e for e in events)

    asyncio.run(_run_test())

if __name__ == "__main__":
    # If run directly as script
    pass
