
import os
import json
import logging
import asyncio
import pandas as pd
import numpy as np
from typing import List, Dict, AsyncGenerator, Optional
from pathlib import Path
from stampli.paths import get_enriched_path, CURRENT_VERSION
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stampli_chat")

# Load API Key
if not os.environ.get("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY not found in environment. Please export it.")

# --- Global Store (In-Memory Data) ---
# --- Global Store (In-Memory Data) ---
class GlobalStore:
    _instance = None
    df: pd.DataFrame = None
    embedding_model = None
    valid_branches: List[str] = []
    active_path: Path = None
    active_version: str = CURRENT_VERSION
    enrichment_coverage: float = 0.0

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self):
        if self.df is not None:
            return
        
        # Load the ENRICHED data
        self.active_path = get_enriched_path()
        if not self.active_path.exists():
            logger.error(f"Enriched data not found at {self.active_path}")
            return

        logger.info(f"Loading {self.active_path} (Version: {self.active_version})...")
        self.df = pd.read_parquet(self.active_path)
        
        # Guardrail: Capture actual available branches
        if "Branch" in self.df.columns:
            self.valid_branches = sorted(self.df["Branch"].dropna().unique().tolist())
            logger.info(f"Valid Branches in Data: {self.valid_branches}")
            
        # Initialize Embeddings
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
                logger.info("Embedding Model Initialized.")
            else:
                logger.warning("OPENAI_API_KEY missing. Embeddings disabled.")
        except Exception as e:
            logger.error(f"Failed to init embeddings: {e}")
        
        # Calculate Coverage
        if 'sentiment_score' in self.df.columns:
            enriched_count = self.df['sentiment_score'].notna().sum()
            self.enrichment_coverage = (enriched_count / len(self.df)) * 100
            
        logger.info(f"Loaded {len(self.df)} rows. Enrichment Coverage: {self.enrichment_coverage:.1f}%")

    def normalize_branch(self, user_input: str) -> Optional[str]:
        """
        Maps user input (e.g. 'Paris', 'California') to canonical Branch names 
        (e.g. 'Disneyland_Paris', 'Disneyland_California').
        Returns None if no obvious map, OR if the mapped branch isn't in valid_branches? 
        No, normalization should just map to canonical. Validation happens later.
        """
        if not user_input: return None
        
        input_lower = user_input.lower().strip()
        
        # 1. Direct Match (Case-insensitive)
        for b in self.valid_branches:
            if b.lower() == input_lower:
                return b
                
        # 2. Known Aliases (Static Map) - extend as needed
        ALIAS_MAP = {
            "paris": "Disneyland_Paris",
            "eurodisney": "Disneyland_Paris",
            "hong kong": "Disneyland_HongKong",
            "hk": "Disneyland_HongKong",
            "california": "Disneyland_California",
            "anaheim": "Disneyland_California",
            "florida": "Disney_World_Florida",
            "orlando": "Disney_World_Florida",
            "disneyland": None # Ambiguous/Generic -> Return None so validator catches it
        }
        
        # Check explicit aliases
        if input_lower in ALIAS_MAP:
            return ALIAS_MAP[input_lower]
            
        # 3. Substring Heuristic (Risky but helpful for "Disneyland Paris")
        # If input contains "Paris", map to Disneyland_Paris
        for alias, target in ALIAS_MAP.items():
            if alias and alias in input_lower and target:
                 return target
                 
        return user_input # Return as-is if no map found (Validator will likely reject it)

# Initialize strictly on startup for speed
GlobalStore.get_instance().initialize()


# --- Models ---
# Season Mapping (Venue -> Season -> Months)
SEASON_MAP = {
  "Disneyland_HongKong": {
    "Spring": [3, 4, 5], 
    "Summer": [6, 7, 8], 
    "Autumn": [9, 10, 11], 
    "Winter": [12, 1, 2]
  },
  "Disneyland_California": {
    "Spring": [3, 4, 5], 
    "Summer": [6, 7, 8], 
    "Autumn": [9, 10, 11], 
    "Winter": [12, 1, 2]
  },
  "Disneyland_Paris": {
    "Spring": [3, 4, 5], 
    "Summer": [6, 7, 8], 
    "Autumn": [9, 10, 11], 
    "Winter": [12, 1, 2]
  },
  "Disney_World_Florida": { 
    "Spring": [3, 4, 5], 
    "Summer": [6, 7, 8], 
    "Autumn": [9, 10, 11], 
    "Winter": [12, 1, 2]
  }
}

class FilterSchema(BaseModel):
    topics: Optional[List[str]] = Field(None, description="List of topics to filter by (e.g. ['Food', 'Queue', 'Price'])")
    sentiment_label: Optional[str] = Field(None, description="Sentiment label (Positive, Negative, Neutral)")
    is_complaint: Optional[bool] = Field(None, description="True if looking for complaints/problems")
    
    # New High-Signal Filters
    crowd_level: Optional[str] = Field(None, description="Filter: Empty, Moderate, Packed")
    staff_sentiment: Optional[str] = Field(None, description="Filter: Friendly, Rude, Neutral")
    price_sensitivity: Optional[str] = Field(None, description="Filter: Worth it, Expensive, Rip-off")
    family_sentiment: Optional[str] = Field(None, description="Filter: Harmonious, Joyful, Neutral, Tired, Stressed, Conflict")
    
    # Metadata Filters
    reviewer_location: Optional[str] = Field(None, description="Country e.g. 'Australia'")
    branch: Optional[str] = Field(None, description="Park Branch e.g. 'Disneyland_HongKong'")
    year_month: Optional[str] = Field(None, description="Specific Date e.g. '2019-4'")
    month: Optional[int] = Field(None, description="Month of the year (1-12) if mentioned without a year (e.g. 'in June' -> 6)")
    season: Optional[str] = Field(None, description="Season e.g. 'Spring', 'Summer', 'Autumn', 'Winter'")

    unsupported_entity: Optional[str] = Field(None, description="If the user asks about a location NOT in [HongKong, California, Paris, Florida], extract it here (e.g. 'Tel-Aviv', 'Tokyo').")

    reasoning: str = Field(..., description="Brief explanation of why these filters were chosen")

def get_months_for_season(branch: str, season: str) -> Optional[List[int]]:
    if not branch or not season: 
        return None
    # normalize
    branch_key = None
    for k in SEASON_MAP.keys():
        if branch.lower() in k.lower():
            branch_key = k
            break
    
    if not branch_key:
        return None
        
    # normalize season
    season_val = season.capitalize()
    return SEASON_MAP[branch_key].get(season_val)

class StampliRequest(BaseModel):
    messages: List[Dict[str, str]]


# --- App Setup ---
app = FastAPI(title="Stampli Filter Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logic: Context Management
MAX_CONTEXT_MSG = 10
TOP_K_REVIEWS = 5

def get_chat_history(messages: List[Dict[str, str]]) -> List:
    lc_messages = []
    recent = messages[-MAX_CONTEXT_MSG:]
    for m in recent:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))
    return lc_messages


async def chat_stream_generator(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    store = GlobalStore.get_instance()
    user_query = messages[-1]["content"]
    
    store = GlobalStore.get_instance()
    user_query = messages[-1]["content"]
    
    # 1. ROUTER: Decide Filters (gpt-4o-mini)
    router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=FilterSchema)
    
    # Get History (excluding current message)
    history_msgs = get_chat_history(messages[:-1])
    history_str = "\n".join([f"{type(m).__name__}: {m.content}" for m in history_msgs])
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query router. Convert the user question into database filters.
        
        CONTEXT RULES:
        1. Use Chat History ONLY if the user is asking a follow-up question (e.g. "Why?", "What about Paris?").
        2. IF the user asks a completely new question (e.g. changing from Hong Kong to California), IGNORE previous filters.
        3. Do NOT merge filters from unrelated turns.
        
        EXTRACTION RULES:
        - `month`: If the user mentions a month (e.g., "June", "November") without a specific year, set `month` to the integer (1-12).
        - `crowd_level`: ONLY set this if the user specifically asks for a level (e.g., "when is it empty?"). If they ask "is it crowded?", keep it null and add 'Queues' to `topics` instead. This allows the semantic search to find both Moderate and Packed reviews rather than over-filtering.
        - `branch`: Always map to canonical names (Disneyland_California, Disneyland_Paris, Disneyland_HongKong, Disney_World_Florida).
        
        {format_instructions}"""),
        ("user", "Chat History:\n{history}\n\nCurrent Question: {query}")
    ])
    
    # Yield log
    yield f"data: {json.dumps({'type': 'log', 'data': {'text': 'Analyzing intent...', 'status': 'thinking'}})}\n\n"
    
    # Execute Router
    try:
        filters: FilterSchema = await (router_prompt | router_llm | parser).ainvoke({
            "history": history_str,
            "query": user_query,
            "format_instructions": parser.get_format_instructions()
        })
        
        # --- GUARDRAIL & NORMALIZATION ---
        
        # 1. Recovery: Check if 'unsupported_entity' is actually a known branch
        if filters.unsupported_entity:
            recovered = store.normalize_branch(filters.unsupported_entity)
            if recovered:
                logger.info(f"Recovered branch from unsupported: {filters.unsupported_entity} -> {recovered}")
                filters.branch = recovered
                filters.unsupported_entity = None

        # 2. Normalize Branch (Handle 'Paris' -> 'Disneyland_Paris')
        if filters.branch:
            normalized = store.normalize_branch(filters.branch)
            if normalized:
                filters.branch = normalized
        
        # 3. Check Validity
        refusal_reason = None
        
        # Explicit unsupported entity from LLM (e.g. Tel-Aviv)
        if filters.unsupported_entity:
            refusal_reason = filters.unsupported_entity
            
        # Implicit invalid branch (e.g. "California" when only Paris/HK loaded)
        elif filters.branch and filters.branch not in store.valid_branches:
            refusal_reason = filters.branch

        # 3. Short-Circuit if Invalid
        if refusal_reason:
            valid_str = ", ".join(store.valid_branches)
            msg = f"I do not have data about {refusal_reason}. I only have data for: {valid_str}."
            
            yield f"data: {json.dumps({'type': 'log', 'data': {'text': f'Guardrail Hit: {refusal_reason}', 'status': 'error'}})}\n\n"
            
            # Synthesize a polite refusal block
            yield f"data: {json.dumps({'type': 'delta', 'data': {'text': msg}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        filters_json = filters.model_dump_json(exclude={'reasoning'})
        yield f"data: {json.dumps({'type': 'log', 'data': {'text': f'Filters: {filters_json}', 'status': 'done'}})}\n\n"
    except Exception as e:
        logger.error(f"Router failed: {e}")
        filters = FilterSchema(reasoning="Failed to parse") # Fallback to no filters

    # 2. ENGINE: Apply Filters (Pandas)
    relevant_reviews = []
    if store.df is not None and not filters.model_dump(exclude_none=True).keys() <= {'reasoning'}:
        df = store.df.copy()
        
        # Apply filters safely
        logger.info(f"Initial DF Size: {len(df)}")
        
        # --- Seasonal Filtering ---
        if filters.season:
             target_months = get_months_for_season(filters.branch, filters.season)
             logger.info(f"Season Filter: {filters.season} @ {filters.branch} -> Months: {target_months}")
             
             if target_months:
                 # Extract month int from Year_Month "2019-4" -> 4
                 def check_month(ym_str):
                     try:
                         if not ym_str: return False
                         m = int(ym_str.split('-')[1])
                         return m in target_months
                     except:
                         return False
                 
                 df = df[df['Year_Month'].apply(check_month)]
                 logger.info(f"After season ({filters.season}): {len(df)}")
             else:
                 logger.warning("Season filter requested but could not map to months (missing branch?). Skipping.")

        if filters.sentiment_label:
            df = df[df['sentiment_label'] == filters.sentiment_label]
            logger.info(f"After sentiment_label: {len(df)}")
        
        if filters.is_complaint is not None:
            df = df[df['is_complaint'] == filters.is_complaint]
            logger.info(f"After is_complaint: {len(df)}")
            
        if filters.crowd_level:
            df = df[df['crowd_level'] == filters.crowd_level]
            logger.info(f"After crowd_level: {len(df)}")
            
        if filters.staff_sentiment:
            df = df[df['staff_sentiment'] == filters.staff_sentiment]
            logger.info(f"After staff_sentiment: {len(df)}")
            
        if filters.price_sensitivity:
            df = df[df['price_sensitivity'] == filters.price_sensitivity]
            logger.info(f"After price_sensitivity: {len(df)}")
            
        if filters.family_sentiment:
            df = df[df['family_sentiment'] == filters.family_sentiment]
            logger.info(f"After family_sentiment: {len(df)}")

        if filters.reviewer_location:
            logger.info(f"Filtering Location: '{filters.reviewer_location}' vs DB Sample: {df['Reviewer_Location'].unique()[:5]}")
            df = df[df['Reviewer_Location'].str.lower() == filters.reviewer_location.lower()]
            logger.info(f"After reviewer_location: {len(df)}")
            
        if filters.branch:
             logger.info(f"Filtering Branch: '{filters.branch}' vs DB Sample: {df['Branch'].unique()[:5]}")
             df = df[df['Branch'].str.lower() == filters.branch.lower()]
             logger.info(f"After branch: {len(df)}")

        if filters.year_month:
             df = df[df['Year_Month'] == filters.year_month]
             logger.info(f"After year_month: {len(df)}")
            
        if filters.month:
             def check_month_only(ym_str):
                 try:
                     if not ym_str: return False
                     return int(ym_str.split('-')[1]) == filters.month
                 except: return False
             df = df[df['Year_Month'].apply(check_month_only)]
             logger.info(f"After month ({filters.month}): {len(df)}")

        if filters.topics:
            def has_topic(row_topics):
                if not isinstance(row_topics, (list, np.ndarray)): return False
                return any(t.lower() in [x.lower() for x in row_topics] for t in filters.topics)
            
            mask = df['topics'].apply(has_topic)
            df = df[mask]
            logger.info(f"After topics: {len(df)}")
        
        if filters.topics:
            def has_topic(row_topics):
                if not isinstance(row_topics, (list, np.ndarray)): return False
                return any(t.lower() in [x.lower() for x in row_topics] for t in filters.topics)
            
            mask = df['topics'].apply(has_topic)
            df = df[mask]
            logger.info(f"After topics: {len(df)}")
        
        # --- SEMANTIC RETRIEVAL ---
        total_records = len(store.df)
        matches = len(df)
        
        # 1. Candidate Selection (Prioritize Recent if massive match, or just take head)
        # Assuming Data is roughly sorted or we rely on 'Year_Month'
        # Parsing Year_Month (YYYY-M) to sorts might be heavy. 
        # For now, let's take a larger pool (e.g. 150) to re-rank.
        CANDIDATE_POOL_SIZE = 150
        candidates = df.head(CANDIDATE_POOL_SIZE).copy()
        
        relevant_reviews = []
        
        if not candidates.empty:
            texts = candidates["Review_Text"].fillna("").tolist()
            
            try:
                # Lazy Init Model if needed (though GlobalStore should have it)
                if store.embedding_model is None:
                    # Fallback init
                     from langchain_openai import OpenAIEmbeddings
                     api_key = os.environ.get("OPENAI_API_KEY")
                     store.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
                
                # Embed Object
                logger.info(f"Embedding Query: {user_query}")
                q_vec = await store.embedding_model.aembed_query(user_query)
                logger.info(f"Embedding {len(texts)} Documents...")
                doc_vecs = await store.embedding_model.aembed_documents(texts)
                
                # Cosine Similarity
                q_vec = np.array(q_vec)
                doc_vecs = np.array(doc_vecs)
                
                norm_q = np.linalg.norm(q_vec)
                norm_docs = np.linalg.norm(doc_vecs, axis=1)
                
                # Avoid div zero
                if norm_q == 0: norm_q = 1e-9
                norm_docs[norm_docs == 0] = 1e-9
                
                sims = np.dot(doc_vecs, q_vec) / (norm_docs * norm_q)
                
                # Sort (Highest Sim First)
                top_indices = np.argsort(sims)[-TOP_K_REVIEWS:][::-1]
                
                for idx in top_indices:
                    row = candidates.iloc[idx]
                    relevant_reviews.append(row.to_dict())
                    
                logger.info("Semantic Sort Complete.")
                
            except Exception as e:
                logger.error(f"Semantic Retrieval Failed: {e}. Falling back to random head.")
                relevant_reviews = candidates.head(TOP_K_REVIEWS).to_dict('records')

        file_info = f"[{store.active_version}: {store.active_path.name} ({store.enrichment_coverage:.1f}% Enriched)]"
        cardinality_msg = f"{file_info} Found {matches} matches of {total_records} entries. Using top {len(relevant_reviews)}."
        
        # Calculate Distribution
        dist_str = "N/A"
        if not df.empty and 'sentiment_label' in df.columns:
            # Drop likely None/NaN before counting if needed, though value_counts handles it
            counts = df['sentiment_label'].value_counts(normalize=True) * 100
            dist_str = ", ".join([f"{k} ({v:.1f}%)" for k, v in counts.items()])
            
        yield f"data: {json.dumps({'type': 'log', 'data': {'text': cardinality_msg, 'status': 'done'}})}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'log', 'data': {'text': 'No matches found / No filters.', 'status': 'neutral'}})}\n\n"
        cardinality_msg = "No filters applied."
        dist_str = "N/A"

    # 3. SYNTHESIZER: Generate Answer (gpt-4o)
    synth_llm = ChatOpenAI(model="gpt-4o", streaming=True)
    
    # Construct Context
    context_str = ""
    if relevant_reviews:
        context_str = "RELEVANT REVIEWS FOUND:\n"
        for i, r in enumerate(relevant_reviews):
            context_str += f"{i+1}. {r.get('Review_Text', '')} (Topics: {r.get('topics', [])}, Branch: {r.get('Branch', '')})\n"
    
    k = len(relevant_reviews)
    
    system_prompt = f"""### ROLE:
You are a disciplined Disney Analyst. Your output MUST follow the TEMPLATE below exactly. 
    
### CONTEXT METADATA:
- CARDINALITY_MSG: "{cardinality_msg}"
- SENTIMENT_STATS: "{dist_str}"
- REVIEW_COUNT: {k}

### MANDATORY TEMPLATE:
1. First Line: Print the CARDINALITY_MSG exactly.
2. Second Line: Print the stats as: Stats: Report the Sentiment Distribution: "{dist_str}"
3. Section: "Review Highlights"
   - Provide exactly {k} bullet points (one for each review in the CONTEXT below).
   - Each bullet should be a 1-sentence summary of that specific review.
4. Section: "Takeaways"
   - A short paragraph synthesizing trends and final answer.

### CONTEXT:
{{context_str}}

### CHAT HISTORY:
{{history_str}}
"""
    
    # We do NOT pass history as messages list anymore because we embed it in system prompt 
    # to control context window and format more strictly? 
    # Originally: history = get_chat_history(...) + [System] + [User]
    # Let's keep the original structure but ENRICH the system prompt with metadata about the context.
    
    prompt_msgs = [SystemMessage(content=system_prompt)]
    # Add recent conversation turns for flow
    prompt_msgs.extend(get_chat_history(messages[:-1]))
    prompt_msgs.append(HumanMessage(content=user_query))
    
    # Stream Response
    async for chunk in synth_llm.astream(prompt_msgs):
        if chunk.content:
            yield f"data: {json.dumps({'type': 'delta', 'data': {'text': chunk.content}})}\n\n"
            
    yield "data: [DONE]\n\n"


@app.post("/api/stampli")
async def chat_endpoint(payload: StampliRequest):
    return StreamingResponse(
        chat_stream_generator(payload.messages),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
