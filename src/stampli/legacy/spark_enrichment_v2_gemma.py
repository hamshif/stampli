# --- CONFIGURATION ---
import os
import sys

# Get the absolute path of the directory containing this script
import sys
from pathlib import Path

# Ensure we can import biocontext modules
SCRIPT_PATH = Path(__file__).resolve()
SRC_DIR = SCRIPT_PATH.parents[4]
sys.path.append(str(SRC_DIR))

from stampli.paths import REVIEWS_BASE_PARQUET, REVIEWS_ENRICHED_V2_GEMMA

# Get the absolute path of the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_PATH = str(REVIEWS_BASE_PARQUET)
ENRICHED_PATH = str(REVIEWS_ENRICHED_V2_GEMMA)

INGESTION_BATCH = 20  # Small test batch

# LLM Configuration
LLM_PROVIDER = "ollama" # Options: "openai", "ollama"
OLLAMA_MODEL = "gemma2:9b"
OPENAI_MODEL = "gpt-4o-mini"

# %% [markdown]
# Cell

# Check available Ollama models if using Ollama
if LLM_PROVIDER == "ollama":
    print(f"Using Local Ollama with model: {OLLAMA_MODEL}")
    try:
        import subprocess
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print("Available Models:\n", res.stdout)
    except Exception as e:
        print("Could not list ollama models (is ollama installed?):", e)
else:
    print(f"Using OpenAI with model: {OPENAI_MODEL}")

# %% [markdown]
# Cell

import pandas as pd
import os

from stampli.util.display import display_scrollable_dataframe


# %% [markdown]
# Cell

# 1. Setup Spark
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, from_json, count
    from pyspark.sql.types import StructType, StructField, StringType, BooleanType, ArrayType, IntegerType
    from stampli.util.runtime import bootstrap_spark_env
except ImportError:
    print("Please install pyspark: pip install pyspark")
    sys.exit(1)

bootstrap_spark_env()

spark = (SparkSession.builder
    .appName("StampliEnrichmentResumable")
    .config("spark.executor.memory", "16g")
    .config("spark.driver.memory", "16g")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")
print("Spark Session Created (16GB)")

# %% [markdown]
# Cell

# 2. Load Base Data
if not os.path.exists(BASE_PATH):
    print(f"Error: Base path {BASE_PATH} does not exist.")
    print(f"Current Working Base: {os.getcwd()}")
    print("Please ensure you have run the ingestion step.")
    sys.exit(1)
    
sdf_base = spark.read.parquet(BASE_PATH)
print(f"Base Rows: {sdf_base.count()}")

# %% [markdown]
# Cell

# 3. Load Prior Enrichment (Handle First Run)
import os

if os.path.exists(ENRICHED_PATH) and len(os.listdir(ENRICHED_PATH)) > 0:
    try: 
        sdf_done = spark.read.parquet(ENRICHED_PATH)
        done_count = sdf_done.count()
        print(f"Already Enriched: {done_count}")
    except Exception:
        # Possible directory exists but is empty or corrupt
        print("Enriched directory empty or invalid. Starting fresh.")
        sdf_done = None
        done_count = 0
else:
    print("No prior enrichment found. Starting fresh.")
    sdf_done = None
    done_count = 0

# %% [markdown]
# Cell

# 4. Calculate To-Do List (Left Anti Join)
if sdf_done is not None:
    # Exclude rows where review_uid is already in df_done
    sdf_todo = sdf_base.join(sdf_done, "review_uid", "left_anti")
else:
    sdf_todo = sdf_base

todo_count = sdf_todo.count()
print(f"Remaining To Do: {todo_count}")

if todo_count == 0:
    print("All done!")
else:
    # 5. Limit Batch
    df_batch = sdf_todo.limit(INGESTION_BATCH)
    print(f"Processing Batch of: {INGESTION_BATCH}")

# %% [markdown]
# Cell

import json
import os
from typing import Iterator
import pandas as pd

# ----------------------------
# 1. Output Schema (Spark)
# ----------------------------
ENRICHED_SCHEMA = StructType([
    # Core Enriched Fields
    StructField("sentiment_score", IntegerType(), True),
    StructField("sentiment_label", StringType(), True),
    StructField("topics", ArrayType(StringType()), True),
    StructField("is_complaint", BooleanType(), True),
    
    # High-Signal Extras
    StructField("crowd_level", StringType(), True),
    StructField("queue_time_rating", StringType(), True),
    StructField("staff_sentiment", StringType(), True),
    StructField("price_sensitivity", StringType(), True),
    StructField("family_sentiment", StringType(), True),
    StructField("summary", StringType(), True),
    StructField("entities", ArrayType(StringType()), True)
])

# Intermediate Schema (Passthrough + JSON)
BATCH_SCHEMA = StructType([
    StructField("review_uid", StringType(), True),
    StructField("Review_Text", StringType(), True),
    
    # Metadata Passthrough
    StructField("Rating", IntegerType(), True),
    StructField("Reviewer_Location", StringType(), True),
    StructField("Year_Month", StringType(), True),
    StructField("Branch", StringType(), True),
    
    # New Payload
    StructField("enriched_json", StringType(), True)
])

# Pass configuration as default args to avoid serialization issues for simple globals
def enrich_partition(iterator: Iterator[pd.DataFrame], provider=LLM_PROVIDER, ol_model=OLLAMA_MODEL, oa_model=OPENAI_MODEL) -> Iterator[pd.DataFrame]:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    from typing import List, Optional

    # ----------------------------
    # 2. Pydantic Model (LLM)
    # ----------------------------
    class ReviewEnrichment(BaseModel):
        sentiment_score: int = Field(..., description="1-5 score")
        sentiment_label: str = Field(..., description="Positive/Negative/Neutral")
        topics: List[str] = Field(..., description="List of main topics")
        is_complaint: bool = Field(..., description="True if complaint")
        
        # High-Signal Fields
        crowd_level: Optional[str] = Field(None, description="Enum: Empty, Moderate, Packed")
        queue_time_rating: Optional[str] = Field(None, description="Enum: Acceptable, Long, Unbearable")
        staff_sentiment: Optional[str] = Field(None, description="Enum: Friendly, Rude, Neutral")
        price_sensitivity: Optional[str] = Field(None, description="Enum: Worth it, Expensive, Rip-off")
        family_sentiment: Optional[str] = Field(None, description="Enum: Harmonious, Joyful, Neutral, Tired, Stressed, Conflict")
        summary: Optional[str] = Field(None, description="1-sentence dense summary")
        entities: Optional[List[str]] = Field(None, description="Named entities mentioned")

    if provider == "ollama":
        # Use OpenAI client pointing to Ollama endpoint
        llm = ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=ol_model,
            temperature=0
        )
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        llm = ChatOpenAI(model=oa_model, api_key=api_key, temperature=0)

    parser = PydanticOutputParser(pydantic_object=ReviewEnrichment)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data extraction engine. Return VALID JSON ONLY. No markdown, no thoughts.\n{format_instructions}"),
        ("user", "{text}")
    ])
    chain = prompt | llm | parser
    
    for pdf in iterator:
        results = []
        format_instr = parser.get_format_instructions()
        for text in pdf["Review_Text"]:
            try:
                obj = chain.invoke({"text": text, "format_instructions": format_instr})
                results.append(obj.model_dump_json())
            except Exception:
                results.append(json.dumps({}))
        
        pdf["enriched_json"] = results
        # Return ALL columns (Review_Text, review_uid + Metadata + JSON)
        yield pdf


# 6. Run Enrichment (Action)
# 6. Run Enrichment
if todo_count > 0:
    target_cols = ["review_uid", "Review_Text", "Rating", "Reviewer_Location", "Year_Month", "Branch"]
    
    CHUNK_SIZE = 50 
    NUM_PARTITIONS = 100 if LLM_PROVIDER == "openai" else 2
    if LLM_PROVIDER == "ollama":
        CHUNK_SIZE = 10 # Very small for test
    
    print(f"Starting Processing loop for V2 (Gemma). Total To Do: {todo_count}")
    print(f"Partitions: {NUM_PARTITIONS} | Batch Size: {CHUNK_SIZE}")

    all_todo_uids = [row.review_uid for row in sdf_todo.select("review_uid").collect()]
    total_chunks = (len(all_todo_uids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for i in range(total_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, len(all_todo_uids))
        current_chunk_uids = all_todo_uids[start_idx:end_idx]
        
        print(f"\n--- Processing Chunk {i+1}/{total_chunks} ({len(current_chunk_uids)} rows) ---")
        
        df_chunk = sdf_base.filter(col("review_uid").isin(current_chunk_uids))
        
        raw_enriched = df_chunk.select(*target_cols).repartition(NUM_PARTITIONS).mapInPandas(enrich_partition, schema=BATCH_SCHEMA)
        
        final_sdf = raw_enriched \
            .withColumn("parsed", from_json(col("enriched_json"), ENRICHED_SCHEMA)) \
            .select(*target_cols, "enriched_json", "parsed.*") 
        
        final_sdf.write.mode("append").parquet(ENRICHED_PATH)
        print(">> Saved Chunk.")
            
    print("\nAll Batches Complete.")

# %% [markdown]
# Cell

# sanity read enriched into pandas dataframe

pd_sanity = pd.read_parquet(ENRICHED_PATH) 

len(pd_sanity)

# %% [markdown]
# Cell

# show top 10 lines
pd_sanity.head(10)