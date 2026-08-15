import os
import asyncio
import threading
from typing import List

import nest_asyncio
from crewai.tools import tool

from src.scrapers.web_scraper import Evidence, DeepWebScout, MarketIndexer

INDEX_DIR = os.path.join(os.getcwd(), "data", "market_index")
os.makedirs(INDEX_DIR, exist_ok=True)

indexer = MarketIndexer(index_dir=INDEX_DIR)
db_lock = threading.Lock()

@tool("search_market_database")
def search_market_database(query: str) -> List[Evidence]:
    """
    Search the local vector database for indexed market evidence.
    Returns a list of Evidence objects.
    """
    with db_lock:
        try:
            return indexer.query(query, k=7)
        except Exception as e:
            raise RuntimeError(f"Market DB query failed: {e}")

@tool("competitor_intelligence_tool")
def competitor_intelligence_tool(url: str, keywords: List[str]) -> List[Evidence]:
    """
    Crawl a competitor website or PDF, extract evidence,
    index it, and return matched Evidence objects.
    """
    scout = DeepWebScout(indexer)
    nest_asyncio.apply()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        evidences = loop.run_until_complete(
            scout.spider_site(base_url=url, keywords=keywords)
        )
        return evidences or []
    except Exception as e:
        raise RuntimeError(f"Competitor intelligence crawl failed for {url}: {e}")
