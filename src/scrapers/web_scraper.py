import asyncio
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Literal
from urllib.parse import urlparse
from io import BytesIO

import numpy as np
import faiss
import requests
import pdfplumber
from sentence_transformers import SentenceTransformer

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter

# ===============================
# DATA STRUCTURES
# ===============================

@dataclass
class Evidence:
    evidence_id: str
    source_id: str
    excerpt: str
    metric: Optional[str] = None
    value: Optional[str] = None
    year: Optional[str] = None
    evidence_type: Literal["explicit", "inferred"] = "explicit"

# ===============================
# PDF HANDLER
# ===============================

class PDFHandler:
    @staticmethod
    def extract_text(pdf_url: str) -> str:
        try:
            resp = requests.get(pdf_url, timeout=30)
            resp.raise_for_status()
            pdf_bytes = BytesIO(resp.content)

            text = []
            with pdfplumber.open(pdf_bytes) as pdf:
                for page in pdf.pages:
                    text.append(page.extract_text() or "")

            return "\n".join(text).strip()
        except Exception as e:
            print(f"❌ PDF extraction failed: {pdf_url} → {e}")
            return ""

# ===============================
# EVIDENCE EXTRACTION
# ===============================

class EvidenceExtractor:
    @staticmethod
    def extract(text: str) -> List[Evidence]:
        evidences = []
        seen_excerpts = set()
        for block in re.split(r"\n{2,}", text):
            block = re.sub(r"!\[.*?\]\(.*?\)", "", block)  # remove images
            block = re.sub(r"\[.*?\]\(.*?\)", "", block)    # remove links
            block = block.strip()
            if not block or block in seen_excerpts:
                continue
            seen_excerpts.add(block)

            year_match = re.search(r"(20\d{2})", block)
            evidences.append(
                Evidence(
                    evidence_id=str(uuid.uuid4()),
                    source_id="",
                    excerpt=block,
                    year=year_match.group(1) if year_match else None,
                )
            )
        return evidences

# ===============================
# MARKET INDEXER
# ===============================

class MarketIndexer:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)

        self.index_path = os.path.join(index_dir, "index.faiss")
        self.map_path = os.path.join(index_dir, "evidence_map.npy")

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.evidence_list: List[Evidence] = []

        # Load FAISS index if exists
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

        # Safe load of evidence list
        if os.path.exists(self.map_path):
            try:
                data = np.load(self.map_path, allow_pickle=True)
                if data.shape == ():  # 0-d array
                    self.evidence_list = []
                else:
                    self.evidence_list = list(data)
            except Exception:
                self.evidence_list = []

    def add_evidence(self, evidences: List[Evidence]):
        if not evidences:
            return

        # Deduplicate new evidence
        new_excerpts = {e.excerpt for e in evidences}
        self.evidence_list = [e for e in self.evidence_list if e.excerpt not in new_excerpts]
        self.evidence_list.extend(evidences)

        texts = [e.excerpt for e in self.evidence_list]
        vectors = self.model.encode(texts).astype("float32")

        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        else:
            self.index.reset()

        self.index.add(vectors)

        faiss.write_index(self.index, self.index_path)
        np.save(self.map_path, np.array(self.evidence_list, dtype=object))

    def query(self, query: str, k=5) -> List[Evidence]:
        if self.index is None or not self.evidence_list:
            return []

        qv = self.model.encode([query]).astype("float32")
        _, idxs = self.index.search(qv, k)
        return [self.evidence_list[i] for i in idxs[0] if i != -1 and i < len(self.evidence_list)]

# ===============================
# DEEP WEB SCOUT
# ===============================

class DeepWebScout:
    def __init__(self, indexer: MarketIndexer):
        self.indexer = indexer

    async def spider_site(self, base_url: str, keywords: List[str]) -> List[Evidence]:
        evidences: List[Evidence] = []

        if base_url.lower().endswith(".pdf"):
            text = PDFHandler.extract_text(base_url)
            evidences = EvidenceExtractor.extract(text)
        else:
            parsed = urlparse(base_url)
            domain = parsed.netloc.replace("www.", "")

            filters = FilterChain([
                DomainFilter([domain]),
                URLPatternFilter([f"*{k}*" for k in keywords])
            ])

            strategy = BFSDeepCrawlStrategy(
                max_depth=2,
                max_pages=20,
                filter_chain=filters
            )

            run_cfg = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                deep_crawl_strategy=strategy
            )

            async with AsyncWebCrawler(
                config=BrowserConfig(headless=True)
            ) as crawler:

                results = await crawler.arun(base_url, run_cfg)

                # Ensure we iterate over list, not async iterator
                for r in results:
                    if r.success and r.markdown:
                        evidences.extend(EvidenceExtractor.extract(r.markdown))

        for e in evidences:
            e.source_id = base_url

        self.indexer.add_evidence(evidences)
        return self.indexer.query(" ".join(keywords))
