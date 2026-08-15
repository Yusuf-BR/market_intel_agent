import textwrap
from crewai import Agent
from crewai_tools import FileWriterTool
from src.utils.tools import search_market_database, competitor_intelligence_tool
from src.scrapers.web_scraper import Evidence
from typing import List
from datetime import datetime

# -------------------------------
# HELPER: Convert Evidence list → Markdown
# -------------------------------
def evidence_to_markdown(evidences: List[Evidence]) -> str:
    md = f"# Market Intelligence Report\nGenerated: {datetime.utcnow().isoformat()} UTC\n\n"
    for e in evidences:
        md += f"## Source ID: {e.source_id}\n"
        md += f"- Content Type: {getattr(e, 'content_type', 'webpage')}\n"
        md += f"- URL: {getattr(e, 'url', 'N/A')}\n"
        md += f"- Crawl Depth: {getattr(e, 'crawl_depth', 'N/A')}\n"
        md += f"- Retrieved At: {getattr(e, 'retrieved_at', 'N/A')}\n"
        md += f"- Evidence Type: {e.evidence_type}\n\n"
        md += f"{e.excerpt}\n\n---\n\n"
    return md

# -------------------------------
# AGENTS
# -------------------------------
class MarketAgents:

    def research_analyst(self, llm, url: str = "{url}", keywords: List[str] = None):
        """
        Deep Scout agent: retrieves structured Evidence objects.
        'url' and 'keywords' are dynamically interpolated from Task context if not provided.
        """
        keywords = keywords or ["{keywords}"]
        goal_text = textwrap.dedent(f"""
            Conduct an autonomous deep-dive into {url} to uncover hidden intelligence on {', '.join(keywords)}.
            Gather a list of Evidence objects for all relevant technical specs, pricing tables,
            sustainability reports, and other hard data buried 2-3 clicks deep.
        """)

        return Agent(
            role='Strategic Intelligence Explorer',
            goal=goal_text,
            backstory=textwrap.dedent("""
                You are an elite corporate operative. You use the 'Deep Scout' capability 
                to navigate links and bypass bot detection. Ignore marketing fluff; 
                focus only on verified data. Every claim you capture must be represented 
                as a fully structured Evidence object including source URL, crawl depth, 
                and retrieval timestamp.
            """),
            tools=[competitor_intelligence_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=True,
            max_iter=10
        )

    def fact_checker(self, llm, url: str = "{url}", keywords: List[str] = None):
        """
        Verification agent: cross-references Evidence objects with the local Intelligence Vault.
        Uses dynamic interpolation from Task context.
        """
        keywords = keywords or ["{keywords}"]
        goal_text = textwrap.dedent(f"""
            Audit the list of Evidence objects retrieved for {url} by cross-referencing with our Intelligence Vault.
            Flag any outdated, inconsistent, or unverified data points for {', '.join(keywords)}.
        """)

        return Agent(
            role='Lead Verification Auditor',
            goal=goal_text,
            backstory=textwrap.dedent("""
                You are a cynical auditor. Only trust structured Evidence
                that can be traced back to its source. Use 'search_market_database' 
                to verify every data point. Identify hallucinations, outdated numbers,
                missing metrics, and statistical anomalies. Focus on 2025-2026 reports if available.
            """),
            tools=[search_market_database],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=True,
            max_iter=5
        )

    def reporter(self, llm, report_name: str = "market_verdict.md"):
        """
        Synthesis agent: converts verified Evidence objects into CEO-ready markdown reports.
        Interpolates Evidence dynamically.
        """
        file_tool = FileWriterTool()

        def report_writer(evidences: List[Evidence]):
            markdown_text = evidence_to_markdown(evidences)
            # Add optional summary metrics
            esg_metrics = [e for e in evidences if getattr(e, 'metric', None) and getattr(e, 'value', None)]
            if esg_metrics:
                markdown_text += "\n# Summary Metrics\n\n"
                for m in esg_metrics:
                    markdown_text += f"- {m.metric} ({getattr(m, 'year', 'N/A')}): {m.value} [{m.evidence_type}]\n"
            file_tool.write_text(file_name=report_name, content=markdown_text)
            return report_name

        return Agent(
            role='Executive Strategic Consultant',
            goal=textwrap.dedent("""
                Synthesize the list of verified Evidence objects into a '2026 Market Verdict'.
                Ensure every claim in the report is sourced (include URL, crawl depth, timestamp)
                and referenced. Save the report as a markdown file suitable for executives.
            """),
            backstory=textwrap.dedent("""
                You are the final gatekeeper of Intelligence Vault output. Only include
                verified Evidence in your report. Prioritize the 'Value-to-Impact' ratio.
                Do not include unsourced or inferred claims. Each Evidence object should appear with
                its excerpt and full source metadata in the report. Use 'report_writer' to produce markdown.
            """),
            tools=[file_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            memory=True,
            max_iter=3
        )
