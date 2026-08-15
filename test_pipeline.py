import asyncio
import os
from src.scrapers.web_scraper import WebScout
from src.database.vector_db import VectorLibrarian

async def main():
    # 1. Initialize our components
    scout = WebScout()
    librarian = VectorLibrarian() # This uses the 'all-MiniLM-L6-v2' model from your PFE [cite: 246]
    
    print("--- Starting Market Intelligence Pipeline ---")
    
    # 2. Scrape (The Scout)
    target_url = "https://openai.com/pricing"
    markdown = await scout.scrape_to_markdown(target_url)
    
    if markdown:
        # 3. Process & Store (The Librarian)
        # We split the markdown into chunks for better RAG performance [cite: 246]
        chunks = librarian.prepare_data(markdown)
        librarian.build_index(chunks)
        
        # 4. Save to the /data folder we created earlier
        librarian.save_local(folder_path='data/faiss_index')
        
        print("\n✅ Success! Data is scraped, vectorized, and stored in /data/faiss_index")
    else:
        print("\n❌ Pipeline failed at the scraping stage.")

if __name__ == "__main__":
    asyncio.run(main())