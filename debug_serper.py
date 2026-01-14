import asyncio
import os
from connectors.youtube_analyzer import get_youtube_analyzer

async def debug():
    analyzer = get_youtube_analyzer()
    query = "Tesla stock news" # Assuming this was the query
    print(f"Searching for: {query}")
    
    # We need to manually call the client inside because search_youtube_videos parses it
    # But let's call search_youtube_videos and see what it returns
    results = await analyzer.search_youtube_videos(query, num_results=5)
    
    print(f"\nFound {len(results)} videos:")
    for i, r in enumerate(results):
        print(f"[{i}] Title: {r.title}")
        print(f"    URL: '{r.url}'")
        print(f"    ID calculation would be: {r.url.split('v=')[-1] if 'v=' in r.url else 'UNKNOWN'}")

if __name__ == "__main__":
    asyncio.run(debug())
