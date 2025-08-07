from langchain_community.tools.tavily_search import TavilySearchResults
import config

class WebSearch:
    def __init__(self):
        self.search_tool = TavilySearchResults(k=config.WEB_SEARCH_RESULTS)
    
    def search(self, query: str) -> str:
        """Perform web search and return results"""
        try:
            results = self.search_tool.run(query)
            return str(results)
        except Exception as e:
            return f"Web search error: {str(e)}"