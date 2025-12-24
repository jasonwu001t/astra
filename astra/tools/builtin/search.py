"""Search tool - Native search implementation"""

import os
from typing import Optional, Dict, Any, List

from ..base import Tool, ToolParameter


class SearchTool(Tool):
    """
    Smart hybrid search tool

    Supports multiple search engine backends with intelligent source selection:
    1. Hybrid mode - Intelligently selects TAVILY or SERPAPI
    2. Tavily API - Professional AI search
    3. SerpApi - Traditional Google search
    """

    def __init__(self, backend: str = "hybrid", tavily_key: Optional[str] = None, serpapi_key: Optional[str] = None):
        super().__init__(
            name="search",
            description="A smart web search engine. Supports hybrid search mode, automatically selects best source. Use this tool when you need to answer questions about current events, facts, or information not in your knowledge base."
        )
        self.backend = backend
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")
        self.available_backends = []
        self.tavily_client = None
        self._setup_backends()

    def _setup_backends(self):
        """Setup search backends"""
        # Check Tavily availability
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
                print("✅ Tavily search engine initialized")
            except ImportError:
                print("⚠️ Tavily not installed, cannot use Tavily search")
        else:
            print("⚠️ TAVILY_API_KEY not set")

        # Check SerpApi availability
        if self.serpapi_key:
            try:
                import serpapi
                self.available_backends.append("serpapi")
                print("✅ SerpApi search engine initialized")
            except ImportError:
                print("⚠️ SerpApi not installed, cannot use SerpApi search")
        else:
            print("⚠️ SERPAPI_API_KEY not set")

        # Determine final backend
        if self.backend == "hybrid":
            if self.available_backends:
                print(f"🔧 Hybrid search mode enabled, available backends: {', '.join(self.available_backends)}")
            else:
                print("⚠️ No search backends available, please configure API keys")
        elif self.backend == "tavily" and "tavily" not in self.available_backends:
            print("⚠️ Tavily not available, check TAVILY_API_KEY configuration")
        elif self.backend == "serpapi" and "serpapi" not in self.available_backends:
            print("⚠️ SerpApi not available, check SERPAPI_API_KEY configuration")
        elif self.backend not in ["tavily", "serpapi", "hybrid"]:
            print("⚠️ Unsupported search backend, will use hybrid mode")
            self.backend = "hybrid"

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        Execute search

        Args:
            parameters: Dict containing input parameter

        Returns:
            Search results
        """
        query = parameters.get("input", "").strip()
        if not query:
            return "Error: Search query cannot be empty"

        print(f"🔍 Executing search: {query}")

        try:
            if self.backend == "hybrid":
                return self._search_hybrid(query)
            elif self.backend == "tavily":
                if "tavily" not in self.available_backends:
                    return self._get_api_config_message()
                return self._search_tavily(query)
            elif self.backend == "serpapi":
                if "serpapi" not in self.available_backends:
                    return self._get_api_config_message()
                return self._search_serpapi(query)
            else:
                return self._get_api_config_message()
        except Exception as e:
            return f"Search error: {str(e)}"

    def _search_hybrid(self, query: str) -> str:
        """Hybrid search - Intelligently selects best source"""
        if not self.available_backends:
            return self._get_api_config_message()

        # Prefer Tavily (AI-optimized search)
        if "tavily" in self.available_backends:
            try:
                print("🎯 Using Tavily for AI-optimized search")
                return self._search_tavily(query)
            except Exception as e:
                print(f"⚠️ Tavily search failed: {e}")
                if "serpapi" in self.available_backends:
                    print("🔄 Switching to SerpApi search")
                    return self._search_serpapi(query)

        elif "serpapi" in self.available_backends:
            try:
                print("🎯 Using SerpApi for Google search")
                return self._search_serpapi(query)
            except Exception as e:
                print(f"⚠️ SerpApi search failed: {e}")

        return "❌ All search sources failed, please check network and API key configuration"

    def _search_tavily(self, query: str) -> str:
        """Search using Tavily"""
        response = self.tavily_client.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=3
        )

        result = f"🎯 Tavily AI Search Results: {response.get('answer', 'No direct answer found')}\n\n"

        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:200]}...\n"
            result += f"    Source: {item.get('url', '')}\n\n"

        return result

    def _search_serpapi(self, query: str) -> str:
        """Search using SerpApi"""
        try:
            from serpapi import SerpApiClient
        except ImportError:
            return "Error: SerpApi not installed, run pip install serpapi"

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "gl": "us",
            "hl": "en",
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        result_text = "🔍 SerpApi Google Search Results:\n\n"

        if "answer_box" in results and "answer" in results["answer_box"]:
            result_text += f"💡 Direct Answer: {results['answer_box']['answer']}\n\n"

        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            result_text += f"📖 Knowledge Graph: {results['knowledge_graph']['description']}\n\n"

        if "organic_results" in results and results["organic_results"]:
            result_text += "🔗 Related Results:\n"
            for i, res in enumerate(results["organic_results"][:3], 1):
                result_text += f"[{i}] {res.get('title', '')}\n"
                result_text += f"    {res.get('snippet', '')}\n"
                result_text += f"    Source: {res.get('link', '')}\n\n"
            return result_text

        return f"Sorry, no information found for '{query}'."

    def _get_api_config_message(self) -> str:
        """Get API configuration message"""
        tavily_key = os.getenv("TAVILY_API_KEY")
        serpapi_key = os.getenv("SERPAPI_API_KEY")

        message = "❌ No search sources available, please check configuration:\n\n"

        message += "1. Tavily API:\n"
        if not tavily_key:
            message += "   ❌ TAVILY_API_KEY env var not set\n"
            message += "   📝 Get at: https://tavily.com/\n"
        else:
            try:
                import tavily
                message += "   ✅ API key configured, package installed\n"
            except ImportError:
                message += "   ❌ API key configured, but need to install: pip install tavily-python\n"

        message += "\n"

        message += "2. SerpAPI:\n"
        if not serpapi_key:
            message += "   ❌ SERPAPI_API_KEY env var not set\n"
            message += "   📝 Get at: https://serpapi.com/\n"
        else:
            try:
                import serpapi
                message += "   ✅ API key configured, package installed\n"
            except ImportError:
                message += "   ❌ API key configured, but need to install: pip install google-search-results\n"

        message += "\nConfiguration:\n"
        message += "- Add to .env: TAVILY_API_KEY=your_key_here\n"
        message += "- Or set env var: export TAVILY_API_KEY=your_key_here\n"
        message += "\nRestart program after configuration."

        return message

    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="input",
                type="string",
                description="Search query keywords",
                required=True
            )
        ]


# Convenience functions
def search(query: str, backend: str = "hybrid") -> str:
    """
    Convenience search function

    Args:
        query: Search query
        backend: Search backend ("hybrid", "tavily", "serpapi")

    Returns:
        Search results
    """
    tool = SearchTool(backend=backend)
    return tool.run({"input": query})

