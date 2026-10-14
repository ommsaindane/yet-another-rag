from ddgs import DDGS
from models.web import WebResult


class DuckDuckGoSearch:
    def __init__(
        self,
        max_results: int = 5,
    ):
        self.max_results = max_results

    def search(
        self,
        query: str,
    ) -> list[WebResult]:
        if not query.strip():
            return []

        results: list[WebResult] = []
        try:
            with DDGS() as ddgs:
                response = ddgs.text(
                    query=query,
                    max_results=self.max_results,
                )
                for item in response:
                    results.append(
                        WebResult(
                            title=item.get("title", "Untitled"),
                            snippet=item.get("body", ""),
                            url=item.get("href", ""),
                        )
                    )
        except Exception as exc:
            print(f"[Warning] DuckDuckGo search failed for query '{query}': {exc}")

        return results