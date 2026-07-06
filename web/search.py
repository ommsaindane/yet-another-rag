from models.web import WebResult

from duckduckgo_search import DDGS


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

        results: list[WebResult] = []

        with DDGS() as ddgs:
            response = ddgs.text(
                keywords=query,
                max_results=self.max_results,
            )

            for item in response:
                results.append(
                    WebResult(
                        title=item["title"],
                        snippet=item["body"],
                        url=item["href"],
                    )
                )

        return results
