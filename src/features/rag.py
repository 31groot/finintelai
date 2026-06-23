import re

from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.retrieval.query_decomposer import QueryDecomposer


compare_keywords = [
    "compare",
    "comparison",
    "rank",
    "ranking",
    "highest",
    "lowest",
    "best",
    "worst",
    "versus",
    "vs",
]


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.decomposer = QueryDecomposer()

    def _get_companies_for_subquery(self, subquery, fallback_companies=None):
        subquery_lower = subquery.lower()
        companies = self.decomposer._find_companies(subquery_lower)

        if companies:
            return companies

        return list(fallback_companies) if fallback_companies else []

    def _extract_fiscal_years(self, text):
        text_lower = text.lower()
        fiscal_years = set()

        fy_matches = re.findall(r"\bfy[\s_-]?(\d{2,4})\b", text_lower)
        for year in fy_matches:
            if len(year) == 2:
                fiscal_years.add(f"FY{year}")
            else:
                fiscal_years.add(f"FY{year[-2:]}")

        fiscal_matches = re.findall(r"\bfiscal[\s_-]?(\d{4})\b", text_lower)
        for year in fiscal_matches:
            fiscal_years.add(f"FY{year[-2:]}")

        def fy_key(fy):
            return int(fy[-2:])

        return sorted(fiscal_years, key=fy_key, reverse=True)

    def _is_comparison_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in compare_keywords)

    def _build_filters_for_subquery(self, query, subquery, memory_companies=None):
        filters = {}

        companies = self._get_companies_for_subquery(
            subquery,
            fallback_companies=memory_companies
        )
        if companies:
            filters["companies"] = companies

        # Prefer fiscal year explicitly attached to the decomposed subquery.
        # If none exists there, fall back to the original query.
        fiscal_years = self._extract_fiscal_years(subquery)
        if not fiscal_years:
            fiscal_years = self._extract_fiscal_years(query)

        if fiscal_years:
            filters["fiscal_years"] = fiscal_years

        return filters

    def _apply_local_filters(self, docs, metadata, filters):
        if not filters:
            return docs, metadata

        filtered_docs = []
        filtered_metadata = []

        company_filters = [c.lower() for c in filters.get("companies", [])]
        fy_filters = [fy.upper() for fy in filters.get("fiscal_years", [])]

        for doc, meta in zip(docs, metadata):
            meta = meta or {}
            keep = True

            if company_filters:
                meta_company = str(meta.get("company", "")).lower()
                source_name = str(meta.get("source", "")).lower()

                if meta_company:
                    if meta_company not in company_filters:
                        keep = False
                else:
                    if not any(company in source_name for company in company_filters):
                        keep = False

            if keep and fy_filters:
                meta_fy = str(meta.get("fiscal_year", "")).upper()
                if meta_fy not in fy_filters:
                    keep = False

            if keep:
                filtered_docs.append(doc)
                filtered_metadata.append(meta)

        return filtered_docs, filtered_metadata

    def _deduplicate_docs(self, docs, metadata):
        seen_docs = set()
        unique_docs = []
        unique_metadata = []

        for doc, meta in zip(docs, metadata):
            if doc not in seen_docs:
                seen_docs.add(doc)
                unique_docs.append(doc)
                unique_metadata.append(meta)

        return unique_docs, unique_metadata

    def _get_rerank_top_k(self, is_comparison: bool, num_subqueries: int) -> int:
        if is_comparison:
            if num_subqueries == 1:
                return 20
            if num_subqueries == 2:
                return 12
            if num_subqueries == 3:
                return 8
            return 6

        if num_subqueries == 1:
            return 15
        if num_subqueries == 2:
            return 8
        if num_subqueries == 3:
            return 5
        return 4

    def get_context(self, query, memory_companies=None):
        subqueries = self.decomposer.decompose(
            query,
            memory_companies=memory_companies
        )

        is_comparison = self._is_comparison_query(query)

        balanced_docs = []
        balanced_metadata = []

        for subquery in subqueries:
            filters = self._build_filters_for_subquery(
                query=query,
                subquery=subquery,
                memory_companies=memory_companies
            )

            results = self.retriever.search(
                query=subquery,
                n_results=22 if is_comparison else 15,
                filters=filters
            )

            retrieved_docs = results["documents"]
            retrieved_metadata = results["metadata"]

            docs, metadata = self._deduplicate_docs(
                retrieved_docs,
                retrieved_metadata
            )

            docs, metadata = self._apply_local_filters(
                docs,
                metadata,
                filters
            )
            has_strict_year_filter = bool(filters.get("fiscal_years"))

            if not docs and not has_strict_year_filter:
                docs, metadata = self._deduplicate_docs(
                    retrieved_docs,
                    retrieved_metadata
                )

            if not docs:
                continue

            rerank_top_k = min(
                self._get_rerank_top_k(
                    is_comparison=is_comparison,
                    num_subqueries=len(subqueries)
                ),
                len(docs)
            )

            top_results = self.reranker.rerank(
                query=subquery,
                documents=docs,
                metadata=metadata,
                top_k=rerank_top_k
            )

            for doc, meta, _score in top_results:
                balanced_docs.append(doc)
                balanced_metadata.append(meta)

        final_docs, final_metadata = self._deduplicate_docs(
            balanced_docs,
            balanced_metadata
        )

        context = "\n\n".join(final_docs)

        return {
            "query": query,
            "subqueries": subqueries,
            "context": context,
            "documents": final_docs,
            "metadata": final_metadata
        }