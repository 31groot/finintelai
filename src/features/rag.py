import re

from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever

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

quarterly_signals = [
    "q1",
    "q2",
    "q3",
    "q4",
    "quarter 1",
    "quarter 2",
    "quarter 3",
    "quarter 4",
    "earnings call",
    "conference call",
    "transcript",
    "investor presentation",
    "presentation",
]

quarter_patterns = {
    "Q1": [r"\bq1\b", r"\bq\s*1\b", r"\bquarter\s*1\b", r"\bfirst quarter\b"],
    "Q2": [r"\bq2\b", r"\bq\s*2\b", r"\bquarter\s*2\b", r"\bsecond quarter\b"],
    "Q3": [r"\bq3\b", r"\bq\s*3\b", r"\bquarter\s*3\b", r"\bthird quarter\b"],
    "Q4": [r"\bq4\b", r"\bq\s*4\b", r"\bquarter\s*4\b", r"\bfourth quarter\b"],
}

metric_keywords = [
    "revenue",
    "growth"
    "profit",
    "headcount"
    "ebitda",
    "margin",
    "attrition",
    "headcount",
    "employees",
    "employee",
    "bookings"
    "tcv",
    "guidance",
    "deal",
    "deals",
    "cash flow"

]

commentary_keywords = [
    "management",
    "commentary",
    "said",
    "saying",
    "call transcript",
    "transcript",
    "conference call",
    "earnings call",
]

metric_keywords_local = [
    "margin",
    "revenue",
    "growth",
    "attrition",
    "headcount",
    "employee",
    "employees",
    "cash flow",
    "free cash flow",
    "client",
    "geography",
    "segment",
    "vertical",
    "deal",
    "deals",
    "guidance",
    "operating margin",
    "profit",
    "ebitda",
]

constant_currency_query_terms = [
    "constant currency",
    "constant-currency",
    "cc terms",
    "cc growth",
    "in cc",
]

reported_basis_query_terms = [
    "reported basis",
    "reported terms",
    "as reported",
    "in reported",
]

guidance_query_terms = [
    "guidance",
    "outlook",
    "forecast",
    "guided",
    "expects",
    "expected to grow",
    "projection",
    "projections",
]

actual_query_terms = [
    "actual",
    "actuals",
    "reported results",
]

PERMISSIVE_META_VALUES = ("unspecified", "", "general")


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
            fiscal_years.add(f"FY{year[-2:]}")

        fiscal_matches = re.findall(r"\bfiscal[\s_-]?(\d{4})\b", text_lower)
        for year in fiscal_matches:
            fiscal_years.add(f"FY{year[-2:]}")

        return list(fiscal_years)

    def _extract_quarters(self, text):
        text_lower = text.lower()
        found = []

        for quarter, patterns in quarter_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                found.append(quarter)

        return found

    def _infer_doc_types_from_query(self, query):
        q = query.lower()

        if any(keyword in q for keyword in commentary_keywords):
            return ["earnings_call", "annual_report"]

        if any(keyword in q for keyword in metric_keywords_local):
            if any(signal in q for signal in quarterly_signals):
                return ["investor_presentation", "earnings_call", "annual_report"]
            return ["annual_report", "investor_presentation"]

        if any(signal in q for signal in quarterly_signals):
            return ["investor_presentation", "earnings_call", "annual_report"]

        return None

    def _infer_source_kinds_from_query(self, query):
        q = query.lower()

        has_quarter_signal = any(signal in q for signal in quarterly_signals)

        if any(keyword in q for keyword in commentary_keywords):
            return ["management_commentary"]

        if has_quarter_signal:
            return None

        if any(keyword in q for keyword in metric_keywords_local):
            return ["annual_filing", "metrics_summary"]

        return None

    def _extract_statement_type(self, query):
        q = query.lower()

        if "standalone" in q:
            return "standalone"
        if "consolidated" in q:
            return "consolidated"

        return None


    def _extract_basis(self, query):

        q = query.lower()

        if any(term in q for term in constant_currency_query_terms):
            return "constant_currency"

        if any(term in q for term in reported_basis_query_terms):
            return "reported"

        return None

    def _extract_figure_type(self, query):
 
        q = query.lower()

        if any(term in q for term in guidance_query_terms):
            return "guidance"

        if any(term in q for term in actual_query_terms):
            return "actual"

        return None

    def _is_comparison_query(self, query):
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in compare_keywords)

    def _extract_metrics_from_query(self, query):
        q = query.lower()
        found = []

        for metric in metric_keywords:
            if metric in q:
                found.append(metric)
        return found

    def _is_financial_kpi_query(self, query, subqueries):
        companies = self.decomposer._find_companies(query.lower())
        metrics = self._extract_metrics_from_query(query)
        fiscal_years = self._extract_fiscal_years(query)
        is_comparison = self._is_comparison_query(query)

        if is_comparison:
            return False

        if len(subqueries) != 1:
            return False

        if len(companies) != 1:
            return False

        if not metrics:
            return False

        if not fiscal_years:
            return False

        return True

    def _build_filters_for_subquery(
        self, query, subquery, memory_companies=None, statement_type=None
    ):
        filters = {}

        companies = self._get_companies_for_subquery(
            subquery, fallback_companies=memory_companies
        )
        if companies:
            filters["companies"] = companies

        fiscal_years = self._extract_fiscal_years(subquery)
        if not fiscal_years:
            fiscal_years = self._extract_fiscal_years(query)
        if fiscal_years:
            filters["fiscal_years"] = fiscal_years

        quarters = self._extract_quarters(subquery)
        if not quarters:
            quarters = self._extract_quarters(query)
        if quarters:
            filters["quarters"] = quarters

        doc_types = self._infer_doc_types_from_query(subquery)
        if not doc_types:
            doc_types = self._infer_doc_types_from_query(query)
        if doc_types:
            filters["doc_types"] = doc_types

        source_kinds = self._infer_source_kinds_from_query(subquery)
        if not source_kinds:
            source_kinds = self._infer_source_kinds_from_query(query)
        if source_kinds:
            filters["source_kinds"] = source_kinds

        if statement_type and not filters.get("quarters"):
            filters["statement_type"] = statement_type

        basis = self._extract_basis(subquery) or self._extract_basis(query)
        if basis:
            filters["basis"] = basis

        figure_type = self._extract_figure_type(subquery) or self._extract_figure_type(
            query
        )
        if figure_type:
            filters["figure_type"] = figure_type

        return filters

    def _apply_local_filters(self, docs, metadata, filters):
        if not filters:
            return docs, metadata

        filtered_docs = []
        filtered_metadata = []

        company_filters = [c.lower() for c in filters.get("companies", [])]
        fy_filters = [fy.upper() for fy in filters.get("fiscal_years", [])]
        doc_type_filters = [d.lower() for d in filters.get("doc_types", [])]
        quarter_filters = [q.upper() for q in filters.get("quarters", [])]
        source_kind_filters = [s.lower() for s in filters.get("source_kinds", [])]
        statement_type_filter = str(filters.get("statement_type", "")).lower().strip()
        basis_filter = str(filters.get("basis", "")).lower().strip()
        figure_type_filter = str(filters.get("figure_type", "")).lower().strip()

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

            if keep and doc_type_filters:
                meta_doc_type = str(meta.get("doc_type", "")).lower()
                if meta_doc_type not in doc_type_filters:
                    keep = False

            if keep and quarter_filters:
                meta_quarter = str(meta.get("quarter", "")).upper()
                if meta_quarter not in quarter_filters:
                    keep = False

            if keep and source_kind_filters:
                meta_source_kind = str(meta.get("source_kind", "")).lower()
                if meta_source_kind and meta_source_kind not in source_kind_filters:
                    keep = False

            if keep and statement_type_filter:
                meta_statement_type = str(meta.get("statement_type", "")).lower().strip()
                if meta_statement_type not in (statement_type_filter, "general", ""):
                    keep = False

            if keep and basis_filter:
                meta_basis = str(meta.get("basis", "")).lower().strip()
                if meta_basis not in PERMISSIVE_META_VALUES:
                    if meta_basis != basis_filter:
                        keep = False

            if keep and figure_type_filter:
                meta_figure_type = str(meta.get("figure_type", "")).lower().strip()
                if meta_figure_type not in PERMISSIVE_META_VALUES:
                    if meta_figure_type != figure_type_filter:
                        keep = False

            if keep:
                filtered_docs.append(doc)
                filtered_metadata.append(meta)

        return filtered_docs, filtered_metadata

    def _deduplicate_docs(self, docs, metadata):
        unique_docs = []
        unique_metadata = []

        for doc, meta in zip(docs, metadata):
            if doc not in unique_docs:
                unique_docs.append(doc)
                unique_metadata.append(meta)

        return unique_docs, unique_metadata

    def _get_rerank_top_k(self, is_comparison, num_subqueries):
        if is_comparison:
            if num_subqueries <= 2:
                return 10
            if num_subqueries <= 4:
                return 8
            return 6

        if num_subqueries == 1:
            return 12
        if num_subqueries == 2:
            return 10
        if num_subqueries == 3:
            return 8
        return 6

    def _relax_filters_stepwise(self, docs, metadata, filters):

        strict_docs, strict_meta = self._apply_local_filters(docs, metadata, filters)
        if strict_docs:
            return strict_docs, strict_meta

        relaxed0 = dict(filters)
        relaxed0.pop("basis", None)
        relaxed0.pop("figure_type", None)

        relaxed_docs0, relaxed_meta0 = self._apply_local_filters(
            docs, metadata, relaxed0
        )
        if relaxed_docs0:
            return relaxed_docs0, relaxed_meta0

        relaxed = dict(relaxed0)
        relaxed.pop("source_kinds", None)

        relaxed_docs, relaxed_meta = self._apply_local_filters(docs, metadata, relaxed)
        if relaxed_docs:
            return relaxed_docs, relaxed_meta

        relaxed2 = dict(relaxed)
        relaxed2.pop("doc_types", None)

        relaxed_docs2, relaxed_meta2 = self._apply_local_filters(
            docs, metadata, relaxed2
        )
        if relaxed_docs2:
            return relaxed_docs2, relaxed_meta2

        relaxed3 = dict(relaxed2)
        relaxed3.pop("quarters", None)

        relaxed_docs3, relaxed_meta3 = self._apply_local_filters(
            docs, metadata, relaxed3
        )
        if relaxed_docs3:
            return relaxed_docs3, relaxed_meta3

        relaxed4 = dict(relaxed3)
        relaxed4.pop("statement_type", None)

        relaxed_docs4, relaxed_meta4 = self._apply_local_filters(
            docs, metadata, relaxed4
        )
        if relaxed_docs4:
            return relaxed_docs4, relaxed_meta4

        return [], []

    BROADENABLE_FILTERS = ("doc_types", "source_kinds", "basis", "figure_type")

    def _broaden_filters(self, filters):
        broadened = dict(filters)
        for key in self.BROADENABLE_FILTERS:
            broadened.pop(key, None)
        return broadened

    def get_context(self, query, memory_companies=None, broaden=False):
        decomposed = self.decomposer.decompose(query, memory_companies=memory_companies)

        if isinstance(decomposed, dict):
            subqueries = decomposed.get("subqueries", [])
            statement_type = decomposed.get("statement_type")
        else:
            subqueries = decomposed
            statement_type = self._extract_statement_type(query)

        is_comparison = self._is_comparison_query(query)

        balanced_docs = []
        balanced_metadata = []

        for subquery in subqueries:
            filters = self._build_filters_for_subquery(
                query=query,
                subquery=subquery,
                memory_companies=memory_companies,
                statement_type=statement_type,
            )

            if broaden:
                filters = self._broaden_filters(filters)

            base_n_results = 24 if is_comparison else 20
            n_results = base_n_results * 2 if broaden else base_n_results

            results = self.retriever.search(
                query=subquery, n_results=n_results, filters=filters
            )

            retrieved_docs = results["documents"]
            retrieved_metadata = results["metadata"]

            docs, metadata = self._deduplicate_docs(retrieved_docs, retrieved_metadata)

            docs, metadata = self._relax_filters_stepwise(docs, metadata, filters)

    
            has_strict_filter = bool(
                filters.get("fiscal_years")
                or filters.get("quarters")
                or filters.get("companies")
                or filters.get("statement_type")
            )

            if not docs and not has_strict_filter:
                docs, metadata = self._deduplicate_docs(
                    retrieved_docs, retrieved_metadata
                )

            if not docs:
                continue

            rerank_top_k = self._get_rerank_top_k(
                is_comparison=is_comparison, num_subqueries=len(subqueries)
            )
            if broaden:
                rerank_top_k = int(rerank_top_k * 1.5)

            rerank_top_k = min(rerank_top_k, len(docs))

            top_results = self.reranker.rerank(
                query=subquery, documents=docs, metadata=metadata, top_k=rerank_top_k
            )

            reranked_docs = []
            reranked_metadata = []

            for doc, meta, _score in top_results:
                reranked_docs.append(doc)
                reranked_metadata.append(meta)

            for doc, meta in zip(reranked_docs, reranked_metadata):
                balanced_docs.append(doc)
                balanced_metadata.append(meta)

        final_docs, final_metadata = self._deduplicate_docs(
            balanced_docs, balanced_metadata
        )

        context = "\n\n".join(final_docs)

        return {
            "query": query,
            "subqueries": subqueries,
            "context": context,
            "documents": final_docs,
            "metadata": final_metadata,
        }