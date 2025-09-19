import requests
from typing import Dict, Any

class SimpleAPIFetcher:
    """Fetch raw text from different medical/clinical APIs"""

    def __init__(self):
        self.base_urls = {
            "clinicaltrials_gov": "https://clinicaltrials.gov/api/query/full_studies",
            "pubmed_search": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            "pubmed_fetch": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            "fda": "https://api.fda.gov/drug/event.json",
        }

    import requests

    def fetch_clinical_trials_gov(self, search_terms: str, max_results: int = 20) -> str:
        """
        Fetches clinical trial summaries from ClinicalTrials.gov using the modernized API (v2.0).

        Args:
            search_terms (str): The search query for clinical trials.
            max_results (int): The maximum number of results to retrieve.

        Returns:
            str: A string containing the titles and brief summaries of the retrieved clinical trials.
        """
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.cond": search_terms,
            "countTotal": "true",
            "fields": "NCTId,BriefTitle,OverallStatus,HasResults",
            "pageSize": max_results,
            "sort": "LastUpdatePostDate:desc"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            studies = response.json().get("studies", [])
            if not studies:
                return "No studies found."

            summaries = []
            for study in studies:
                title = study.get("briefTitle", "No title available")
                status = study.get("overallStatus", "No status available")
                has_results = study.get("hasResults", False)
                summaries.append(
                    f"Title: {title}\nStatus: {status}\nResults Available: {'Yes' if has_results else 'No'}\n")

            return "\n\n".join(summaries)

        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"

    def fetch_pubmed(self, search_term: str, max_results: int = 20, email: str = None) -> str:
        params = {
            "db": "pubmed",
            "term": search_term,
            "retmax": max_results,
            "retmode": "json",
        }
        if email:
            params["email"] = email
        r = requests.get(self.base_urls["pubmed_search"], params=params)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return ""
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "text",
            "rettype": "abstract"
        }
        r2 = requests.get(self.base_urls["pubmed_fetch"], params=fetch_params)
        r2.raise_for_status()
        return r2.text

    def fetch_fda(self, search_term: str, limit: int = 20) -> str:
        params = {"search": search_term, "limit": limit}
        r = requests.get(self.base_urls["fda"], params=params)
        r.raise_for_status()
        events = r.json().get("results", [])
        texts = []
        for e in events:
            drug = e.get("patient", {}).get("drug", [{}])[0].get("medicinalproduct", "")
            reactions = [rxn.get("reactionmeddrapt", "") for rxn in e.get("patient", {}).get("reaction", [])]
            texts.append(f"{drug}: {', '.join(reactions)}")
        return "\n".join(texts)

    def fetch(self, source: str, **kwargs) -> str:
        if source == "clinicaltrials_gov":
            return self.fetch_clinical_trials_gov(**kwargs)
        elif source == "pubmed":
            return self.fetch_pubmed(**kwargs)
        elif source == "fda":
            return self.fetch_fda(**kwargs)
        else:
            raise ValueError(f"Unknown source: {source}")

    def fetch_all(self, search_config: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        results = {}
        for source, params in search_config.items():
            try:
                results[source] = self.fetch(source, **params)
            except Exception as e:
                results[source] = f"ERROR fetching {source}: {e}"
        return results

# Example usage
if __name__ == "__main__":
    fetcher = SimpleAPIFetcher()

    search_config = {
        "clinicaltrials_gov": {"search_terms": "diabetes", "max_results": 5},
        "pubmed": {"search_term": "diabetes clinical trial", "max_results": 5},
        "fda": {"search_term": "insulin", "limit": 5},
    }

    results = fetcher.fetch_all(search_config)

    for source, text in results.items():
        print(f"\n--- {source.upper()} ---\n{text[:500]} ...")
