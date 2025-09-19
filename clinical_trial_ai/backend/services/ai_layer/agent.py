import os
import faiss
import pickle
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


# -----------------------
# Load FAISS Index
# -----------------------
def load_index(file_prefix: str):
    """Load FAISS index and corresponding chunks."""
    index = faiss.read_index(f"{file_prefix}.index")
    with open(f"{file_prefix}_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index.chunks = chunks
    return index, chunks


# -----------------------
# Search Tool
# -----------------------
class SearchTool:
    def __init__(self, vector_store, embedder, k: int = 5):
        self.vs = vector_store
        self.embedder = embedder
        self.k = k

    def __call__(self, query: str) -> List[Dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        distances, indices = self.vs.search(q_emb.reshape(1, -1), self.k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text": self.vs.chunks[idx],
                "metadata": {"id": idx},
                "score": float(distances[0][i])
            })
        return results


# -----------------------
# Helper: format LLM output into numbered bullets
# -----------------------
def format_numbered_list(llm_response, max_items: int = None) -> Dict[str, str]:
    """Return both raw Gemini output and formatted numbered list. Optionally cap items."""
    if hasattr(llm_response, "content"):
        text = llm_response.content
    else:
        text = str(llm_response)

    lines = text.strip().split("\n")
    formatted_lines = []
    count = 1
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("*", "-", "•")):
            line = line[1:].strip()
        formatted_lines.append(f"{count}. {line}")
        count += 1
        if max_items is not None and len(formatted_lines) >= max_items:
            break
    return {"raw": text, "formatted": "\n".join(formatted_lines)}


# -----------------------
# Summary Tool using Gemini
# -----------------------
class SummaryTool:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def __call__(self, query: str, results: List[Dict]) -> Dict[str, str]:
        def clip(text): return text[:800] + "..." if len(text) > 800 else text

        context = "\n\n".join([f"Excerpt: {clip(r['text'])}" for r in results[:5]])

        prompt = f"""You are a clinical research expert.
User query: {query}

Context:
{context}

Summarize key objectives, outcomes, safety signals, and common themes in up to 5 ultra‑concise bullets.
- Max 12 words per bullet
- No repetition
- Prioritize the most decision-relevant findings"""
        response = self.llm.invoke(prompt)
        return format_numbered_list(response, max_items=5)


# -----------------------
# Reason Tool using Gemini
# -----------------------
class ReasonTool:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def __call__(self, query: str, summary: str, results: List[Dict]) -> Dict[str, str]:
        prompt = f"""Critically reason over the summarized evidence to answer:
{query}

Summary:
{summary}

Return the answer as up to 5 ultra‑concise bullets covering:
- Comparative efficacy
- Safety/AE profile contrasts
- Evidence quality/limitations
- Clinical implications/next steps
Avoid redundancy; skip if not supported by evidence."""
        response = self.llm.invoke(prompt)
        return format_numbered_list(response, max_items=5)


# -----------------------
# Final Verdict Tool using Gemini
# -----------------------
class FinalVerdictTool:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    def __call__(self, query: str, results: List[Dict]) -> str:
        def clip(text): return text[:800] + "..." if len(text) > 800 else text

        context = "\n\n".join([f"Excerpt: {clip(r['text'])}" for r in results[:5]])

        prompt = f"""You are a clinical research expert.
User query: {query}

Context from relevant clinical trial reports:
{context}

Provide a **Final Verdict — Top 3 points**:
- Exactly 3 numbered bullets
- 1 line per bullet, max 18 words
- Be decisive, evidence-grounded, and non-redundant"""

        response = self.llm.invoke(prompt)
        return response.content


# -----------------------
# Interactive Loop
# -----------------------
def interactive_loop(api_key: str = None):
    print("Select disease type:")
    print("1. Liver")
    print("2. Fracture")
    print("3. Heart")
    print("4. Diabetes")
    print("5. Cancer")
    choice = input("Enter which you want to know about: ").strip()

    if choice == "1":
        file_prefix = "faiss_index"
        print("✅ Loading Liver index...")
    elif choice == "2":
        file_prefix = "faiss_index_fracture"
        print("✅ Loading Fracture index...")
    elif choice == "3":
        file_prefix = "faiss_index_heart"
        print("✅ Loading Heart index...")
    elif choice == "4":
        file_prefix = "faiss_index_diabetes"
        print("✅ Loading Diabetes index...")
    elif choice == "5":
        file_prefix = "faiss_index_cancer"
        print("✅ Loading Cancer index...")
    else:
        print("❌ Invalid choice. Exiting.")
        return

    embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    index, chunks = load_index(file_prefix)

    search_tool = SearchTool(index, embedder)
    summary_tool = SummaryTool(api_key=api_key)
    reason_tool = ReasonTool(api_key=api_key)
    verdict_tool = FinalVerdictTool(api_key=api_key)

    print("\n✅ Agent ready! Ask about clinical trials. Type 'exit' to quit.\n")

    while True:
        query = input("🔎 Enter query: ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        results = search_tool(query)
        summary_res = summary_tool(query, results)
        reasoning_res = reason_tool(query, summary_res["formatted"], results)
        final_verdict = verdict_tool(query, results)

        print("\n📌 Summary:\n", summary_res["formatted"])
        print("\n🤔 Reasoning:\n", reasoning_res["formatted"])
        print("\n🟢 Gemini Raw Summary Output:\n", summary_res["raw"])
        print("\n🟢 Gemini Raw Reasoning Output:\n", reasoning_res["raw"])
        print("\n✅ Final Verdict (Gemini 2.5):\n", final_verdict)
        print("-" * 60)


if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERROR: Please set your GOOGLE_API_KEY environment variable.")
    else:
        interactive_loop(api_key=api_key)
