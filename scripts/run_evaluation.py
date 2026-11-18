"""Run evaluation queries and merge measured workflow telemetry into evaluation_results.json.

Human-scored fields stay estimated until manual_scores.json contains values.
"""
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000/api/v1"


def call(path: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(f"{API}{path}", data=body, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def main() -> None:
    queries = json.loads((ROOT / "evaluation_queries.json").read_text())["queries"]
    results = json.loads((ROOT / "evaluation_results.json").read_text())
    proposed = next(item for item in results["systems"] if item["name"].startswith("Proposed"))
    measured = []
    for query in queries:
        started = time.perf_counter()
        created = call("/research/query", "POST", {"query": query, "filters": {"sources": ["arxiv", "semantic_scholar"], "analysis_type": "literature_review"}})
        session_id = created["session"]["session_id"]
        call(f"/research/sessions/{session_id}/run", "POST")
        while True:
            session = call(f"/research/sessions/{session_id}")["session"]
            if session["status"] != "processing":
                break
            time.sleep(1)
        retrieval = session.get("retrieval") or {}
        papers = len(session.get("papers", []))
        measured.append({
            "query": query,
            "newly_processed_papers": papers,
            "retrieved_chunks": len(retrieval.get("selected_evidence", [])),
            "latency_seconds": round(time.perf_counter() - started, 3),
            "source": "measured",
        })
    results["memory_reuse"] = measured
    if measured:
        proposed["latency_seconds"] = {"value": sum(item["latency_seconds"] for item in measured) / len(measured), "source": "measured"}
        proposed["papers_retrieved"] = {"value": sum(item["newly_processed_papers"] for item in measured) / len(measured), "source": "measured"}
    (ROOT / "evaluation_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
