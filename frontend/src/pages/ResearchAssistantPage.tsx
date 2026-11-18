import { FormEvent, useState } from "react";
import { createReport, getSession, QueryFilters, runCompleteWorkflow, submitQuery, ResearchSession } from "../api";
import { QueryForm } from "../components/QueryForm";
import { ReportView } from "../components/ReportView";
import { WorkflowStepper } from "../components/WorkflowStepper";

const initialFilters: QueryFilters = { sources: ["arxiv", "semantic_scholar"], analysis_type: "literature_review" };
export function ResearchAssistantPage() {
  const [query, setQuery] = useState(""); const [filters, setFilters] = useState(initialFilters); const [session, setSession] = useState<ResearchSession | null>(null); const [loading, setLoading] = useState(false); const [error, setError] = useState(""); const [latency, setLatency] = useState<number | null>(null);
  async function submit(event: FormEvent<HTMLFormElement>) { event.preventDefault(); setLoading(true); setError(""); const workflowStartedAt = performance.now(); try { const created = await submitQuery(query, filters); setSession(created.session); let current = (await runCompleteWorkflow(created.session.session_id)).session; setSession(current); while (current.status === "processing") { await new Promise((resolve) => setTimeout(resolve, 700)); current = (await getSession(created.session.session_id)).session; setSession(current); } setLatency(performance.now() - workflowStartedAt); if (current.status === "failed") throw new Error(current.message); } catch (reason) { setError(reason instanceof Error ? reason.message : "Workflow failed."); } finally { setLoading(false); } }
  async function exportReport() { if (!session) return; const result = await createReport(session.session_id); const blob = new Blob([result.report.markdown], { type: "text/markdown" }); const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = `research-report-${session.session_id.slice(0, 8)}.md`; link.click(); URL.revokeObjectURL(link.href); }
  async function copyReport() { if (session?.report) await navigator.clipboard.writeText(session.report); }
  return <main className="page-shell"><div className="page-intro"><p className="eyebrow">MCP-BASED AGENTIC RESEARCH ASSISTANT</p><h1>From a question to a defensible evidence trail.</h1><p>Acquire literature, reuse persistent memory, retrieve and rerank evidence, evaluate the answer, and produce a traceable report.</p></div><QueryForm query={query} filters={filters} loading={loading} onQueryChange={setQuery} onFiltersChange={setFilters} onSubmit={submit}/>{error && <p className="error">{error}</p>}<WorkflowStepper session={session} latency={latency}/><ReportView session={session} onCopy={copyReport} onExport={exportReport}/></main>;
}
