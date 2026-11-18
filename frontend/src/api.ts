export type AnalysisType =
  | "literature_review"
  | "paper_comparison"
  | "methodology_analysis"
  | "trend_analysis"
  | "research_gaps";

export interface QueryFilters {
  year_from?: number;
  year_to?: number;
  sources: string[];
  domain?: string;
  analysis_type: AnalysisType;
}

export interface ResearchSession {
  session_id: string;
  query: string;
  filters: QueryFilters;
  status: string;
  current_stage: string;
  message: string;
  created_at: string;
  citations: Citation[];
  report: string | null;
  initial_jev: JEVDecision | null;
  final_jev: JEVDecision | null;
  papers: PaperRecord[];
  source_results: SourceResult[];
  refinement_cycles: number;
  retrieval: RetrievalResponse | null;
  answer: GeneratedAnswer | null;
  evaluation: EvaluationResult | null;
  report_document: ResearchReport | null;
  workflow_traces: WorkflowTrace[];
}
export interface WorkflowTrace { stage: string; status: string; started_at: string; duration_ms: number; detail: string; }
export interface Citation { citation_id: string; title: string; source: string; url: string | null; year: number | null; }

export interface RetrievalCandidate {
  chunk: { chunk_id: string; paper_id: string; text: string; page_start: number | null; source: string };
  dense_score: number | null;
  bm25_score: number | null;
  rerank_score: number;
  retrieval_sources: string[];
}

export interface RetrievalResponse {
  query: string;
  dense_candidates: number;
  bm25_candidates: number;
  merged_candidates: number;
  selected_evidence: RetrievalCandidate[];
  context: string;
}

export interface GeneratedAnswer {
  answer: string;
  provider: string;
  model: string;
  citation_ids: string[];
  grounded: boolean;
}

export interface EvaluationResult {
  relevance: number;
  evidence_coverage: number;
  citation_support: number;
  factual_consistency: number;
  completeness: number;
  overall_score: number;
  supported_claims: number;
  unsupported_claims: number;
  warnings: string[];
  passed: boolean;
}

export interface ResearchReport {
  session_id: string;
  title: string;
  executive_summary: string;
  answer: string;
  methodology_comparison: string[];
  findings: string[];
  research_gaps: string[];
  evidence_summary: string[];
  limitations: string[];
  references: { citation_id: string; title: string; source: string; year: number | null }[];
  evaluation_score: number | null;
  generated_at: string;
}

export type MetricSource = "measured" | "estimated";
export interface EvaluationMetric { value: number; source: MetricSource; }
export interface EvaluationSystem {
  name: string;
  answer_quality: EvaluationMetric;
  citation_correctness: EvaluationMetric;
  evidence_coverage: EvaluationMetric;
  papers_retrieved: EvaluationMetric;
  newly_processed_papers: EvaluationMetric;
  duplicate_rate: EvaluationMetric;
  mcp_calls: EvaluationMetric;
  latency_seconds: EvaluationMetric;
  tokens: EvaluationMetric;
  cost: EvaluationMetric;
  refinement_cycles: EvaluationMetric;
}
export type EvaluationMetricKey = Exclude<keyof EvaluationSystem, "name">;
export interface EvaluationResults {
  systems: EvaluationSystem[];
  latency_breakdown: { system: string; mcp: number; ingestion: number; retrieval: number; rerank: number; llm: number; evaluation: number; source: MetricSource }[];
  memory_reuse: { query: string; newly_processed_papers: number; latency_seconds: number; source: MetricSource }[];
  retrieval_quality: { method: string; recall_at_k: EvaluationMetric; mrr: EvaluationMetric; ndcg_at_k: EvaluationMetric }[];
  jev_actions: { action: string; initial: number; final: number; source: MetricSource }[];
  average_refinement_cycles?: EvaluationMetric;
  live_sessions?: { count: number; source: MetricSource; latest_query?: string | null; latest_completed_at?: string | null };
}

export interface WorkflowMetrics {
  session_id: string;
  total_duration_ms: number;
  papers_considered: number;
  retrieved_chunks: number;
  selected_evidence: number;
  citations: number;
  refinement_cycles: number;
  token_usage: number | null;
  estimated_cost: number | null;
  errors: number;
}

export interface JEVDecision {
  action: string;
  explanation: string;
  confidence: number | null;
  implementation: string;
}

export interface PaperRecord {
  paper_id: string;
  title: string;
  authors: string[];
  abstract: string | null;
  year: number | null;
  source: string;
  paper_url: string | null;
  pdf_url: string | null;
}

export interface SourceResult {
  source: string;
  papers: PaperRecord[];
  duration_ms: number;
  error: string | null;
}

export interface SystemStatus {
  status: string;
  environment: string;
  components: { name: string; status: string; detail: string }[];
}

export interface MemoryStats {
  papers: number;
  chunks: number;
  embedded_chunks: number;
  storage: string;
  embedding_model: string;
}

const apiBaseUrl = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getSystemStatus(): Promise<SystemStatus> {
  return request<SystemStatus>("/status");
}

export function getMemoryStats(): Promise<MemoryStats> {
  return request<MemoryStats>("/memory/stats");
}

export function getEvaluationResults(): Promise<EvaluationResults> {
  return fetch(`${apiBaseUrl.replace("/api/v1", "/api")}/evaluation/results`).then(async (response) => {
    if (!response.ok) throw new Error(`Evaluation results failed with status ${response.status}`);
    return response.json() as Promise<EvaluationResults>;
  });
}

export function submitQuery(query: string, filters: QueryFilters): Promise<{ session: ResearchSession }> {
  return request<{ session: ResearchSession }>("/research/query", {
    method: "POST",
    body: JSON.stringify({ query, filters }),
  });
}

export function searchLiterature(sessionId: string, limit = 5): Promise<{ session: ResearchSession }> {
  return request<{ session: ResearchSession }>(`/research/sessions/${sessionId}/literature`, {
    method: "POST",
    body: JSON.stringify({ limit }),
  });
}

export function runCompleteWorkflow(sessionId: string): Promise<{ session: ResearchSession }> {
  return request<{ session: ResearchSession }>(`/research/sessions/${sessionId}/run`, {
    method: "POST",
  });
}

export function getSession(sessionId: string): Promise<{ session: ResearchSession }> {
  return request<{ session: ResearchSession }>(`/research/sessions/${sessionId}`);
}

export function submitFinalDecision(sessionId: string, evaluationSufficient: boolean): Promise<ResearchSession> {
  return request<ResearchSession>(`/research/sessions/${sessionId}/final-decision`, {
    method: "POST",
    body: JSON.stringify({ evaluation_sufficient: evaluationSufficient }),
  });
}

export function retrieveEvidence(sessionId: string, query: string): Promise<{ session: ResearchSession; retrieval: RetrievalResponse }> {
  return request<{ session: ResearchSession; retrieval: RetrievalResponse }>(`/research/sessions/${sessionId}/retrieve`, {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export function generateAnswer(sessionId: string, query?: string): Promise<{ session: ResearchSession; answer: GeneratedAnswer; evaluation: EvaluationResult }> {
  return request<{ session: ResearchSession; answer: GeneratedAnswer; evaluation: EvaluationResult }>(`/research/sessions/${sessionId}/generate`, {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export function createReport(sessionId: string): Promise<{ session: ResearchSession; report: { report: ResearchReport; markdown: string } }> {
  return request<{ session: ResearchSession; report: { report: ResearchReport; markdown: string } }>(`/research/sessions/${sessionId}/report`, { method: "POST" });
}

export function getMetrics(sessionId: string): Promise<WorkflowMetrics> {
  return request<WorkflowMetrics>(`/research/sessions/${sessionId}/metrics`);
}
