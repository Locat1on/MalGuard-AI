export type Verdict = "malicious" | "benign";

export interface AttckTag {
  tactic: string;
  technique: string;
}

export interface DetectionResult {
  filename: string;
  verdict: Verdict;
  confidence: number;
  family: string | null;
  gradcamUrl: string | null;
  attck: AttckTag[];
  llmReport: string;
  modelAgreement: "agree" | "disagree";
  lgbmScore: number;
  mlpScore: number;
  llmVerdict: Verdict | null;
  llmConfidence: number | null;
}

export interface ModelMetric {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface HistoryEntry {
  id: string;
  filename: string;
  verdict: Verdict;
  confidence: number;
  family: string | null;
  timestamp: string;
}
