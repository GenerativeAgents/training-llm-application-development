import type { GeneratedDraft, InquiryTopic, QualityScores } from "./db";

interface GenerateRequest {
  customer_name: string;
  company_name?: string;
  content: string;
}

interface GenerateResponse {
  topic: InquiryTopic;
  classification_confidence: number;
  generated_draft: GeneratedDraft | null;
  quality_alert: boolean;
  weave_call_id: string | null;
}

interface FastAPIResponse {
  topic: InquiryTopic;
  classification_confidence: number;
  generated_draft: {
    subject: string;
    body: string;
    quality_scores: QualityScores;
  } | null;
  weave_call_id: string | null;
}

const LLM_API_URL = process.env.LLM_API_URL ?? "http://localhost:8000";

export async function generateDraft(
  request: GenerateRequest
): Promise<GenerateResponse> {
  const response = await fetch(`${LLM_API_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal: AbortSignal.timeout(300_000),
  });

  if (!response.ok) {
    throw new Error(`LLM API error: ${response.status} ${response.statusText}`);
  }

  const data: FastAPIResponse = await response.json();

  const qualityAlert = data.generated_draft
    ? data.generated_draft.quality_scores.politeness === "NG"
    : false;

  return {
    topic: data.topic,
    classification_confidence: data.classification_confidence,
    generated_draft: data.generated_draft,
    quality_alert: qualityAlert,
    weave_call_id: data.weave_call_id,
  };
}
