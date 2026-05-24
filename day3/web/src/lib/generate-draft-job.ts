import { markInquiryAsError, updateInquiryWithGeneratedDraft } from "./db";
import { generateDraft } from "./llm";

interface GenerateDraftJobInput {
  inquiryId: string;
  customer_name: string;
  company_name?: string | null;
  content: string;
}

export async function runGenerateDraftJob({
  inquiryId,
  customer_name,
  company_name,
  content,
}: GenerateDraftJobInput): Promise<void> {
  try {
    const draftResult = await generateDraft({
      customer_name,
      company_name: company_name ?? undefined,
      content,
    });

    updateInquiryWithGeneratedDraft(inquiryId, {
      topic: draftResult.topic,
      classification_confidence: draftResult.classification_confidence,
      generated_draft: draftResult.generated_draft,
      quality_alert: draftResult.quality_alert,
      weave_call_id: draftResult.weave_call_id,
    });

    console.log(`AI response generated for inquiry ${inquiryId}`);
  } catch (error) {
    console.error(
      `Failed to generate AI response for ${inquiryId}:`,
      error
    );
    markInquiryAsError(inquiryId);
  }
}
