import { NextResponse } from "next/server";
import { markAsSent } from "@/lib/db";
import { withInquiry } from "@/lib/api-helpers";

const LLM_API_URL = process.env.LLM_API_URL ?? "http://localhost:8000";

export const POST = (request: Request, context: { params: Promise<{ id: string }> }) =>
  withInquiry(request, context, async (inquiry, body) => {
    const { subject, body: responseBody } = body as { subject?: string; body?: string };

    if (!subject || !responseBody) {
      return NextResponse.json(
        { error: "件名と本文は必須です" },
        { status: 400 },
      );
    }

    let editDistance: number | null = null;
    let operatorEditedTopic: boolean = inquiry.topic !== inquiry.original_topic;

    if (inquiry.run_id && inquiry.original_topic && inquiry.topic) {
      const fbRes = await fetch(`${LLM_API_URL}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          run_id: inquiry.run_id,
          ai_body: inquiry.generated_draft?.body ?? "",
          final_body: responseBody,
          original_topic: inquiry.original_topic,
          current_topic: inquiry.topic,
        }),
      });
      const fbData = await fbRes.json();
      editDistance = fbData.edit_distance;
      operatorEditedTopic = fbData.operator_edited_topic ?? operatorEditedTopic;
    }

    const updated = markAsSent(inquiry.id, {
      subject,
      body: responseBody,
      edit_distance: editDistance,
      operator_edited_topic: operatorEditedTopic,
    });

    return NextResponse.json({
      id: updated!.id,
      status: updated!.status,
      sent_at: updated!.sent_at,
    });
  });
