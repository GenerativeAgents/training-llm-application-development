import { NextResponse } from "next/server";
import { markAsSent } from "@/lib/db";
import { withInquiry } from "@/lib/api-helpers";

export const POST = (request: Request, context: { params: Promise<{ id: string }> }) =>
  withInquiry(request, context, async (inquiry, body) => {
    const { subject, body: responseBody } = body as { subject?: string; body?: string };

    if (!subject || !responseBody) {
      return NextResponse.json(
        { error: "件名と本文は必須です" },
        { status: 400 },
      );
    }

    const updated = markAsSent(inquiry.id, {
      subject,
      body: responseBody,
      edit_distance: null,
      operator_edited_topic: null,
    });

    return NextResponse.json({
      id: updated!.id,
      status: updated!.status,
      sent_at: updated!.sent_at,
    });
  });
