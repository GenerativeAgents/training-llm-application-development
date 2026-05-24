import { NextResponse } from "next/server";
import { updateInquiryTopic } from "@/lib/db";
import { withInquiry } from "@/lib/api-helpers";
import { INQUIRY_TOPICS } from "@/lib/constants";

export const POST = (request: Request, context: { params: Promise<{ id: string }> }) =>
  withInquiry(request, context, (inquiry, body) => {
    const { topic } = body;

    if (!topic || !INQUIRY_TOPICS.includes(topic as typeof INQUIRY_TOPICS[number])) {
      return NextResponse.json(
        { error: "有効な分類を指定してください（product, development, other, spam）" },
        { status: 400 },
      );
    }

    const updated = updateInquiryTopic(inquiry.id, topic as typeof INQUIRY_TOPICS[number]);

    return NextResponse.json({
      id: updated!.id,
      topic: updated!.topic,
      updated_at: updated!.updated_at,
    });
  });
