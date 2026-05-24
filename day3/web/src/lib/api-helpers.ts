import { NextResponse } from "next/server";
import { getInquiryById, type Inquiry } from "./db";

type RouteContext = { params: Promise<{ id: string }> };

export async function withInquiry(
  request: Request,
  context: RouteContext,
  handler: (inquiry: Inquiry, body: Record<string, unknown>) => Promise<NextResponse> | NextResponse,
): Promise<NextResponse> {
  try {
    const { id } = await context.params;
    const inquiry = getInquiryById(id);
    if (!inquiry) {
      return NextResponse.json(
        { error: "お問い合わせが見つかりません" },
        { status: 404 },
      );
    }
    const body = await request.json().catch(() => ({}));
    return await handler(inquiry, body as Record<string, unknown>);
  } catch (error) {
    console.error("API error:", error);
    return NextResponse.json(
      { error: "処理に失敗しました" },
      { status: 500 },
    );
  }
}
