import { NextResponse } from "next/server";
import { listInquiries, type InquiryStatus, type InquiryTopic } from "@/lib/db";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const status = searchParams.get("status") as InquiryStatus | null;
    const topic = searchParams.get("topic") as InquiryTopic | null;
    const limit = parseInt(searchParams.get("limit") || "20", 10);
    const offset = parseInt(searchParams.get("offset") || "0", 10);

    const result = listInquiries({
      status: status || undefined,
      topic: topic || undefined,
      limit,
      offset,
    });

    return NextResponse.json({
      items: result.items,
      total: result.total,
      limit,
      offset,
    });
  } catch (error) {
    console.error("Error listing inquiries:", error);
    return NextResponse.json(
      { error: "お問い合わせ一覧の取得に失敗しました" },
      { status: 500 }
    );
  }
}
