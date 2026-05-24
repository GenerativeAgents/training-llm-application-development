import { NextResponse } from "next/server";
import { getInquiryById } from "@/lib/db";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const inquiry = getInquiryById(id);

    if (!inquiry) {
      return NextResponse.json(
        { error: "お問い合わせが見つかりません" },
        { status: 404 }
      );
    }

    return NextResponse.json(inquiry);
  } catch (error) {
    console.error("Error getting inquiry:", error);
    return NextResponse.json(
      { error: "お問い合わせの取得に失敗しました" },
      { status: 500 }
    );
  }
}
