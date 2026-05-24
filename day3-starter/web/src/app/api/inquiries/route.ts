import { NextResponse } from "next/server";
import { after } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { createInquiry } from "@/lib/db";
import { runGenerateDraftJob } from "@/lib/generate-draft-job";

export async function POST(request: Request) {
  try {
    const body = await request.json();

    // 必須フィールドのバリデーション
    const { customer_name, customer_email, company_name, content } =
      body;

    if (!customer_name || !customer_email || !content) {
      return NextResponse.json(
        { error: "必須項目が入力されていません" },
        { status: 400 }
      );
    }

    // 処理中ステータスでお問い合わせを作成
    const inquiryId = `inq_${uuidv4().replace(/-/g, "").substring(0, 12)}`;

    createInquiry({
      id: inquiryId,
      customer_name,
      customer_email,
      company_name,
      content,
    });

    // Next.js 15 の after() でクライアントへのレスポンス送信後に AI 応答を生成
    after(() =>
      runGenerateDraftJob({
        inquiryId,
        customer_name,
        company_name,
        content,
      })
    );

    return NextResponse.json({
      message: "お問い合わせを受け付けました",
    });
  } catch (error) {
    console.error("Error creating inquiry:", error);
    return NextResponse.json(
      { error: "お問い合わせの送信に失敗しました" },
      { status: 500 }
    );
  }
}
