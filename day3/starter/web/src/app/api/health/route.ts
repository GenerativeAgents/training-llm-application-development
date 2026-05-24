import { NextResponse } from "next/server";

const LLM_API_URL = process.env.LLM_API_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${LLM_API_URL}/api/health`, {
      signal: AbortSignal.timeout(5_000),
    });
    if (!res.ok) {
      return NextResponse.json(
        { status: "error", message: "LLM service returned an error" },
        { status: 503 }
      );
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { status: "error", message: "LLM service is not available" },
      { status: 503 }
    );
  }
}
