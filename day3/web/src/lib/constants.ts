import type { InquiryStatus, InquiryTopic } from "./db";

export const INQUIRY_TOPICS = [
  "development",
  "product",
  "other",
  "spam",
] as const satisfies readonly InquiryTopic[];

export const statusLabels: Record<InquiryStatus, string> = {
  processing: "処理中",
  draft: "下書き",
  sent: "送信済み",
  error: "エラー",
};

export const statusColors: Record<InquiryStatus, string> = {
  processing: "bg-gray-100 text-gray-700",
  draft: "bg-green-100 text-green-800",
  sent: "bg-gray-500 text-gray-50",
  error: "bg-red-100 text-red-800",
};

export const topicLabels: Record<InquiryTopic, string> = {
  development: "開発支援",
  product: "プロダクト",
  other: "その他",
  spam: "スパム",
};

export function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleString("ja-JP", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}
