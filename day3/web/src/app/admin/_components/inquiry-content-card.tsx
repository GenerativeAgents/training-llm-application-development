import type { Inquiry } from "@/lib/db";
import { formatDate } from "@/lib/constants";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function InquiryContentCard({ inquiry }: { inquiry: Inquiry }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">お問い合わせ内容</CardTitle>
        <CardDescription>
          {formatDate(inquiry.created_at)} 受付
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">お名前:</span>{" "}
            {inquiry.customer_name}
          </div>
          <div>
            <span className="text-gray-500">メール:</span>{" "}
            {inquiry.customer_email}
          </div>
          {inquiry.company_name && (
            <div className="col-span-2">
              <span className="text-gray-500">会社名:</span>{" "}
              {inquiry.company_name}
            </div>
          )}
        </div>
        <div>
          <span className="text-gray-500 text-sm">内容:</span>
          <p className="whitespace-pre-wrap mt-1">{inquiry.content}</p>
        </div>
      </CardContent>
    </Card>
  );
}
