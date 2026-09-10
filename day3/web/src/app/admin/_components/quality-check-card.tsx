import type { Inquiry } from "@/lib/db";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function QualityCheckCard({ inquiry }: { inquiry: Inquiry }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          {inquiry.generated_draft ? "AI品質チェック" : "スパム検知"}
          {inquiry.quality_alert && (
            <Badge variant="destructive">品質要注意</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {inquiry.generated_draft ? (
          <div className="grid grid-cols-2 gap-3 text-sm">
            {inquiry.classification_confidence != null && (
              <div>
                <span className="text-muted-foreground">分類の確信度:</span>{" "}
                <span>
                  {(inquiry.classification_confidence * 100).toFixed(0)}%
                </span>
              </div>
            )}
            <div>
              <span className="text-muted-foreground">丁寧さ:</span>{" "}
              <span
                className={
                  inquiry.generated_draft.quality_scores.politeness === "NG"
                    ? "text-destructive font-medium"
                    : "text-green-600 dark:text-green-400 font-medium"
                }
              >
                {inquiry.generated_draft.quality_scores.politeness}
              </span>
              {inquiry.generated_draft.quality_scores.politeness_reason && (
                <p
                  className={`text-xs mt-0.5 ${inquiry.generated_draft.quality_scores.politeness === "NG" ? "text-destructive" : "text-muted-foreground"}`}
                >
                  {inquiry.generated_draft.quality_scores.politeness_reason}
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="text-sm space-y-2">
            <p>このお問い合わせはスパムと判定されました。対応不要です。</p>
            {inquiry.classification_confidence != null && (
              <p className="text-muted-foreground">
                分類の確信度:{" "}
                {(inquiry.classification_confidence * 100).toFixed(0)}%
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
