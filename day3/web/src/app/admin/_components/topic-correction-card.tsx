import type { Inquiry } from "@/lib/db";
import { topicLabels, INQUIRY_TOPICS } from "@/lib/constants";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface TopicCorrectionCardProps {
  inquiry: Inquiry;
  isUpdatingTopic: boolean;
  onTopicChange: (topic: string) => void;
}

export function TopicCorrectionCard({
  inquiry,
  isUpdatingTopic,
  onTopicChange,
}: TopicCorrectionCardProps) {
  if (!inquiry.topic) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">分類修正</CardTitle>
        {inquiry.original_topic &&
          inquiry.topic !== inquiry.original_topic && (
            <CardDescription>
              AI元分類: {topicLabels[inquiry.original_topic]}
            </CardDescription>
          )}
      </CardHeader>
      <CardContent>
        <Select
          value={inquiry.topic}
          onValueChange={onTopicChange}
          disabled={isUpdatingTopic || inquiry.status === "sent"}
        >
          <SelectTrigger className="w-48">
            <SelectValue placeholder="分類を選択" />
          </SelectTrigger>
          <SelectContent>
            {INQUIRY_TOPICS.map((topic) => (
              <SelectItem key={topic} value={topic}>
                {topicLabels[topic]}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </CardContent>
    </Card>
  );
}
