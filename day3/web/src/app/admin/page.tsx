"use client";

import type { InquiryTopic } from "@/lib/db";
import {
  statusLabels,
  statusColors,
  topicLabels,
  formatDate,
  INQUIRY_TOPICS,
} from "@/lib/constants";
import { useInquiries } from "./_hooks/use-inquiries";
import { useDraftEditor } from "./_hooks/use-draft-editor";
import { InquiryContentCard } from "./_components/inquiry-content-card";
import { QualityCheckCard } from "./_components/quality-check-card";
import { TopicCorrectionCard } from "./_components/topic-correction-card";
import { ResponseCard } from "./_components/response-card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { HealthCheckModal } from "@/components/health-check-modal";

const statusFilterOptions: { value: string; label: string }[] = [
  { value: "all", label: "すべて" },
  { value: "processing", label: statusLabels.processing },
  { value: "draft", label: statusLabels.draft },
  { value: "sent", label: statusLabels.sent },
  { value: "error", label: statusLabels.error },
];

export default function AdminInquiriesPage() {
  const {
    inquiries,
    selectedInquiry,
    setSelectedInquiry,
    statusFilter,
    setStatusFilter,
    topicFilter,
    setTopicFilter,
    isLoading,
    error,
    fetchInquiries,
    fetchInquiryDetail,
    retryGeneration,
    isRetrying,
  } = useInquiries();

  const {
    editSubject,
    setEditSubject,
    editBody,
    setEditBody,
    isSaving,
    isSending,
    isUpdatingTopic,
    saveMessage,
    handleSaveDraft,
    handleSend,
    handleTopicChange,
  } = useDraftEditor({
    selectedInquiry,
    setSelectedInquiry,
    fetchInquiries,
    fetchInquiryDetail,
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <HealthCheckModal />
        <p className="text-gray-500">読み込み中...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <HealthCheckModal />
        <Alert variant="destructive">
          <AlertTitle>エラー</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <HealthCheckModal />
      <div className="border-b bg-white px-6 py-4">
        <h1 className="text-2xl font-bold">お問い合わせ管理</h1>
      </div>

      <div className="flex h-[calc(100vh-73px)]">
        {/* Left Panel - Inquiry List */}
        <div className="w-1/2 border-r bg-white overflow-hidden flex flex-col">
          <div className="p-4 border-b">
            <div className="flex gap-4">
              <div className="space-y-1">
                <Label>ステータス</Label>
                <Select value={statusFilter} onValueChange={setStatusFilter}>
                  <SelectTrigger className="w-48">
                    <SelectValue placeholder="ステータス" />
                  </SelectTrigger>
                  <SelectContent>
                    {statusFilterOptions.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1">
                <Label>分類</Label>
                <Select value={topicFilter} onValueChange={setTopicFilter}>
                  <SelectTrigger className="w-48">
                    <SelectValue placeholder="分類" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">すべて</SelectItem>
                    {INQUIRY_TOPICS.map((topic) => (
                      <SelectItem key={topic} value={topic}>
                        {topicLabels[topic]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>顧客名</TableHead>
                  <TableHead>分類</TableHead>
                  <TableHead>ステータス</TableHead>
                  <TableHead>受付日時</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {inquiries.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center py-8">
                      お問い合わせがありません
                    </TableCell>
                  </TableRow>
                ) : (
                  inquiries.map((inquiry) => (
                    <TableRow
                      key={inquiry.id}
                      className={`cursor-pointer hover:bg-gray-50 ${
                        selectedInquiry?.id === inquiry.id ? "bg-blue-50" : ""
                      }`}
                      onClick={() => fetchInquiryDetail(inquiry.id)}
                    >
                      <TableCell>
                        <div>
                          <div className="font-medium">
                            {inquiry.customer_name}
                          </div>
                          {inquiry.company_name && (
                            <div className="text-sm text-gray-500">
                              {inquiry.company_name}
                            </div>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        {inquiry.topic && (
                          <Badge variant="outline">
                            {topicLabels[inquiry.topic as InquiryTopic]}
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge className={statusColors[inquiry.status]}>
                          {statusLabels[inquiry.status]}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-sm text-gray-500">
                        {formatDate(inquiry.created_at)}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </div>

        {/* Right Panel - Inquiry Detail */}
        <div className="w-1/2 overflow-auto p-6">
          {selectedInquiry ? (
            <div className="space-y-6">
              <InquiryContentCard inquiry={selectedInquiry} />

              {selectedInquiry.status === "processing" && (
                <Alert>
                  <AlertTitle>AI処理中</AlertTitle>
                  <AlertDescription>
                    AI回答を生成しています。しばらくお待ちください...
                  </AlertDescription>
                </Alert>
              )}

              {selectedInquiry.status === "error" && (
                <Alert variant="destructive">
                  <AlertTitle>AI回答の生成に失敗しました</AlertTitle>
                  <AlertDescription className="flex flex-col gap-3">
                    <span>
                      FastAPIからの応答取得に失敗しました。再生成を実行してください。
                    </span>
                    <Button
                      variant="destructive"
                      size="sm"
                      disabled={isRetrying}
                      onClick={() => retryGeneration(selectedInquiry.id)}
                      className="self-start"
                    >
                      {isRetrying ? "再生成中..." : "AI回答を再生成"}
                    </Button>
                  </AlertDescription>
                </Alert>
              )}

              {selectedInquiry.status !== "processing" &&
                selectedInquiry.status !== "error" && (
                  <QualityCheckCard inquiry={selectedInquiry} />
                )}

              <TopicCorrectionCard
                inquiry={selectedInquiry}
                isUpdatingTopic={isUpdatingTopic}
                onTopicChange={handleTopicChange}
              />

              {selectedInquiry.status !== "processing" &&
                selectedInquiry.status !== "error" &&
                selectedInquiry.generated_draft && (
                  <ResponseCard
                    inquiry={selectedInquiry}
                    editSubject={editSubject}
                    setEditSubject={setEditSubject}
                    editBody={editBody}
                    setEditBody={setEditBody}
                    isSaving={isSaving}
                    isSending={isSending}
                    saveMessage={saveMessage}
                    onSaveDraft={handleSaveDraft}
                    onSend={handleSend}
                  />
                )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-gray-500">
              左の一覧からお問い合わせを選択してください
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
