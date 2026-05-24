"use client";

import { useState, useEffect } from "react";
import type { Inquiry, InquiryTopic } from "@/lib/db";

interface UseDraftEditorParams {
  selectedInquiry: Inquiry | null;
  setSelectedInquiry: (inquiry: Inquiry | null) => void;
  fetchInquiries: () => Promise<void>;
  fetchInquiryDetail: (id: string) => Promise<void>;
}

export function useDraftEditor({
  selectedInquiry,
  setSelectedInquiry,
  fetchInquiries,
  fetchInquiryDetail,
}: UseDraftEditorParams) {
  const [editSubject, setEditSubject] = useState("");
  const [editBody, setEditBody] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isUpdatingTopic, setIsUpdatingTopic] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);

  // selectedInquiry が変わったら編集フィールドを初期化
  useEffect(
    () => {
      if (!selectedInquiry) {
        setEditSubject("");
        setEditBody("");
        setSaveMessage(null);
        return;
      }
      if (selectedInquiry.final_response) {
        setEditSubject(selectedInquiry.final_response.subject);
        setEditBody(selectedInquiry.final_response.body);
      } else if (selectedInquiry.generated_draft) {
        setEditSubject(selectedInquiry.generated_draft.subject);
        setEditBody(selectedInquiry.generated_draft.body);
      } else {
        setEditSubject("");
        setEditBody("");
      }
      setSaveMessage(null);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps -- id と updated_at の変化のみで発火させる
    [selectedInquiry?.id, selectedInquiry?.updated_at],
  );

  async function submitDraft(
    endpoint: "draft" | "send",
    inquiryId: string,
    subject: string,
    body: string,
  ): Promise<void> {
    const response = await fetch(
      `/api/admin/inquiries/${inquiryId}/${endpoint}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ subject, body }),
      },
    );
    if (!response.ok) throw new Error(`Failed to ${endpoint}`);
  }

  const handleSaveDraft = async () => {
    if (!selectedInquiry) return;
    setIsSaving(true);
    setSaveMessage(null);
    try {
      await submitDraft("draft", selectedInquiry.id, editSubject, editBody);
      setSaveMessage({ type: "success", text: "下書きを保存しました" });
      fetchInquiries();
      fetchInquiryDetail(selectedInquiry.id);
    } catch (err) {
      setSaveMessage({
        type: "error",
        text: err instanceof Error ? err.message : "保存に失敗しました",
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleSend = async () => {
    if (!selectedInquiry) return;
    if (
      !confirm(
        "この内容でメールを送信します。よろしいですか？\n\n※実際のメール送信は行われません",
      )
    ) {
      return;
    }
    setIsSending(true);
    setSaveMessage(null);
    try {
      await submitDraft("send", selectedInquiry.id, editSubject, editBody);
      setSaveMessage({ type: "success", text: "送信しました" });
      fetchInquiries();
      fetchInquiryDetail(selectedInquiry.id);
    } catch (err) {
      setSaveMessage({
        type: "error",
        text: err instanceof Error ? err.message : "送信に失敗しました",
      });
    } finally {
      setIsSending(false);
    }
  };

  const handleTopicChange = async (newTopic: string) => {
    if (!selectedInquiry) return;
    setIsUpdatingTopic(true);
    // 楽観的更新
    setSelectedInquiry({
      ...selectedInquiry,
      topic: newTopic as InquiryTopic,
    });
    try {
      const response = await fetch(
        `/api/admin/inquiries/${selectedInquiry.id}/topic`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ topic: newTopic }),
        },
      );
      if (!response.ok) throw new Error("Failed to update topic");
      fetchInquiries();
    } catch (err) {
      // 失敗時にロールバック
      setSelectedInquiry(selectedInquiry);
      console.error("Error updating topic:", err);
    } finally {
      setIsUpdatingTopic(false);
    }
  };

  return {
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
  };
}
