"use client";

import { useState, useEffect, useCallback } from "react";
import type { Inquiry, InquiryListItem } from "@/lib/db";

export function useInquiries() {
  const [inquiries, setInquiries] = useState<InquiryListItem[]>([]);
  const [selectedInquiry, setSelectedInquiry] = useState<Inquiry | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [topicFilter, setTopicFilter] = useState<string>("all");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchInquiries = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (statusFilter !== "all") {
        params.set("status", statusFilter);
      }
      if (topicFilter !== "all") {
        params.set("topic", topicFilter);
      }
      params.set("limit", "50");

      const response = await fetch(`/api/admin/inquiries?${params}`);
      if (!response.ok) throw new Error("Failed to fetch inquiries");

      const data = await response.json();
      setInquiries(data.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "エラーが発生しました");
    } finally {
      setIsLoading(false);
    }
  }, [statusFilter, topicFilter]);

  const fetchInquiryDetail = useCallback(async (id: string) => {
    try {
      const response = await fetch(`/api/admin/inquiries/${id}`);
      if (!response.ok) throw new Error("Failed to fetch inquiry details");

      const data: Inquiry = await response.json();
      setSelectedInquiry(data);
    } catch (err) {
      console.error("Error fetching inquiry details:", err);
    }
  }, []);

  const [isRetrying, setIsRetrying] = useState(false);

  const retryGeneration = useCallback(
    async (id: string) => {
      setIsRetrying(true);
      try {
        const response = await fetch(
          `/api/admin/inquiries/${id}/retry`,
          { method: "POST" }
        );
        if (!response.ok) throw new Error("Failed to retry generation");
        await fetchInquiryDetail(id);
        await fetchInquiries();
      } catch (err) {
        console.error("Error retrying generation:", err);
      } finally {
        setIsRetrying(false);
      }
    },
    [fetchInquiryDetail, fetchInquiries]
  );

  useEffect(() => {
    fetchInquiries();
  }, [fetchInquiries]);

  // 5秒ごとに自動更新
  useEffect(() => {
    const interval = setInterval(fetchInquiries, 5000);
    return () => clearInterval(interval);
  }, [fetchInquiries]);

  return {
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
  };
}
