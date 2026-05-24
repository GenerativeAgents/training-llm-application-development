"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { HealthCheckModal } from "@/components/health-check-modal";

export default function Home() {
  const [formData, setFormData] = useState({
    customer_name: "",
    customer_email: "",
    company_name: "",
    content: "",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);

    try {
      const response = await fetch("/api/inquiries", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || "送信に失敗しました");
      }

      setIsSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "エラーが発生しました");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <HealthCheckModal />

      {/* ヒーローセクション */}
      <section className="bg-white border-b">
        <div className="mx-auto max-w-5xl px-4 py-20 text-center">
          <h1 className="text-4xl font-bold tracking-tight">
            株式会社サンプルエージェント
          </h1>
          <p className="mt-4 text-lg text-gray-600">
            AIの力で、ビジネスの可能性を広げる
          </p>
        </div>
      </section>

      {/* 事業概要 */}
      <section className="mx-auto max-w-5xl px-4 py-16">
        <h2 className="text-2xl font-bold text-center mb-8">事業内容</h2>
        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>AIエージェント開発支援</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                お客様の業務課題に合わせたAIエージェントの設計・開発を支援します。業務効率化から顧客対応まで、幅広い領域でAI活用をサポートします。
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>AgentBoard</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">
                法人向けChatGPT型AIプラットフォーム。社内ナレッジとの連携やチーム利用に最適化された、セキュアなAI環境を提供します。
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* お問い合わせフォーム */}
      <section className="bg-white border-t">
        <div className="mx-auto max-w-2xl px-4 py-16">
          <h2 className="text-2xl font-bold text-center mb-4">お問い合わせ</h2>
          <p className="text-gray-600 text-center mb-8">
            サービスに関するご質問・ご相談はお気軽にどうぞ
          </p>

          {isSubmitted ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-green-600">送信完了</CardTitle>
                <CardDescription>
                  お問い合わせいただきありがとうございます
                </CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-gray-600 mb-4">
                  お問い合わせを受け付けました。担当者より順次ご連絡いたします。
                </p>
                <Button
                  onClick={() => {
                    setIsSubmitted(false);
                    setFormData({
                      customer_name: "",
                      customer_email: "",
                      company_name: "",
                      content: "",
                    });
                  }}
                  variant="outline"
                >
                  新しいお問い合わせ
                </Button>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>お問い合わせフォーム</CardTitle>
                <CardDescription>
                  株式会社サンプルエージェントへのお問い合わせはこちらから
                </CardDescription>
              </CardHeader>
              <CardContent>
                {error && (
                  <Alert variant="destructive" className="mb-6">
                    <AlertTitle>エラー</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div className="space-y-2">
                      <Label htmlFor="customer_name">
                        お名前 <span className="text-red-500">*</span>
                      </Label>
                      <Input
                        id="customer_name"
                        name="customer_name"
                        value={formData.customer_name}
                        onChange={handleChange}
                        required
                        placeholder="山田 太郎"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="customer_email">
                        メールアドレス <span className="text-red-500">*</span>
                      </Label>
                      <Input
                        id="customer_email"
                        name="customer_email"
                        type="email"
                        value={formData.customer_email}
                        onChange={handleChange}
                        required
                        placeholder="example@example.com"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="company_name">会社名</Label>
                    <Input
                      id="company_name"
                      name="company_name"
                      value={formData.company_name}
                      onChange={handleChange}
                      placeholder="株式会社サンプル"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="content">
                      お問い合わせ内容 <span className="text-red-500">*</span>
                    </Label>
                    <Textarea
                      id="content"
                      name="content"
                      value={formData.content}
                      onChange={handleChange}
                      required
                      rows={6}
                      placeholder="お問い合わせ内容をご記入ください"
                    />
                  </div>

                  <Button type="submit" disabled={isSubmitting} className="w-full">
                    {isSubmitting ? "送信中..." : "送信する"}
                  </Button>
                </form>
              </CardContent>
            </Card>
          )}
        </div>
      </section>
    </div>
  );
}
