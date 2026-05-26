import Database from "better-sqlite3";
import path from "path";
import fs from "fs";
import { randomUUID } from "crypto";
import yaml from "js-yaml";
import { initSchema } from "../src/lib/db";

// ---------- DB setup ----------
// スキーマ定義は web/src/lib/db.ts の initSchema を信頼ソースとして再利用する

const dbPath = path.join(__dirname, "..", "data", "inquiries.db");
const db = new Database(dbPath);
db.pragma("journal_mode = WAL");

initSchema(db);

// ---------- Clear existing data ----------

db.exec("DELETE FROM inquiries");
console.log("Cleared existing data.");

// ---------- Load YAML ----------

const now = new Date();

function ts(minutesAgo: number): string {
  return new Date(now.getTime() - minutesAgo * 60_000).toISOString();
}

function tsOrNull(minutesAgo: number | null | undefined): string | null {
  return minutesAgo == null ? null : ts(minutesAgo);
}

interface ResponseEntry {
  subject: string;
  body: string;
  quality_scores?: {
    politeness: string;
    politeness_reason: string;
  };
}

interface SeedEntry {
  customer_name: string;
  customer_email: string;
  company_name: string | null;
  content: string;
  status: "processing" | "draft" | "sent";
  topic: string | null;
  original_topic: string | null;
  operator_edited_topic: boolean;
  classification_confidence: number | null;
  quality_alert: boolean;
  edit_distance: number | null;
  minutes_ago: number;
  updated_minutes_ago: number;
  sent_minutes_ago: number | null;
  ai_response: ResponseEntry | null;
  final_response: ResponseEntry | null;
}

interface SeedRow {
  id: string;
  customer_name: string;
  customer_email: string;
  company_name: string | null;
  content: string;
  status: "processing" | "draft" | "sent";
  topic: string | null;
  original_topic: string | null;
  operator_edited_topic: number;
  ai_response: string | null;
  final_response: string | null;
  classification_confidence: number | null;
  quality_alert: number;
  edit_distance: number | null;
  created_at: string;
  updated_at: string;
  sent_at: string | null;
}

const yamlPath = path.join(__dirname, "seed-data.yaml");
const entries = yaml.load(fs.readFileSync(yamlPath, "utf8")) as SeedEntry[];

function toRow(entry: SeedEntry): SeedRow {
  return {
    id: randomUUID(),
    customer_name: entry.customer_name,
    customer_email: entry.customer_email,
    company_name: entry.company_name,
    content: entry.content,
    status: entry.status,
    topic: entry.topic,
    original_topic: entry.original_topic,
    operator_edited_topic: entry.operator_edited_topic ? 1 : 0,
    ai_response: entry.ai_response ? JSON.stringify(entry.ai_response) : null,
    final_response: entry.final_response ? JSON.stringify(entry.final_response) : null,
    classification_confidence: entry.classification_confidence,
    quality_alert: entry.quality_alert ? 1 : 0,
    edit_distance: entry.edit_distance,
    created_at: ts(entry.minutes_ago),
    updated_at: ts(entry.updated_minutes_ago),
    sent_at: tsOrNull(entry.sent_minutes_ago),
  };
}

const rows = entries.map(toRow);

// ---------- Insert ----------

const insert = db.prepare(`
  INSERT INTO inquiries (
    id, customer_name, customer_email, company_name,
    content, status, topic, original_topic,
    operator_edited_topic, ai_response, final_response,
    classification_confidence, quality_alert, edit_distance,
    created_at, updated_at, sent_at
  ) VALUES (
    @id, @customer_name, @customer_email, @company_name,
    @content, @status, @topic, @original_topic,
    @operator_edited_topic, @ai_response, @final_response,
    @classification_confidence, @quality_alert, @edit_distance,
    @created_at, @updated_at, @sent_at
  )
`);

const insertAll = db.transaction((items: SeedRow[]) => {
  for (const item of items) {
    insert.run(item);
  }
});

insertAll(rows);

console.log(`Seeded ${rows.length} inquiries.`);

// サマリーを表示
for (const r of rows) {
  console.log(`  - ${r.customer_name} [${r.status}] ${r.topic ?? "(none)"} ${r.quality_alert ? "⚠ quality_alert" : ""}`);
}

db.close();
