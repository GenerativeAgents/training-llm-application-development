import Database from "better-sqlite3";
import path from "path";

// 型定義
export type InquiryStatus = "processing" | "draft" | "sent" | "error";
export type InquiryTopic = "development" | "product" | "other" | "spam";

export interface QualityScores {
  politeness: "OK" | "NG";
  politeness_reason: string;
}

export interface GeneratedDraft {
  subject: string;
  body: string;
  quality_scores: QualityScores;
}

export interface FinalResponse {
  subject: string;
  body: string;
}

export interface Inquiry {
  id: string;
  customer_name: string;
  customer_email: string;
  company_name: string | null;
  content: string;
  status: InquiryStatus;
  topic: InquiryTopic | null;
  original_topic: InquiryTopic | null;
  operator_edited_topic: boolean | null;
  generated_draft: GeneratedDraft | null;
  final_response: FinalResponse | null;
  classification_confidence: number | null;
  quality_alert: boolean;
  edit_distance: number | null;
  weave_call_id: string | null;
  created_at: string;
  updated_at: string;
  sent_at: string | null;
}

export interface InquiryListItem {
  id: string;
  customer_name: string;
  company_name: string | null;
  status: InquiryStatus;
  topic: InquiryTopic | null;
  quality_alert: boolean;
  created_at: string;
}

// データベースシングルトン
let db: Database.Database | null = null;

export function getDb(): Database.Database {
  if (!db) {
    const dbPath = path.join(process.cwd(), "data", "inquiries.db");
    db = new Database(dbPath);
    db.pragma("journal_mode = WAL");
    initSchema(db);
  }
  return db;
}

function initSchema(database: Database.Database) {
  database.exec(`
    CREATE TABLE IF NOT EXISTS inquiries (
      id TEXT PRIMARY KEY,
      customer_name TEXT NOT NULL,
      customer_email TEXT NOT NULL,
      company_name TEXT,
      content TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'processing',
      topic TEXT,
      original_topic TEXT,
      operator_edited_topic INTEGER,
      ai_response TEXT,
      final_response TEXT,
      classification_confidence REAL,
      quality_alert INTEGER NOT NULL DEFAULT 0,
      edit_distance REAL,
      weave_call_id TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      sent_at TEXT
    )
  `);

  database.exec(`
    CREATE INDEX IF NOT EXISTS idx_inquiries_status ON inquiries(status);
    CREATE INDEX IF NOT EXISTS idx_inquiries_created_at ON inquiries(created_at DESC);
  `);
}

// CRUD 操作
export function createInquiry(data: {
  id: string;
  customer_name: string;
  customer_email: string;
  company_name?: string;
  content: string;
}): Inquiry {
  const db = getDb();
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    INSERT INTO inquiries (id, customer_name, customer_email, company_name, content, status, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, 'processing', ?, ?)
  `);

  stmt.run(
    data.id,
    data.customer_name,
    data.customer_email,
    data.company_name || null,
    data.content,
    now,
    now
  );

  return getInquiryById(data.id)!;
}

export function getInquiryById(id: string): Inquiry | null {
  const db = getDb();
  const stmt = db.prepare("SELECT * FROM inquiries WHERE id = ?");
  const row = stmt.get(id) as Record<string, unknown> | undefined;

  if (!row) return null;

  return parseInquiryRow(row);
}

export function listInquiries(options: {
  status?: InquiryStatus;
  topic?: InquiryTopic;
  limit?: number;
  offset?: number;
}): { items: InquiryListItem[]; total: number } {
  const db = getDb();
  const { status, topic, limit = 20, offset = 0 } = options;

  const conditions: string[] = [];
  const params: unknown[] = [];

  if (status) {
    conditions.push("status = ?");
    params.push(status);
  }
  if (topic) {
    conditions.push("topic = ?");
    params.push(topic);
  }

  const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

  const countStmt = db.prepare(
    `SELECT COUNT(*) as count FROM inquiries ${whereClause}`
  );
  const countResult = countStmt.get(...params) as { count: number };

  const listStmt = db.prepare(`
    SELECT id, customer_name, company_name, status, topic, quality_alert, created_at
    FROM inquiries
    ${whereClause}
    ORDER BY created_at DESC
    LIMIT ? OFFSET ?
  `);

  const rows = listStmt.all(...params, limit, offset) as Record<
    string,
    unknown
  >[];

  return {
    items: rows.map((row) => ({
      id: row.id as string,
      customer_name: row.customer_name as string,
      company_name: row.company_name as string | null,
      status: row.status as InquiryStatus,
      topic: row.topic as InquiryTopic | null,
      quality_alert: Boolean(row.quality_alert),
      created_at: row.created_at as string,
    })),
    total: countResult.count,
  };
}

export function updateInquiryWithGeneratedDraft(
  id: string,
  data: {
    topic: InquiryTopic;
    classification_confidence: number;
    generated_draft: GeneratedDraft | null;
    quality_alert: boolean;
    weave_call_id?: string | null;
  }
): Inquiry | null {
  const db = getDb();
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    UPDATE inquiries
    SET topic = ?, original_topic = ?, classification_confidence = ?, ai_response = ?, quality_alert = ?, weave_call_id = ?, status = 'draft', updated_at = ?
    WHERE id = ?
  `);

  stmt.run(
    data.topic,
    data.topic,
    data.classification_confidence,
    data.generated_draft ? JSON.stringify(data.generated_draft) : null,
    data.quality_alert ? 1 : 0,
    data.weave_call_id ?? null,
    now,
    id
  );

  return getInquiryById(id);
}

export function markInquiryAsError(id: string): Inquiry | null {
  const db = getDb();
  const now = new Date().toISOString();
  const stmt = db.prepare(
    `UPDATE inquiries SET status = 'error', updated_at = ? WHERE id = ?`
  );
  stmt.run(now, id);
  return getInquiryById(id);
}

export function markInquiryAsProcessing(id: string): Inquiry | null {
  const db = getDb();
  const now = new Date().toISOString();
  const stmt = db.prepare(
    `UPDATE inquiries SET status = 'processing', updated_at = ? WHERE id = ?`
  );
  stmt.run(now, id);
  return getInquiryById(id);
}

export function saveDraft(
  id: string,
  data: { subject: string; body: string }
): Inquiry | null {
  const db = getDb();
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    UPDATE inquiries
    SET final_response = ?, status = 'draft', updated_at = ?
    WHERE id = ?
  `);

  stmt.run(JSON.stringify(data), now, id);

  return getInquiryById(id);
}

export function updateInquiryTopic(id: string, topic: InquiryTopic): Inquiry | null {
  const db = getDb();
  const now = new Date().toISOString();
  const stmt = db.prepare(`UPDATE inquiries SET topic = ?, updated_at = ? WHERE id = ?`);
  stmt.run(topic, now, id);
  return getInquiryById(id);
}

export function markAsSent(
  id: string,
  data: {
    subject: string;
    body: string;
    edit_distance: number | null;
    operator_edited_topic: boolean | null;
  }
): Inquiry | null {
  const db = getDb();
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    UPDATE inquiries
    SET final_response = ?, status = 'sent', updated_at = ?, sent_at = ?, edit_distance = ?, operator_edited_topic = ?
    WHERE id = ?
  `);

  stmt.run(
    JSON.stringify({ subject: data.subject, body: data.body }),
    now,
    now,
    data.edit_distance,
    data.operator_edited_topic != null ? (data.operator_edited_topic ? 1 : 0) : null,
    id
  );

  return getInquiryById(id);
}

function parseInquiryRow(row: Record<string, unknown>): Inquiry {
  return {
    id: row.id as string,
    customer_name: row.customer_name as string,
    customer_email: row.customer_email as string,
    company_name: row.company_name as string | null,
    content: row.content as string,
    status: row.status as InquiryStatus,
    topic: row.topic as InquiryTopic | null,
    original_topic: row.original_topic as InquiryTopic | null,
    operator_edited_topic:
      row.operator_edited_topic == null ? null : Boolean(row.operator_edited_topic),
    generated_draft: row.ai_response
      ? (JSON.parse(row.ai_response as string) as GeneratedDraft)
      : null,
    final_response: row.final_response
      ? (JSON.parse(row.final_response as string) as FinalResponse)
      : null,
    classification_confidence:
      row.classification_confidence != null
        ? (row.classification_confidence as number)
        : null,
    quality_alert: Boolean(row.quality_alert),
    edit_distance: row.edit_distance != null ? (row.edit_distance as number) : null,
    weave_call_id: (row.weave_call_id as string | null) ?? null,
    created_at: row.created_at as string,
    updated_at: row.updated_at as string,
    sent_at: row.sent_at as string | null,
  };
}
