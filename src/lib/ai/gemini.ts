/**
 * BandQuest IELTS — Gemini AI provider abstraction (SERVER-SIDE ONLY)
 * All privileged Gemini requests run here. The API key never reaches the browser.
 *
 * NOTE (2026-10-18): Uses @google/generative-ai (not deprecated, verified working
 * end-to-end). Google's newer @google/genai SDK exists; migration is optional
 * tech debt — only migrate if the old SDK breaks or a feature needs the new one.
 */
import { GoogleGenerativeAI } from "@google/generative-ai";
import { getDb } from "../db";

export const AI_DISCLAIMER =
  "This is an estimated practice result, not an official IELTS Test Report Form or a guaranteed prediction of your actual exam score.";

export const RUBRIC_VERSION = "ielts-public-v1";

async function getConfig(key: string, fallback: string): Promise<string> {
  try {
    const db = getDb();
    const row = await db.prepare("SELECT value FROM system_config WHERE key = ?").get(key) as
      | { value: string }
      | undefined;
    return row?.value ?? fallback;
  } catch {
    return fallback;
  }
}

export async function aiEnabled(): Promise<boolean> {
  // Env key is the real gate; DB flag is an admin kill-switch (defaults on).
  return Boolean(process.env.GEMINI_API_KEY) && (await getConfig("ai.enabled", "1")) === "1";
}

export async function textModel(): Promise<string> {
  return process.env.GEMINI_TEXT_MODEL || getConfig("ai.text_model", "gemini-3.8-flash");
}

let client: GoogleGenerativeAI | null = null;
function getClient(): GoogleGenerativeAI {
  if (!client) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) throw new Error("AI_NOT_CONFIGURED");
    client = new GoogleGenerativeAI(key);
  }
  return client;
}

/** Per-user daily AI rate limit. Returns true if allowed. */
export async function checkRateLimit(userId: string, feature: string): Promise<boolean> {
  const db = getDb();
  const limit = parseInt(await getConfig("ai.daily_limit_per_user", "50"), 10);
  const row = await db
    .prepare(
      `SELECT COUNT(*) as c FROM ai_usage
       WHERE user_id = ? AND feature = ? AND date(created_at) = date('now')`
    )
    .get(userId, feature) as { c: number };
  return row.c < limit;
}

export async function logUsage(
  userId: string,
  feature: string,
  tokensIn = 0,
  tokensOut = 0
): Promise<void> {
  try {
    const db = getDb();
    await db.prepare(
      `INSERT INTO ai_usage (id, user_id, feature, tokens_in, tokens_out)
       VALUES (lower(hex(randomblob(16))), ?, ?, ?, ?)`
    ).run(userId, feature, tokensIn, tokensOut);
  } catch {
    // usage logging must never break the request
  }
}

export interface AiCallOptions {
  systemPrompt: string;
  userPrompt: string;
  audioBase64?: string;
  audioMimeType?: string;
  jsonMode?: boolean;
  maxTokens?: number;
}

export interface AiCallResult {
  text: string;
  tokensIn: number;
  tokensOut: number;
  model: string;
}

/** Single entry point for all Gemini text/audio calls. Throws AI_NOT_CONFIGURED / AI_QUOTA / AI_ERROR. */
export async function callGemini(opts: AiCallOptions): Promise<AiCallResult> {
  const modelName = await textModel();
  let model;
  try {
    model = getClient().getGenerativeModel({
      model: modelName,
      systemInstruction: opts.systemPrompt,
      generationConfig: {
        maxOutputTokens: opts.maxTokens ?? 2048,
        temperature: 0.4,
        ...(opts.jsonMode ? { responseMimeType: "application/json" } : {}),
      },
    });
  } catch (e) {
    if (e instanceof Error && e.message === "AI_NOT_CONFIGURED") throw e;
    throw new Error("AI_ERROR");
  }

  try {
    // Netlify free functions die if the upstream hangs — fail fast so the
    // route can fall back to rule-based feedback instead of timing out.
    const withTimeout = <T>(p: Promise<T>, ms = 25000): Promise<T> =>
      Promise.race([
        p,
        new Promise<T>((_, reject) =>
          setTimeout(() => reject(new Error("AI_TIMEOUT")), ms)
        ),
      ]);
    let result;
    if (opts.audioBase64) {
      result = await withTimeout(
        model.generateContent([
          { text: opts.userPrompt },
          { inlineData: { data: opts.audioBase64, mimeType: opts.audioMimeType || "audio/webm" } },
        ])
      );
    } else {
      result = await withTimeout(model.generateContent(opts.userPrompt));
    }
    const text = result.response.text();
    const usage = result.response.usageMetadata;
    return {
      text,
      tokensIn: usage?.promptTokenCount ?? 0,
      tokensOut: usage?.candidatesTokenCount ?? 0,
      model: modelName,
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : "";
    if (/quota|429|resource.?exhausted/i.test(msg)) throw new Error("AI_QUOTA");
    throw new Error("AI_ERROR");
  }
}

/** Parse + validate a JSON AI response against a simple shape check. */
export function parseJsonResponse<T>(text: string, requiredKeys: (keyof T)[]): T {
  const cleaned = text.replace(/^```json\s*/i, "").replace(/```\s*$/, "").trim();
  let obj: unknown;
  try {
    obj = JSON.parse(cleaned);
  } catch {
    throw new Error("AI_MALFORMED");
  }
  if (typeof obj !== "object" || obj === null) throw new Error("AI_MALFORMED");
  for (const k of requiredKeys) {
    if (!(k in (obj as Record<string, unknown>))) throw new Error("AI_MALFORMED");
  }
  return obj as T;
}

export function aiErrorMessage(code: string): string {
  switch (code) {
    case "AI_NOT_CONFIGURED":
      return "AI feedback is not configured yet. Your work has been saved — you can continue with all non-AI lessons, and try AI feedback again once the administrator enables it.";
    case "AI_QUOTA":
      return "Our AI service is temporarily at capacity. Your work has been saved — please try again in a little while, or continue with non-AI practice.";
    case "AI_RATE_LIMIT":
      return "You've reached today's AI usage limit. Your work is saved — the limit resets tomorrow, and all non-AI lessons remain available.";
    default:
      return "AI analysis hit a temporary problem. Your work has been saved — please try again shortly.";
  }
}
