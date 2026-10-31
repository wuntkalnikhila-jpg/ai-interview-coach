require("dotenv").config();
const express = require("express");
const cors = require("cors");
const Groq = require("groq-sdk");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 3002;

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

function tryParse(s) {
  try { return JSON.parse(s); } catch (e) { return undefined; }
}

function toArray(v) {
  if (Array.isArray(v)) return v;
  if (v && typeof v === "object") {
    for (const k of Object.keys(v)) {
      if (Array.isArray(v[k])) return v[k];
    }
  }
  return null;
}

// Turn messy AI output into a clean JSON array string (or null)
function cleanArrayReply(text) {
  let t = String(text || "").replace(/<think>[\s\S]*?<\/think>/gi, "");
  t = t.replace(/```json/gi, "").replace(/```/g, "");
  t = t.replace(/[\u0000-\u001F]+/g, " ").trim();
  let arr = toArray(tryParse(t));
  if (!arr) {
    const s = t.indexOf("["), e = t.lastIndexOf("]");
    if (s >= 0 && e > s) arr = toArray(tryParse(t.slice(s, e + 1)));
  }
  if (!arr) {
    // reply was cut off: keep only the complete items
    const s = t.indexOf("["), last = t.lastIndexOf("}");
    if (s >= 0 && last > s) arr = toArray(tryParse(t.slice(s, last + 1) + "]"));
  }
  return arr && arr.length ? JSON.stringify(arr) : null;
}

// The page shows AI text as HTML, so tags like <video> or <!DOCTYPE> would break it.
// Turn < > & into safe text before sending.
function escapeStrings(v) {
  if (typeof v === "string") {
    return v.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  if (Array.isArray(v)) return v.map(escapeStrings);
  if (v && typeof v === "object") {
    const out = {};
    for (const k of Object.keys(v)) out[k] = escapeStrings(v[k]);
    return out;
  }
  return v;
}

// For single-object replies (like answer feedback): parse, escape, send back as JSON
function safeObjectReply(text) {
  let t = String(text || "").replace(/<think>[\s\S]*?<\/think>/gi, "");
  t = t.replace(/```json/gi, "").replace(/```/g, "").trim();
  let obj = tryParse(t);
  if (obj === undefined) {
    const s = t.indexOf("{"), e = t.lastIndexOf("}");
    if (s >= 0 && e > s) obj = tryParse(t.slice(s, e + 1));
  }
  return obj && typeof obj === "object" ? JSON.stringify(escapeStrings(obj)) : text;
}

// Models to use, in order. Groq retired llama-3.3-70b-versatile (Aug 2026).
// To force one, add GROQ_MODEL=model-name in .env
const MODEL_LIST = [process.env.GROQ_MODEL, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
  .filter((m, i, a) => m && a.indexOf(m) === i);
let modelIndex = 0;

function modelIsGone(e) {
  const msg = String((e && e.message) || "");
  return !!e && (e.status === 404 || /model_not_found|model_decommissioned|does not exist|decommissioned/i.test(msg));
}

// One call to Groq.
// - model not available: switch to the next model
// - rate limit (free plan): wait and try again
async function groqChat(prompt, system, maxTokens) {
  let rateRetries = 0;
  while (true) {
    const model = MODEL_LIST[modelIndex];
    try {
      const body = {
        model: model,
        max_completion_tokens: maxTokens,
        messages: [
          { role: "system", content: system || "You are an expert interview coach." },
          { role: "user", content: prompt }
        ]
      };
      if (model.startsWith("openai/gpt-oss")) body.reasoning_effort = "low";
      const res = await groq.chat.completions.create(body);
      return res.choices[0].message.content || "";
    } catch (e) {
      if (modelIsGone(e) && modelIndex < MODEL_LIST.length - 1) {
        modelIndex++;
        console.log("Model " + model + " is not available. Switching to " + MODEL_LIST[modelIndex]);
        continue;
      }
      const limited = e && (e.status === 429 || /rate limit/i.test(e.message || ""));
      if (!limited || rateRetries >= 4) throw e;
      rateRetries++;
      const m = /try again in ([\d.]+)s/i.exec(e.message || "");
      const wait = Math.min(30, (m ? parseFloat(m[1]) : 10) + 1);
      console.log("Rate limit reached, waiting " + Math.round(wait) + " seconds...");
      await new Promise((r) => setTimeout(r, wait * 1000));
    }
  }
}

// Many questions: the AI gives too few in one go, so ask in small batches and join them
async function askMany(prompt, system, total) {
  const all = [];
  const seen = new Set();
  const maxCalls = Math.ceil(total / 8) + 2;
  for (let call = 0; call < maxCalls && all.length < total; call++) {
    const n = Math.min(10, total - all.length);
    let p = prompt.replace(/^Generate \d+ /, "Generate " + n + " ");
    if (all.length) {
      p += "\nThese were already asked. Do NOT repeat or rephrase them: " +
        all.map((q) => q.question).join(" | ");
    }
    let items = null;
    try {
      const cleaned = cleanArrayReply(await groqChat(p, system, 3500));
      items = cleaned ? JSON.parse(cleaned) : null;
    } catch (e) {
      if (!all.length) throw e;
      console.log("Stopped early:", e.message);
      break;
    }
    if (!items) continue;
    for (const q of items) {
      const key = String((q && q.question) || "").trim().toLowerCase();
      if (!key || seen.has(key)) continue;
      seen.add(key);
      all.push(q);
    }
    console.log("Questions so far: " + all.length + " / " + total);
  }
  if (!all.length) throw new Error("AI returned no readable questions. Please try again.");
  return JSON.stringify(escapeStrings(all.slice(0, total)));
}

// Ask the AI for fuller, clearer answers when the prompt has an idealAnswer field
function withAnswerStyle(prompt) {
  if (!/idealAnswer/.test(prompt)) return prompt;
  return prompt + "\nIn every item, write idealAnswer as a clear, complete model answer in simple language (about 3 sentences) that a fresher can learn from. Keep tip to one short line.";
}

async function ask(prompt, system) {
  prompt = withAnswerStyle(prompt);
  const wantsArray = /JSON array/i.test(prompt);
  const m = /^Generate (\d+) /.exec(prompt);
  if (wantsArray && m && parseInt(m[1], 10) > 10) {
    return askMany(prompt, system, parseInt(m[1], 10));
  }
  const text = await groqChat(prompt, system, 3500);
  if (wantsArray) {
    const cleaned = cleanArrayReply(text);
    if (cleaned) return JSON.stringify(escapeStrings(JSON.parse(cleaned)));
    console.log("Could not read AI reply. First 300 characters:\n" + text.slice(0, 300));
    return text;
  }
  return safeObjectReply(text);
}

// Style for the Questions page: show every question with its answer, clearly.
// Added here by the server, so index.html does not need to be edited.
const ANSWER_CSS = `<style>
.q-card{padding:18px;margin-bottom:14px}
.q-card-text{font-size:15px;line-height:1.6;margin-bottom:8px}
.q-card-tip{font-size:12px;line-height:1.6}
.q-card-ans{display:block !important;background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:12px 14px;margin-top:12px;font-size:13px;line-height:1.8}
</style>`;

// Added to the page by the server (so index.html does not need to be edited):
// when someone submits feedback, send it to the server first (which notifies your phone).
const FEEDBACK_SCRIPT = `<script>
(function () {
  var original = window.submitFeedback;
  if (typeof original !== "function") return;
  if (String(original).indexOf("/api/feedback") >= 0) return;
  window.submitFeedback = async function () {
    var name = (document.getElementById("fb-name").value || "").trim();
    var rating = typeof _selectedRating !== "undefined" ? _selectedRating : "";
    if (!rating || !name) { return original(); }
    var data = {
      name: name,
      rating: rating,
      feature: document.getElementById("fb-feature").value,
      recommend: typeof _selectedRecommend !== "undefined" ? _selectedRecommend : "",
      comments: document.getElementById("fb-comments").value.trim()
    };
    try {
      var res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      if (!res.ok) throw new Error("send failed");
    } catch (e) {
      alert("Could not send feedback. Please check your connection and try again.");
      return;
    }
    original();
  };
})();
</script>`;

app.get("/", (req, res) => {
  fs.readFile(path.join(__dirname, "index.html"), "utf8", (err, html) => {
    if (err) return res.status(404).send("index.html not found next to server.js");
    let page = html.includes("</head>")
      ? html.replace("</head>", () => ANSWER_CSS + "</head>")
      : ANSWER_CSS + html;
    const end = page.lastIndexOf("</body>");
    page = end >= 0
      ? page.slice(0, end) + FEEDBACK_SCRIPT + page.slice(end)
      : page + FEEDBACK_SCRIPT;
    res.set("Cache-Control", "no-store");
    res.type("html").send(page);
  });
});

// Feedback: save a copy in feedbacks.json and send it to your phone
app.post("/api/feedback", async (req, res) => {
  try {
    const b = req.body || {};
    const clip = (v, n) => String(v || "").slice(0, n);
    const fb = {
      name: clip(b.name, 100),
      rating: clip(b.rating, 20),
      feature: clip(b.feature, 60),
      recommend: clip(b.recommend, 10),
      comments: clip(b.comments, 1500),
      date: new Date().toISOString()
    };
    if (!fb.name || !fb.rating) return res.status(400).json({ error: "name and rating required" });

    try {
      const file = path.join(__dirname, "feedbacks.json");
      let list = [];
      try { list = JSON.parse(fs.readFileSync(file, "utf8")); } catch (e) {}
      list.push(fb);
      fs.writeFileSync(file, JSON.stringify(list, null, 2));
    } catch (e) { console.error("could not save feedbacks.json:", e.message); }

    // Phone alert text
    const text =
      "Name: " + fb.name + "\n" +
      "Rating: " + fb.rating + "\n" +
      "Liked most: " + (fb.feature || "-") + "\n" +
      "Recommend: " + (fb.recommend || "-") + "\n" +
      "Comments: " + (fb.comments || "(none)").slice(0, 800);

    let sent = false;
    const ntfyTopic = process.env.NTFY_TOPIC;
    const token = process.env.TELEGRAM_BOT_TOKEN;
    const chatId = process.env.TELEGRAM_CHAT_ID;

    // Option 1: ntfy app (free, no account)
    if (ntfyTopic) {
      try {
        const r = await fetch("https://ntfy.sh/" + encodeURIComponent(ntfyTopic), {
          method: "POST",
          headers: { "Title": "New feedback - AI Interview Coach", "Priority": "high", "Tags": "speech_balloon" },
          body: text
        });
        if (r.ok) sent = true; else console.error("ntfy error:", await r.text());
      } catch (e) {
        console.error("ntfy send failed:", e.message);
      }
    }

    // Option 2: Telegram bot
    if (token && chatId) {
      try {
        const r = await fetch("https://api.telegram.org/bot" + token + "/sendMessage", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ chat_id: chatId, text: "New feedback - AI Interview Coach\n\n" + text })
        });
        if (r.ok) sent = true; else console.error("Telegram error:", await r.text());
      } catch (e) {
        console.error("Telegram send failed:", e.message);
      }
    }

    if (!ntfyTopic && !(token && chatId)) {
      console.log("Phone alerts are not set up in .env - feedback saved to feedbacks.json only");
    }
    res.json({ ok: true, sent: sent });
  } catch (e) {
    console.error("feedback error:", e.message);
    res.status(500).json({ error: "could not save feedback" });
  }
});

app.post("/api/chat", async (req, res) => {
  try {
    const { prompt, system } = req.body;
    if (!prompt) return res.status(400).json({ error: "prompt required" });
    const reply = await ask(prompt, system);
    res.json({ reply });
  } catch (e) {
    console.error("chat error:", e.message);
    res.status(500).json({ error: e.message });
  }
});

app.use((req, res) => {
  res.status(404).json({ error: "Route not found: " + req.path });
});

app.listen(PORT, () => {
  const page = path.join(__dirname, "index.html");
  console.log("\nAI Interview Coach running at http://localhost:" + PORT + "\n");
  console.log("Server file : " + __filename);
  console.log("Page file   : " + page + (fs.existsSync(page) ? "" : "   <-- NOT FOUND"));
  console.log("Model       : " + MODEL_LIST[0] + " (switches automatically if unavailable)");
  const alerts = [];
  if (process.env.NTFY_TOPIC) alerts.push("ntfy");
  if (process.env.TELEGRAM_BOT_TOKEN && process.env.TELEGRAM_CHAT_ID) alerts.push("Telegram");
  console.log("Phone alerts: " + (alerts.length ? "ON (" + alerts.join(", ") + ")" : "OFF - add NTFY_TOPIC to .env"));
  console.log("Version     : phone alerts + new Groq model + answers + batching + safe text\n");
  if (!process.env.GROQ_API_KEY) console.log("WARNING: GROQ_API_KEY is missing in .env\n");
});