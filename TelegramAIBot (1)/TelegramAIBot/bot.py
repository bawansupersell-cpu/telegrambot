"""
Telegram AI bot powered by the Claude API (no extra libraries needed).
Run:  python bot.py
"""
import base64
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")
DB_PATH = os.path.join(BASE, "bot_data.db")

DEFAULTS = {
    "telegram_token": "",                 # لە @BotFather وەریبگرە
    "anthropic_api_key": "",              # لە console.anthropic.com
    "model": "claude-sonnet-5-5",
    "allowed_user_ids": [],               # ID ی تلگرامی خۆت (بۆتەکە /id ی پێ دەڵێت)
    "max_tokens": 2000,
    "history_messages": 20,               # چەند نامەی پێشوو لەبیر بێت
    "daily_token_budget": 0,              # 0 = بێسنور؛ ژمارە = سنووری ڕۆژانە بۆ هەر کەس
    "system_prompt": (
        "تۆ یاریدەدەرێکی زیرەکی و دۆستانەی. بە هەمان زمانی بەکارهێنەر وەڵام بدەرەوە "
        "(بە زۆری کوردی سۆرانی، ئینگلیزی یان عەرەبی). ڕاستگۆ و ڕوون و یارمەتیدەر بە. "
        "ئەگەر شتێکت نەزانی، بڵێ نازانم."
    ),
}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8-sig") as f:
            cfg.update(json.load(f))
    else:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULTS, f, ensure_ascii=False, indent=2)
    return cfg


# ------------------------------------------------------------------ storage
class Store:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER,
                role TEXT, content TEXT);
            CREATE TABLE IF NOT EXISTS usage(
                user_id INTEGER, day TEXT, tokens INTEGER DEFAULT 0,
                PRIMARY KEY(user_id, day));
            """)
            self.conn.commit()

    def history(self, chat_id, limit):
        with self.lock:
            rows = self.conn.execute(
                "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
                (chat_id, limit)).fetchall()
        msgs = [{"role": r, "content": c} for r, c in reversed(rows)]
        while msgs and msgs[0]["role"] != "user":      # API needs to start with a user turn
            msgs.pop(0)
        return msgs

    def add(self, chat_id, role, content):
        with self.lock:
            self.conn.execute("INSERT INTO messages(chat_id,role,content) VALUES(?,?,?)",
                              (chat_id, role, content))
            self.conn.commit()

    def reset(self, chat_id):
        with self.lock:
            self.conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
            self.conn.commit()

    def add_usage(self, user_id, tokens):
        day = date.today().isoformat()
        with self.lock:
            self.conn.execute(
                "INSERT INTO usage(user_id,day,tokens) VALUES(?,?,?) "
                "ON CONFLICT(user_id,day) DO UPDATE SET tokens=tokens+excluded.tokens",
                (user_id, day, tokens))
            self.conn.commit()

    def usage_today(self, user_id):
        with self.lock:
            r = self.conn.execute("SELECT tokens FROM usage WHERE user_id=? AND day=?",
                                  (user_id, date.today().isoformat())).fetchone()
        return r[0] if r else 0


# --------------------------------------------------------------- HTTP clients
def _post_json(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Telegram:
    def __init__(self, token):
        self.api = f"https://api.telegram.org/bot{token}"
        self.file_api = f"https://api.telegram.org/file/bot{token}"

    def call(self, method, payload, timeout=60):
        return _post_json(f"{self.api}/{method}", payload, {}, timeout)

    def get_updates(self, offset):
        return self.call("getUpdates", {"offset": offset, "timeout": 30,
                                        "allowed_updates": ["message"]}, timeout=45).get("result", [])

    def send(self, chat_id, text):
        for chunk in split_text(text):
            self.call("sendMessage", {"chat_id": chat_id, "text": chunk})

    def typing(self, chat_id):
        try:
            self.call("sendChatAction", {"chat_id": chat_id, "action": "typing"}, timeout=10)
        except Exception:
            pass

    def download_photo(self, file_id):
        info = self.call("getFile", {"file_id": file_id})["result"]
        with urllib.request.urlopen(f"{self.file_api}/{info['file_path']}", timeout=60) as r:
            return r.read()


def split_text(text, limit=4000):
    text = text.strip() or "…"
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    parts.append(text)
    return parts


def ask_claude(cfg, messages):
    """Returns (reply_text, tokens_used)."""
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"model": cfg["model"], "max_tokens": int(cfg["max_tokens"]),
         "system": cfg["system_prompt"], "messages": messages},
        {"x-api-key": cfg["anthropic_api_key"], "anthropic-version": "2023-06-01"})
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage", {})
    return text, int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))


# ----------------------------------------------------------------------- bot
class Bot:
    def __init__(self, cfg, tg=None, claude=None):
        self.cfg = cfg
        self.tg = tg or Telegram(cfg["telegram_token"])
        self.claude = claude or ask_claude
        self.store = Store(DB_PATH)
        self.chat_locks = {}
        self.locks_guard = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=8)

    def _lock(self, chat_id):
        with self.locks_guard:
            return self.chat_locks.setdefault(chat_id, threading.Lock())

    def allowed(self, user_id):
        return user_id in [int(x) for x in self.cfg["allowed_user_ids"]]

    def handle(self, msg):
        chat_id = msg["chat"]["id"]
        user_id = msg.get("from", {}).get("id", 0)
        text = (msg.get("text") or msg.get("caption") or "").strip()

        if text.startswith("/id"):
            return self.tg.send(chat_id, f"ID ی تلگرامی تۆ: {user_id}\nئەمە بخە ناو allowed_user_ids لە config.json")
        if not self.allowed(user_id):
            return self.tg.send(chat_id, "ئەم بۆتە تایبەتە. /id بنووسە و ID ەکەت بدە بە خاوەنەکەی.")
        if text.startswith("/start") or text.startswith("/help"):
            return self.tg.send(chat_id, "سڵاو! هەر شتێک بنووسە یان وێنەیەک بنێرە.\n"
                                         "/new = گفتوگۆی نوێ\n/usage = بەکارهێنانی ئەمڕۆ\n/id = ID ی تۆ")
        if text.startswith("/new"):
            self.store.reset(chat_id)
            return self.tg.send(chat_id, "گفتوگۆی نوێ دەستی پێکرد ✅")
        if text.startswith("/usage"):
            return self.tg.send(chat_id, f"ئەمڕۆ {self.store.usage_today(user_id):,} تۆکن بەکارهێنراوە.")

        budget = int(self.cfg["daily_token_budget"])
        if budget and self.store.usage_today(user_id) >= budget:
            return self.tg.send(chat_id, "سنووری ڕۆژانە تەواو بوو. سبەینێ هەوڵ بدەرەوە.")

        content = self._build_content(msg, text)
        if content is None:
            return self.tg.send(chat_id, "تەنها نووسین و وێنە پشتگیری دەکرێن.")

        with self._lock(chat_id):
            self.tg.typing(chat_id)
            history = self.store.history(chat_id, int(self.cfg["history_messages"]))
            messages = history + [{"role": "user", "content": content}]
            try:
                reply, used = self.claude(self.cfg, messages)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")[:300]
                print(f"Claude API error {e.code}: {body}")
                hint = {401: "کلیلی Anthropic هەڵەیە.", 429: "زۆر داواکاری؛ کەمێک چاوەڕێ بکە.",
                        529: "سێرڤەر قەرەباڵغە؛ دووبارە هەوڵ بدەرەوە."}.get(e.code, f"هەڵەی API ({e.code}).")
                if e.code == 400 and "credit" in body.lower():
                    hint = "باڵانسی API تەواو بووە. لە console.anthropic.com پارە زیاد بکە."
                return self.tg.send(chat_id, "⚠ " + hint)
            except Exception as e:
                print(f"Claude call failed: {e}")
                return self.tg.send(chat_id, "⚠ کێشەیەک ڕوویدا، دووبارە هەوڵ بدەرەوە.")
            self.store.add(chat_id, "user", text or "[وێنە]")
            self.store.add(chat_id, "assistant", reply)
            self.store.add_usage(user_id, used)
            self.tg.send(chat_id, reply)

    def _build_content(self, msg, text):
        if msg.get("photo"):
            data = self.tg.download_photo(msg["photo"][-1]["file_id"])
            blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                   "data": base64.b64encode(data).decode("ascii")}}]
            blocks.append({"type": "text", "text": text or "ئەم وێنەیە باس بکە."})
            return blocks
        return text if text else None

    def run(self):
        print("Bot is running. Press Ctrl+C to stop.")
        offset = 0
        while True:
            try:
                for upd in self.tg.get_updates(offset):
                    offset = upd["update_id"] + 1
                    if "message" in upd:
                        self.pool.submit(self._safe, upd["message"])
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Polling error: {e}")
                time.sleep(5)

    def _safe(self, msg):
        try:
            self.handle(msg)
        except Exception as e:
            print(f"Handler error: {e}")


def main():
    cfg = load_config()
    if not cfg["telegram_token"] or not cfg["anthropic_api_key"]:
        print("config.json ی بکەرەوە و telegram_token و anthropic_api_key پڕ بکەرەوە، پاشان دووبارە ڕای بکە.")
        input("Enter...")
        sys.exit(1)
    Bot(cfg).run()


if __name__ == "__main__":
    main()
