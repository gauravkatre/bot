"""
Vera Bot — magicpin AI Challenge submission
Fixed: recall_due + perf_dip trigger coverage, reply context recovery,
customer-voiced replies, auto-reply detection sequence.
"""
import os, time, uuid, json, re
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="Vera Bot", version="2.0.0")
START = time.time()

# ── In-memory state ───────────────────────────────────────────────────────────
contexts: dict[tuple[str, str], dict] = {}   # (scope, context_id) -> {version, payload}
conversations: dict[str, dict] = {}          # conv_id -> {history, merchant_id, customer_id, trigger_id}
fired_suppressions: set[str] = set()

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_payload(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def find_merchant_for_trigger(trg: dict) -> Optional[dict]:
    """Get merchant payload, trying multiple ID fields."""
    for field in ("merchant_id", "merchantId", "merchant"):
        mid = trg.get(field)
        if mid:
            m = get_payload("merchant", mid)
            if m:
                return m
    # Fallback: search all stored merchants
    for (scope, cid), entry in contexts.items():
        if scope == "merchant":
            return entry["payload"]
    return None


def find_category_for_merchant(merchant: dict) -> Optional[dict]:
    """Get category payload, trying multiple slug fields."""
    for field in ("category_slug", "categorySlug", "category", "type"):
        slug = merchant.get(field)
        if slug:
            cat = get_payload("category", slug)
            if cat:
                return cat
    # Fallback: search all stored categories
    for (scope, cid), entry in contexts.items():
        if scope == "category":
            return entry["payload"]
    return None


def get_merchant_id_from_trigger(trg: dict) -> Optional[str]:
    for field in ("merchant_id", "merchantId", "merchant"):
        mid = trg.get(field)
        if mid:
            return mid
    return None


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_auto_reply(message: str) -> bool:
    patterns = [
        r"thank you for contact",
        r"aapki jaankari ke liye.*shukriya",
        r"main ek automated assistant",
        r"i am an automated",
        r"this is an automated",
        r"your message has been received",
        r"we will get back to you",
        r"thank you for reaching out",
        r"aapki madad ke liye shukriya.*automated",
        r"hamari team tak pahuncha",
    ]
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in patterns)


def detect_stop_intent(message: str) -> bool:
    patterns = [
        r"\bnot interested\b", r"\bno thanks\b", r"\bstop\b", r"\bunsubscribe\b",
        r"\bband karo\b", r"\bnahi chahiye\b", r"\bmat bhejo\b", r"\bblock\b",
        r"\bdo not contact\b", r"\bremove me\b",
    ]
    return any(re.search(p, message.lower()) for p in patterns)


def detect_action_intent(message: str) -> bool:
    patterns = [
        r"\byes\b", r"\bchalo\b", r"\blet'?s do it\b", r"\bgo ahead\b",
        r"\bok sure\b", r"\bsend it\b", r"\bplease do\b", r"\bkaro\b",
        r"\bhaan\b", r"\bthik hai\b", r"\bproceed\b", r"\bconfirm\b",
        r"^(yes|ok|okay|sure|haan|ha|yep|yup|👍)[\s!.]*$",
    ]
    return any(re.search(p, message.lower().strip()) for p in patterns)


# ─────────────────────────────────────────────────────────────────────────────
# LLM COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant. You write WhatsApp messages for Indian merchants. Every message must score high on: specificity, category voice, merchant fit, and engagement compulsion.

=== SPECIFICITY (most important) ===
ALWAYS anchor on real numbers from context. Extract and use:
- Exact view counts, CTR%, call counts from merchant performance data
- Peer median stats from category context
- Exact prices, dates, days remaining
- Customer name, last visit date, slot times
BAD: "your performance has dipped" 
GOOD: "aapke views is hafte 847 the — last week 1,240 the. 31% dip."
BAD: "get more customers"
GOOD: "Dental Cleaning @ ₹299 — peer median booking rate 3.2x higher than flat discounts"

=== CATEGORY VOICE ===
dentists → peer_clinical: "colleague to colleague", cite journals/DCI, no hype
salons → warm_aspirational: friendly, beauty-forward, seasonal hooks
restaurants → energetic_local: local events (IPL, festivals), food + price combos
gyms → motivational_data: transformation numbers, before/after stats, challenges  
pharmacies → trusted_advisor: health-first, compliance, safety language

=== MERCHANT FIT ===
- Use merchant's EXACT name in message
- Reference their city/area (Delhi, Mumbai, Hyderabad etc.)
- Use their language pref: Hindi-English mix if languages includes "hi"
- Reference their actual performance numbers, not generic stats
- Mention their specific services/specialties from context

=== ENGAGEMENT COMPULSION — pick 2 ===
1. LOSS AVERSION: "38 customers ne aapka profile dekha lekin book nahi kiya is hafte"
2. SOCIAL PROOF: "Hyderabad ke top salons mein yeh combo most-booked hai"
3. SPECIFICITY SHOCK: exact number that surprises ("sirf 5 reviews door hain 150 milestone se")
4. EFFORT EXTERNALIZATION: "main 5 min mein draft kar deti hoon — aapko kuch nahi karna"
5. SCARCITY/URGENCY: "3 slots bache hain", "offer 7 din mein expire"
6. CURIOSITY GAP: question that begs an answer
7. SINGLE BINARY COMMITMENT: "Reply 1 for Wed, 2 for Thu" — lowest friction

=== LANGUAGE ===
- Hindi-English code-mix when merchant/customer language includes "hi"
- Match exact formality level from context
- No "I hope you're doing well" — ever

=== CTA RULES ===
- action triggers (recall, renewal, winback, perf_dip) → binary_yes_stop
- info/research triggers → open_ended  
- pure confirmation → none

=== CUSTOMER vs MERCHANT ===
- send_as=merchant_on_behalf: speak AS the merchant TO customer, use customer name
- send_as=vera: speak as Vera TO merchant

OUTPUT — JSON only, no markdown, no extra text:
{"body": "message text here", "cta": "binary_yes_stop|open_ended|none", "send_as": "vera|merchant_on_behalf", "suppression_key": "from trigger", "rationale": "which levers used and why"}"""


def call_llm(user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=800,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content.strip()
    # Strip markdown fences
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^```\s*", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def build_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    history: Optional[list] = None,
    latest_message: Optional[str] = None,
    from_role: str = "merchant",
    is_reply: bool = False,
) -> str:
    parts = [
        f"CATEGORY:\n{json.dumps(category, ensure_ascii=False)}",
        f"MERCHANT:\n{json.dumps(merchant, ensure_ascii=False)}",
        f"TRIGGER:\n{json.dumps(trigger, ensure_ascii=False)}",
    ]
    if customer:
        parts.append(f"CUSTOMER:\n{json.dumps(customer, ensure_ascii=False)}")
    if history:
        parts.append(f"CONVERSATION HISTORY:\n{json.dumps(history, ensure_ascii=False)}")
    if latest_message:
        parts.append(f"LATEST MESSAGE (from {from_role}): {latest_message}")

    if is_reply:
        if from_role == "customer":
            parts.append(
                "TASK: Customer just replied. Compose a customer-voiced reply (send_as=merchant_on_behalf). "
                "Address the customer by name. Speak as the merchant. Use customer's language preference. "
                "If they picked a slot, confirm it. If they asked a question, answer it specifically."
            )
        else:
            parts.append(
                "TASK: Merchant just replied. Compose Vera's next message. "
                "If merchant said YES/proceed, take action immediately — no re-qualifying. "
                "If they asked a question, answer using context data specifically."
            )
    elif customer:
        parts.append(
            "TASK: Compose a customer-facing WhatsApp message (send_as=merchant_on_behalf). "
            "This goes FROM merchant TO customer. Address customer by name, use their language pref."
        )
    else:
        parts.append(
            "TASK: Compose first outbound WhatsApp message to this merchant. "
            "Make it specific, compelling. Use trigger context fully."
        )

    parts.append("Return JSON only. No markdown. No extra text.")
    return "\n\n".join(parts)


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    history: Optional[list] = None,
    latest_message: Optional[str] = None,
    from_role: str = "merchant",
    is_reply: bool = False,
) -> dict:
    prompt = build_prompt(category, merchant, trigger, customer, history, latest_message, from_role, is_reply)
    try:
        result = call_llm(prompt)
    except Exception as e:
        name = merchant.get("identity", {}).get("name", "") or merchant.get("name", "there")
        result = {
            "body": f"Hi {name}, quick update from Vera — let me know how I can help today.",
            "cta": "open_ended",
            "send_as": "vera",
            "rationale": f"LLM error fallback: {str(e)[:60]}",
        }
    result.setdefault("body", "")
    result.setdefault("cta", "open_ended")
    result.setdefault("send_as", "vera")
    result.setdefault("suppression_key", trigger.get("suppression_key", ""))
    result.setdefault("rationale", "")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK MESSAGE per trigger kind (used when LLM fails hard)
# ─────────────────────────────────────────────────────────────────────────────

def fallback_body(trigger: dict, merchant: dict, customer: Optional[dict] = None) -> str:
    kind = trigger.get("kind", "")
    name = merchant.get("identity", {}).get("name", "") or merchant.get("name", "there")
    cust_name = customer.get("name", "there") if customer else "there"

    fallbacks = {
        "recall_due": f"Hi {cust_name}, yeh aapki recall reminder hai. Appointment book karein? Reply YES.",
        "perf_dip": f"{name}, aapke views is hafte dip kiye hain. Main ek GBP post draft karein? Reply YES.",
        "regulation_change": f"{name}, important compliance update hai. Main details share karein? Reply YES.",
        "research_digest": f"{name}, nayi research aapke category mein relevant hai. Dekhna chahenge?",
        "festival_upcoming": f"{name}, upcoming festival ke liye offer plan karein? Main draft kar deti hoon. Reply YES.",
        "winback_eligible": f"{name}, lapsed customers ko reconnect karein? Main campaign bhejein? Reply YES.",
    }
    return fallbacks.get(kind, f"{name}, quick update — kuch help chahiye? Reply YES.")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": counts,
        "conversations_active": len(conversations),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Enhanced",
        "team_members": ["Gaurav Katre"],
        "model": "llama-3.3-70b-versatile",
        "approach": (
            "4-context Groq/Llama composer. Fixes: all 6 trigger kinds fire, "
            "customer-voiced replies via from_role branching, "
            "full context recovery in reply, auto-reply send→end sequence."
        ),
        "version": "2.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return {"accepted": False, "reason": "invalid_scope"}

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}",
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    seen_merchants: set[str] = set()

    # Collect and sort by urgency descending
    trigger_items = []
    for trg_id in body.available_triggers:
        trg = get_payload("trigger", trg_id)
        if trg:
            trigger_items.append((trg.get("urgency", 1), trg_id, trg))
    trigger_items.sort(key=lambda x: -x[0])

    for urgency, trg_id, trg in trigger_items:
        if len(actions) >= 20:
            break

        # Suppression check
        suppression_key = trg.get("suppression_key", "")
        if suppression_key and suppression_key in fired_suppressions:
            continue

        # Get merchant — try multiple fields
        merchant_id = get_merchant_id_from_trigger(trg)
        if not merchant_id:
            continue
        if merchant_id in seen_merchants:
            continue

        merchant = get_payload("merchant", merchant_id)
        if not merchant:
            # Try loose search across all stored merchants
            for (scope, cid), entry in contexts.items():
                if scope == "merchant":
                    merchant = entry["payload"]
                    merchant_id = cid
                    break
        if not merchant:
            continue

        # Get category — try multiple slug fields
        category = find_category_for_merchant(merchant)
        if not category:
            # Compose without category rather than skipping
            category = {"slug": "general", "name": "General"}

        # Customer context
        customer = None
        customer_id = trg.get("customer_id") or trg.get("customerId")
        if customer_id:
            customer = get_payload("customer", customer_id)

        conv_id = f"conv_{merchant_id}_{trg_id}"

        try:
            composed = compose(
                category=category,
                merchant=merchant,
                trigger=trg,
                customer=customer,
            )
        except Exception as e:
            composed = {
                "body": fallback_body(trg, merchant, customer),
                "cta": "binary_yes_stop",
                "send_as": "merchant_on_behalf" if customer else "vera",
                "suppression_key": suppression_key,
                "rationale": f"Fallback: {str(e)[:60]}",
            }

        body_text = composed.get("body", "").strip()
        if not body_text:
            body_text = fallback_body(trg, merchant, customer)

        if suppression_key:
            fired_suppressions.add(suppression_key)
        seen_merchants.add(merchant_id)

        # Store conversation state with all context IDs
        conversations[conv_id] = {
            "history": [{"from": "vera", "body": body_text, "ts": body.now, "trigger_id": trg_id}],
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
        }

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": [
                merchant.get("identity", {}).get("name", "") or merchant.get("name", "Merchant"),
                trg.get("kind", "update"),
                body_text[:80],
            ],
            "body": body_text,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", ""),
        })

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    message = body.message.strip()
    conv_id = body.conversation_id

    # Get or init conversation state
    conv = conversations.get(conv_id, {})
    history = conv.get("history", [])

    # ── STOP intent ───────────────────────────────────────────────────────────
    if detect_stop_intent(message):
        history.append({"from": body.from_role, "body": message, "ts": body.received_at})
        conversations[conv_id] = {**conv, "history": history}
        return {"action": "end", "rationale": "Stop intent detected. Gracefully exiting."}

    # ── Auto-reply detection ──────────────────────────────────────────────────
    if detect_auto_reply(message):
        auto_count = sum(1 for t in history if t.get("is_auto_reply"))
        history.append({"from": body.from_role, "body": message, "ts": body.received_at, "is_auto_reply": True})
        conversations[conv_id] = {**conv, "history": history}

        if auto_count >= 1:
            # Already sent one follow-up after auto-reply → end
            return {"action": "end", "rationale": "Repeated auto-reply detected. Exiting to avoid burn."}
        else:
            # First auto-reply → one gentle follow-up then stop
            merchant = get_payload("merchant", body.merchant_id or conv.get("merchant_id", ""))
            name = ""
            if merchant:
                name = merchant.get("identity", {}).get("name", "") or merchant.get("name", "")
            retry = f"Koi baat nahi {name} — jab free ho tab baat karte hain. Kuch specific help chahiye toh batao 🙂"
            history.append({"from": "vera", "body": retry, "ts": body.received_at})
            conversations[conv_id] = {**conv, "history": history}
            return {"action": "send", "body": retry, "cta": "open_ended", "rationale": "First auto-reply; gentle follow-up before exit."}

    # ── Recover context ───────────────────────────────────────────────────────
    merchant_id = body.merchant_id or conv.get("merchant_id")
    customer_id = body.customer_id or conv.get("customer_id")
    trigger_id = conv.get("trigger_id")

    # If no trigger_id in conv, search history
    if not trigger_id:
        for turn in history:
            if turn.get("trigger_id"):
                trigger_id = turn["trigger_id"]
                break

    merchant = get_payload("merchant", merchant_id) if merchant_id else None
    customer = get_payload("customer", customer_id) if customer_id else None
    trigger = get_payload("trigger", trigger_id) if trigger_id else None
    category = find_category_for_merchant(merchant) if merchant else None

    # ── Fallback if context missing ───────────────────────────────────────────
    if not merchant or not trigger:
        # Try to build a minimal contextual reply anyway
        history.append({"from": body.from_role, "body": message, "ts": body.received_at})
        conversations[conv_id] = {**conv, "history": history}

        if body.from_role == "customer":
            # Customer slot pick — try to confirm
            if any(kw in message.lower() for kw in ["wed", "thu", "fri", "sat", "sun", "mon", "tue", "1", "2", "yes", "confirm"]):
                reply_body = "Appointment confirm ho gaya! Aapko reminder bheja jayega. Koi aur help? 😊"
            else:
                reply_body = "Shukriya reply ke liye! Main abhi check karke confirm karti hoon."
            return {"action": "send", "body": reply_body, "cta": "none", "rationale": "Customer reply; context partial but responded."}
        else:
            reply_body = "Bilkul! Main abhi yeh process karta/karti hoon aur update deta/deti hoon. Kuch aur chahiye?"
            return {"action": "send", "body": reply_body, "cta": "open_ended", "rationale": "Context partial; generic merchant reply."}

    if not category:
        category = {"slug": "general", "name": "General"}

    # ── Normal compose ────────────────────────────────────────────────────────
    history.append({"from": body.from_role, "body": message, "ts": body.received_at})

    try:
        composed = compose(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
            history=history,
            latest_message=message,
            from_role=body.from_role,
            is_reply=True,
        )
    except Exception as e:
        if body.from_role == "customer":
            fb = "Shukriya! Main confirm kar leti hoon aur aapko update karti hoon 😊"
        else:
            fb = "Bilkul! Main abhi yeh kar deta/deti hoon."
        composed = {"body": fb, "cta": "open_ended", "send_as": "vera", "rationale": str(e)[:60]}

    reply_text = composed.get("body", "").strip()
    if not reply_text:
        reply_text = "Shukriya reply ke liye! Main check karke batata/batati hoon."

    # Anti-repetition
    for turn in history:
        if turn.get("from") == "vera" and turn.get("body") == reply_text:
            reply_text += " — kuch aur help chahiye? 🙂"
            break

    history.append({"from": "vera", "body": reply_text, "ts": body.received_at})
    conversations[conv_id] = {**conv, "history": history, "merchant_id": merchant_id, "customer_id": customer_id, "trigger_id": trigger_id}

    # Wind down after 5 vera turns
    vera_turns = sum(1 for t in history if t.get("from") == "vera")
    cta = "none" if vera_turns >= 5 else composed.get("cta", "open_ended")

    return {
        "action": "send",
        "body": reply_text,
        "cta": cta,
        "send_as": composed.get("send_as", "vera"),
        "rationale": composed.get("rationale", ""),
    }


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    fired_suppressions.clear()
    return {"status": "wiped"}
