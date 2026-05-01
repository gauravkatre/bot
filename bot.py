"""
Vera Bot — magicpin AI Challenge submission
A FastAPI server implementing the 5-endpoint contract with Claude-powered composition.
"""
import os, time, uuid, json, re
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="Vera Bot", version="1.0.0")
START = time.time()

# ── In-memory state ──────────────────────────────────────────────────────────
contexts: dict[tuple[str, str], dict] = {}    # (scope, context_id) -> {version, payload}
conversations: dict[str, list] = {}           # conversation_id -> [turns]
fired_suppressions: set[str] = set()          # suppression_key -> already sent this tick

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_payload(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def detect_auto_reply(message: str) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    auto_patterns = [
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
    return any(re.search(p, msg_lower) for p in auto_patterns)


def detect_stop_intent(message: str) -> bool:
    """Detect clear disengagement signals."""
    stop_patterns = [
        r"\bnot interested\b", r"\bno thanks\b", r"\bstop\b", r"\bunsubscribe\b",
        r"\bband karo\b", r"\bnahi chahiye\b", r"\bmat bhejo\b", r"\bblock\b",
        r"\bdo not contact\b", r"\bremove me\b",
    ]
    msg_lower = message.lower()
    return any(re.search(p, msg_lower) for p in stop_patterns)


def detect_action_intent(message: str) -> bool:
    """Detect clear 'let's go / yes / proceed' signals."""
    yes_patterns = [
        r"\byes\b", r"\bchalo\b", r"\blet'?s do it\b", r"\bgo ahead\b",
        r"\bok sure\b", r"\bsend it\b", r"\bplease do\b", r"\bkaro\b",
        r"\bhaan\b", r"\bthik hai\b", r"\bproceed\b", r"\bconfirm\b",
        r"^(yes|ok|okay|sure|haan|ha|yep|yup|👍)[\s!.]*$",
    ]
    msg_lower = message.lower().strip()
    return any(re.search(p, msg_lower) for p in yes_patterns)


def build_system_prompt() -> str:
    return """You are Vera, magicpin's merchant AI assistant. You craft WhatsApp messages for merchants across India.

CORE RULES:
1. SPECIFICITY WINS — anchor every message on a concrete, verifiable fact (number, date, headline, peer stat). Never say "increase your sales" — say "your CTR is 2.1% vs peer median 3.0%".
2. SERVICE+PRICE over flat discounts — "Haircut @ ₹99" beats "10% off".
3. VOICE MATCH — peer/colleague tone, not promotional. Match category voice (dentists = peer_clinical, salons = warm_aspirational, restaurants = energetic_local, gyms = motivational_data, pharmacies = trusted_advisor).
4. LANGUAGE — Hindi-English code-mix when merchant languages include "hi". Match the merchant's preference.
5. SINGLE CTA — binary (Reply YES/STOP) for action triggers; open-ended for info/curiosity triggers; none for pure-information.
6. NO FABRICATION — only cite data present in the context. Never invent competitor names, paper titles, or statistics.
7. COMPULSION LEVERS — use 1-2 per message: specificity, loss aversion, social proof, effort externalization, curiosity, reciprocity, single binary commitment.
8. NO PREAMBLE — start with the hook. Never "I hope you're doing well."
9. DON'T RE-INTRODUCE — after turn 1, no "Hi, I'm Vera."
10. ANTI-REPETITION — never send the same body twice in a conversation.

OUTPUT FORMAT (JSON only, no markdown):
{
  "body": "the WhatsApp message text",
  "cta": "binary_yes_stop" | "open_ended" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "copied from trigger",
  "rationale": "1-2 sentences: why this message, what lever it pulls"
}"""


def build_compose_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
    latest_merchant_message: Optional[str] = None,
    is_reply: bool = False,
) -> str:
    parts = []
    parts.append(f"=== CATEGORY CONTEXT ===\n{json.dumps(category, ensure_ascii=False, indent=2)}")
    parts.append(f"=== MERCHANT CONTEXT ===\n{json.dumps(merchant, ensure_ascii=False, indent=2)}")
    parts.append(f"=== TRIGGER ===\n{json.dumps(trigger, ensure_ascii=False, indent=2)}")
    if customer:
        parts.append(f"=== CUSTOMER CONTEXT ===\n{json.dumps(customer, ensure_ascii=False, indent=2)}")
    if conversation_history:
        parts.append(f"=== CONVERSATION SO FAR ===\n{json.dumps(conversation_history, ensure_ascii=False, indent=2)}")
    if latest_merchant_message:
        parts.append(f"=== MERCHANT'S LATEST MESSAGE ===\n{latest_merchant_message}")

    if is_reply and latest_merchant_message:
        task = """TASK: The merchant just replied. Craft your next message.
- If they said YES/proceed: move to action immediately. Don't re-qualify.
- If they asked a question: answer it specifically using context data.
- If they gave info: acknowledge + advance the conversation.
- If they seem confused: clarify simply."""
    elif customer:
        task = """TASK: Compose a customer-facing WhatsApp message (send_as = "merchant_on_behalf").
The message comes FROM the merchant TO their customer. Use the customer's name, language pref, and relationship state."""
    else:
        task = """TASK: Compose the first outbound WhatsApp message to the merchant.
This is a template message (first outbound). Make it count — merchants get many messages."""

    parts.append(task)
    parts.append("Respond with JSON only. No markdown. No preamble.")
    return "\n\n".join(parts)


def call_claude(system: str, user: str) -> dict:
    """Call Groq and parse JSON response."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1000,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = response.choices[0].message.content.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
    latest_merchant_message: Optional[str] = None,
    is_reply: bool = False,
) -> dict:
    """Core composition function — calls Claude with full context."""
    system = build_system_prompt()
    user = build_compose_prompt(
        category, merchant, trigger, customer,
        conversation_history, latest_merchant_message, is_reply
    )
    result = call_claude(system, user)
    # Validate required fields
    result.setdefault("body", "")
    result.setdefault("cta", "open_ended")
    result.setdefault("send_as", "vera")
    result.setdefault("suppression_key", trigger.get("suppression_key", ""))
    result.setdefault("rationale", "Composed from context")
    return result


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
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Enhanced",
        "team_members": ["Submission"],
        "model": "llama-3.3-70b-versatile",
        "approach": (
            "4-context Claude composer with auto-reply detection, intent routing, "
            "multi-turn state management, and category-voice enforcement"
        ),
        "contact_email": "submission@example.com",
        "version": "1.0.0",
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
        return {"accepted": False, "reason": "invalid_scope",
                "details": f"scope must be one of {valid_scopes}"}

    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload}
    ack_id = f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}"
    return {
        "accepted": True,
        "ack_id": ack_id,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    seen_merchants = set()  # one action per merchant per tick

    # Sort triggers by urgency (descending) — process most urgent first
    trigger_items = []
    for trg_id in body.available_triggers:
        trg = get_payload("trigger", trg_id)
        if trg:
            trigger_items.append((trg.get("urgency", 1), trg_id, trg))
    trigger_items.sort(key=lambda x: -x[0])

    for urgency, trg_id, trg in trigger_items:
        if len(actions) >= 20:
            break

        suppression_key = trg.get("suppression_key", "")
        if suppression_key and suppression_key in fired_suppressions:
            continue

        merchant_id = trg.get("merchant_id")
        if not merchant_id or merchant_id in seen_merchants:
            continue

        merchant = get_payload("merchant", merchant_id)
        if not merchant:
            continue

        category_slug = merchant.get("category_slug", "")
        category = get_payload("category", category_slug)
        if not category:
            continue

        # Customer context (for customer-scoped triggers)
        customer = None
        customer_id = trg.get("customer_id")
        if customer_id:
            customer = get_payload("customer", customer_id)

        # Skip if we already have an open conversation for this merchant
        conv_id = f"conv_{merchant_id}_{trg_id}"

        try:
            composed = compose_message(
                category=category,
                merchant=merchant,
                trigger=trg,
                customer=customer,
            )
        except Exception as e:
            # Fallback minimal message on error
            name = merchant.get("identity", {}).get("name", "there")
            composed = {
                "body": f"Hi {name}, checking in — let me know if there's anything I can help with.",
                "cta": "open_ended",
                "send_as": "vera",
                "suppression_key": suppression_key,
                "rationale": f"Fallback due to composition error: {str(e)[:80]}",
            }

        if not composed.get("body"):
            continue

        if suppression_key:
            fired_suppressions.add(suppression_key)
        seen_merchants.add(merchant_id)

        # Record in conversation history
        conversations[conv_id] = [{
            "from": "vera",
            "body": composed["body"],
            "ts": body.now,
            "trigger_id": trg_id,
        }]

        # Build template params (first 3 words of merchant name + trigger kind + snippet)
        template_params = [
            merchant.get("identity", {}).get("name", "Merchant"),
            trg.get("kind", "update"),
            composed["body"][:80],
        ]

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": template_params,
            "body": composed["body"],
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": suppression_key,
            "rationale": composed.get("rationale", ""),
        }
        actions.append(action)

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
    history = conversations.get(conv_id, [])

    # ── Auto-reply detection ──────────────────────────────────────────────────
    if detect_auto_reply(message):
        # Count consecutive auto-replies
        recent_auto = sum(
            1 for t in history[-4:]
            if t.get("from") == body.from_role and t.get("is_auto_reply")
        )
        if recent_auto >= 1:
            # Second auto-reply → graceful exit
            conversations[conv_id] = history + [{
                "from": "vera", "body": "[ended]", "ts": body.received_at
            }]
            return {
                "action": "end",
                "rationale": "Detected repeated auto-reply (WA Business canned response). Exiting gracefully to avoid burn."
            }
        else:
            # First auto-reply → try once more with a direct hook
            history.append({
                "from": body.from_role, "body": message,
                "ts": body.received_at, "is_auto_reply": True
            })
            conversations[conv_id] = history
            # One more attempt with a shorter, more direct message
            merchant = None
            if body.merchant_id:
                merchant = get_payload("merchant", body.merchant_id)
            name = merchant.get("identity", {}).get("name", "there") if merchant else "there"
            retry_body = (
                f"Koi baat nahi — jab free hoon tab baat karte hain. "
                f"Ek quick question: {name} mein is hafte sabse zyada kya service chal rahi hai? 🙂"
            )
            conversations[conv_id] = history + [{
                "from": "vera", "body": retry_body, "ts": body.received_at
            }]
            return {
                "action": "send",
                "body": retry_body,
                "cta": "open_ended",
                "rationale": "Detected auto-reply; sending one direct follow-up before exit."
            }

    # ── Stop intent detection ─────────────────────────────────────────────────
    if detect_stop_intent(message):
        conversations[conv_id] = history + [{
            "from": "vera", "body": "[ended]", "ts": body.received_at
        }]
        return {
            "action": "end",
            "rationale": "Merchant signaled disinterest/stop. Gracefully exiting."
        }

    # ── Normal reply — compose with Claude ───────────────────────────────────
    history.append({
        "from": body.from_role, "body": message, "ts": body.received_at
    })

    # Recover context for this conversation
    # conv_id format: conv_{merchant_id}_{trg_id}
    merchant_id = body.merchant_id
    customer_id = body.customer_id

    merchant = get_payload("merchant", merchant_id) if merchant_id else None
    customer = get_payload("customer", customer_id) if customer_id else None
    category = None
    trigger = None

    # Find trigger from conversation history
    for turn in history:
        if turn.get("trigger_id"):
            trigger = get_payload("trigger", turn["trigger_id"])
            break

    if merchant:
        category_slug = merchant.get("category_slug", "")
        category = get_payload("category", category_slug)

    if not (merchant and category and trigger):
        # Minimal fallback
        return {
            "action": "send",
            "body": "Theek hai, main check karke batata/batati hoon. Kuch aur help chahiye?",
            "cta": "open_ended",
            "rationale": "Context unavailable; minimal fallback."
        }

    # Check if action intent — move to action mode
    is_action_intent = detect_action_intent(message)

    try:
        composed = compose_message(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
            conversation_history=history,
            latest_merchant_message=message,
            is_reply=True,
        )
    except Exception as e:
        return {
            "action": "send",
            "body": "Bilkul! Main abhi yeh kar deta/deti hoon aur update karti hoon.",
            "cta": "open_ended",
            "rationale": f"Composition error fallback: {str(e)[:80]}"
        }

    body_text = composed.get("body", "")
    if not body_text:
        return {
            "action": "end",
            "rationale": "Empty composition; ending conversation."
        }

    # Anti-repetition check
    for turn in history:
        if turn.get("from") == "vera" and turn.get("body") == body_text:
            body_text = body_text + " — koi aur cheez? 🙂"

    history.append({"from": "vera", "body": body_text, "ts": body.received_at})
    conversations[conv_id] = history

    # After 5 turns from vera, wind down
    vera_turns = sum(1 for t in history if t.get("from") == "vera")
    if vera_turns >= 5:
        return {
            "action": "send",
            "body": body_text,
            "cta": "none",
            "rationale": composed.get("rationale", "") + " [final turn — winding down]"
        }

    return {
        "action": "send",
        "body": body_text,
        "cta": composed.get("cta", "open_ended"),
        "rationale": composed.get("rationale", ""),
    }


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    fired_suppressions.clear()
    return {"status": "wiped"}
