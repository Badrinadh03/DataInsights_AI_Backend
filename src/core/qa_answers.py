import re, unicodedata

# --- Fixed answers you provided ---
FIXED_ANSWERS = {
    "how’s the market today?": (
        "Like your auditor during year-end — very unpredictable, slightly moody, "
        "and asking for too many reconciliations"
    ),
    "hows the market today?": (
        "Like your auditor during year-end — very unpredictable, slightly moody, "
        "and asking for too many reconciliations"
    ),
    "tell me the secret of making guaranteed profits in the stock market.": (
        "Simple! Marry a broker, make friends with SEBI, and always buy HDFC shares at Diwali. "
        "Disclaimer: Past performance may or may not impress your mother-in-law."
    ),
    "how is our investment portfolio doing today?": (
        "Think of it like an Indian wedding buffet — the equity section is spicy and giving you heartburn, "
        "the bonds are the plain curd rice keeping everything stable, and the alternative investments are like "
        "that mysterious dessert… nobody’s sure what it is, but everyone pretends it’s a delicacy."
    ),
}

# Optional fuzzy triggers (so minor wording changes still match)
FIXED_TRIGGERS = [
    (re.compile(r"\bhow'?s?\s+the\s+market\s+today\b", re.I), FIXED_ANSWERS["hows the market today?"]),
    (re.compile(r"\bsecret\b.*\bguaranteed\b.*\b(profit|profits)\b.*\bstock\s*market\b", re.I),
     FIXED_ANSWERS["tell me the secret of making guaranteed profits in the stock market."]),
    (re.compile(r"\bhow\s+is\s+our\s+investment\s+portfolio\s+doing\s+today\b", re.I),
     FIXED_ANSWERS["how is our investment portfolio doing today?"]),
]

def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)  # normalize quotes etc.
    return re.sub(r"\s+", " ", s).lower()

def get_fixed_answer(question: str):
    q_norm = _normalize_text(question)
    if q_norm in FIXED_ANSWERS:
        return FIXED_ANSWERS[q_norm]
    for rx, ans in FIXED_TRIGGERS:
        if rx.search(q_norm):
            return ans
    return None
