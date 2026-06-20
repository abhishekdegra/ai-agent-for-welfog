"""
AI-first delivery location resolver — city name or PIN → live check_pincode API.

Scoped to delivery / serviceability / pincode intents only — does not affect other APIs.

Primary: delivery micro-classifier (meaning + recent chat context, any language).
Geocode: Google Maps (optional) → Nominatim pan-India → major-city offline fallback.
PIN prompt: only when place is vague/gibberish, question incomplete, or AI cannot resolve.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from utils.reasoning_log import log_reasoning

_DELIVERY_TURN_CACHE = threading.local()
_DELIVERY_AI_CONF_SERVICEABILITY = 0.55
_DELIVERY_AI_CONF_FOLLOWUP = 0.50
_DELIVERY_AI_CONF_LOCATION = 0.48

_PIN_RE = re.compile(r"\b([1-9]\d{5})\b")
_CITY_GEO_CACHE: dict[str, tuple[float, dict]] = {}
_CITY_GEO_TTL_SEC = 3600

# Offline fallback when geocoders are down — representative central PIN (pan-India metros).
_MAJOR_CITY_PINS: dict[str, str] = {
    "agra": "282001", "ahmedabad": "380001", "ajmer": "305001", "aligarh": "202001",
    "allahabad": "211001", "prayagraj": "211001", "amritsar": "143001", "aurangabad": "431001",
    "bangalore": "560001", "bengaluru": "560001", "bhopal": "462001", "bhubaneswar": "751001",
    "bikaner": "334001", "chandigarh": "160001", "chennai": "600001", "coimbatore": "641001",
    "dehradun": "248001", "delhi": "110001", "new delhi": "110001", "dhanbad": "826001",
    "faridabad": "121001", "ghaziabad": "201001", "goa": "403001", "panaji": "403001",
    "guwahati": "781001", "gurgaon": "122001", "gurugram": "122001", "gwalior": "474001",
    "hubli": "580001", "hyderabad": "500001", "indore": "452001", "jabalpur": "482001",
    "jaipur": "302001", "jalandhar": "144001", "jammu": "180001", "jodhpur": "342001",
    "kanpur": "208001", "kochi": "682001", "cochin": "682001", "kolkata": "700001",
    "kota": "324001", "lucknow": "226001", "ludhiana": "141001", "madurai": "625001",
    "mangalore": "575001", "meerut": "250001", "mumbai": "400001", "mysore": "570001",
    "mysuru": "570001", "nagpur": "440001", "nashik": "422001", "noida": "201301",
    "patna": "800001", "pondicherry": "605001", "puducherry": "605001", "pune": "411001",
    "raipur": "492001", "rajkot": "360001", "ranchi": "834001", "shimla": "171001",
    "srinagar": "190001", "surat": "395001", "thiruvananthapuram": "695001",
    "trivandrum": "695001", "udaipur": "313001", "vadodara": "390001", "varanasi": "221001",
    "vijayawada": "520001", "visakhapatnam": "530001", "vizag": "530001",
    "bharatpur": "321001", "sikar": "332001", "alwar": "301001",
}

_NOMINATIM_HEADERS = {"User-Agent": "WelfogSupportBot/1.0 (delivery-check; pan-india)"}

_DELIVERY_PLACE_STOPWORDS_RE = re.compile(
    r"\b(?:"
    r"aur|bhi|kya|ky|mil|jaygi|jayega|jaaygi|jaayega|milega|milegi|milta|milti|"
    r"delivery|deliver|delevery|delivry|service|dega|degi|pahucha|pahunch|"
    r"check|kr|kar|kro|krke|karke|bta|bata|btao|batao|btana|bol|"
    r"bhai|welfog|ho|hai|hogi|hoga|h|na|nahi|ni|"
    r"me|mein|par|pe|per|in|at|to|for|"
    r"yaha|yahan|waha|wahan|idhar|udhar|"
    r"please|pls|bhejo|dena|dede|the|a|an|is|will|can|"
    r"order|place|karna|karenge|kru|karu|chahta|chahti"
    r")\b",
    re.I,
)

# Tokens that name a person/relation/zone — not a geocodable city (structural, not product keywords).
_RELATIONAL_PLACE_WORDS = frozenset(
    {
        "dost", "friend", "friends", "yaar", "bhai", "behan", "sister", "brother",
        "cousin", "relative", "relatives", "papa", "mummy", "mom", "dad", "parents",
        "wife", "husband", "ghar", "home", "office", "college", "school", "mere",
        "mera", "meri", "uska", "uski", "unka", "unke", "didi", "bhabhi", "chacha",
        "mama", "nana", "nani", "saas", "sasur", "beti", "beta", "saheli", "saheli",
        "area", "locality", "ilaka", "jagah", "jagha", "pas", "paas", "aas", "ke",
    }
)


@dataclass
class ResolvedDeliveryLocation:
    kind: str  # pincode | city_geocoded | ask_pin | none
    pincode: str = ""
    city_label: str = ""
    state_hint: str = ""
    source: str = ""
    confidence: str = ""
    geocode_display: str = ""
    lat: float | None = None
    lng: float | None = None
    reasoning: str = ""


@dataclass
class DeliveryTurnUnderstanding:
    """Single-turn delivery/serviceability understanding (AI-first)."""

    is_serviceability: bool = False
    is_area_followup: bool = False
    location_kind: str = "none"  # none | pincode | city | ask_pin
    pincode: str = ""
    city_name: str = ""
    state_hint: str = ""
    user_meaning: str = ""
    confidence: float = 0.0
    source: str = ""


def _combined(original_msg: str, msg_en: str = "") -> str:
    return f"{original_msg or ''} {msg_en or ''}".strip()


def _norm_city_key(city: str) -> str:
    return re.sub(r"\s+", " ", (city or "").strip().lower())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def _fuzzy_major_city_typo(place: str, *, max_edits: int = 2) -> str:
    """Typo-tolerant match against major cities (e.g. noice → noida) — not a static keyword router."""
    key = _norm_city_key(place)
    if len(key) < 4 or key in _MAJOR_CITY_PINS:
        return ""
    best_city = ""
    best_dist = max_edits + 1
    for city in _MAJOR_CITY_PINS:
        d = _levenshtein(key, city)
        if d < best_dist:
            best_dist = d
            best_city = city
    if best_city and best_dist <= max_edits:
        log_reasoning(f"City typo fuzzy: {place!r} → {best_city!r} (edit distance {best_dist})")
        return best_city
    return ""


def extract_place_query_from_delivery_message(text: str) -> str:
    """
    Strip delivery/Hinglish filler — leave likely city/area/locality name.
    Used when LLM says ask_pin but message still names a place.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    cleaned = _DELIVERY_PLACE_STOPWORDS_RE.sub(" ", raw)
    cleaned = re.sub(r"[^\w\s\-]", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = [t for t in cleaned.split() if len(t) >= 3 and not t.isdigit()]
    if not tokens:
        return ""
    return " ".join(tokens[:4]).strip()


def _place_is_relational_or_vague_reference(place: str) -> bool:
    """Person/zone reference (friend's area) — not a geocodable city name."""
    p = re.sub(r"\s+", " ", (place or "").strip().lower())
    if not p:
        return False
    tokens = [t for t in p.split() if len(t) >= 2]
    if not tokens:
        return False
    if all(t in _RELATIONAL_PLACE_WORDS for t in tokens):
        return True
    if len(tokens) == 1 and tokens[0] in ("area", "locality", "jagah", "jagha", "ilaka"):
        return True
    return False


def _place_name_looks_gibberish(place: str) -> bool:
    """Random / nonsense place — safe to ask for PIN."""
    if _place_is_relational_or_vague_reference(place):
        return True
    p = re.sub(r"\s+", " ", (place or "").strip().lower())
    if not p or len(p) < 2:
        return True
    if re.fullmatch(r"[\d\s\-]+", p):
        return True
    alpha = re.sub(r"[^a-z]", "", p)
    if len(alpha) < 2:
        return True
    if len(p) <= 5 and p not in _MAJOR_CITY_PINS and not re.search(r"[aeiouy]", alpha):
        return True
    if re.search(r"(.)\1{3,}", alpha):
        return True
    return False


def _delivery_question_incomplete(
    comb: str,
    understood: DeliveryTurnUnderstanding | None,
) -> bool:
    """User wants delivery check but did not name any place (half question)."""
    u = understood or DeliveryTurnUnderstanding()
    place = (u.city_name or "").strip() or extract_place_query_from_delivery_message(comb)
    if place and _place_is_relational_or_vague_reference(place):
        return True
    if place and len(place) >= 3 and not _place_name_looks_gibberish(place):
        return False
    return bool(
        u.is_serviceability
        or u.is_area_followup
        or u.location_kind in ("ask_pin", "city")
    )


def should_ask_user_for_pincode(
    understood: DeliveryTurnUnderstanding | None,
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    *,
    loc: ResolvedDeliveryLocation | None = None,
) -> bool:
    """
    PIN prompt only inside delivery/serviceability flow when:
    - question is incomplete (no place named), OR
    - place name looks gibberish, OR
    - AI explicitly could not resolve location (ask_pin) after geocode attempts.
    """
    comb = _combined(original_msg, msg_en)
    u = understood or DeliveryTurnUnderstanding()

    if not (u.is_serviceability or u.is_area_followup or u.location_kind in ("ask_pin", "city")):
        return False

    place = (u.city_name or "").strip() or extract_place_query_from_delivery_message(comb)

    if place and _place_is_relational_or_vague_reference(place):
        log_reasoning("Delivery: ask PIN — relational/vague area reference (not a city).")
        return True

    if _delivery_question_incomplete(comb, u):
        log_reasoning("Delivery: ask PIN — question incomplete (no clear place).")
        return True

    if place and _place_name_looks_gibberish(place):
        log_reasoning(f"Delivery: ask PIN — place {place!r} looks unresolvable.")
        return True

    if (loc or ResolvedDeliveryLocation()).source == "city_unresolved" and place:
        log_reasoning(f"Delivery: ask PIN — geocoder could not map {place!r}.")
        return True

    if u.location_kind == "ask_pin" and u.confidence >= _DELIVERY_AI_CONF_SERVICEABILITY:
        resolved_pin = (loc or ResolvedDeliveryLocation()).pincode
        if not resolved_pin:
            log_reasoning("Delivery: ask PIN — delivery classifier (no resolved PIN).")
            return True
        if not place:
            return True
        if _place_name_looks_gibberish(place):
            return True

    return False


def _row_to_geocode_out(row: dict, city: str, state_hint: str, source: str) -> dict[str, Any]:
    addr = row.get("address") or {}
    postcode = re.sub(r"\D", "", str(addr.get("postcode") or ""))
    if len(postcode) != 6 or postcode[0] == "0":
        return {}
    return {
        "pincode": postcode,
        "city_label": str(
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("suburb")
            or addr.get("state_district")
            or city.strip()
        ),
        "state_hint": str(addr.get("state") or state_hint or "").strip(),
        "lat": float(row["lat"]) if row.get("lat") else None,
        "lng": float(row["lon"]) if row.get("lon") else None,
        "display_name": str(row.get("display_name") or "")[:200],
        "confidence": "high" if source == "nominatim" else "medium",
        "source": source,
    }


def _nominatim_reverse_pincode(lat: float, lng: float) -> str:
    try:
        res = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lng,
                "format": "json",
                "addressdetails": 1,
                "zoom": 18,
            },
            headers=_NOMINATIM_HEADERS,
            timeout=10,
        )
        if res.status_code == 200:
            addr = (res.json() or {}).get("address") or {}
            digits = re.sub(r"\D", "", str(addr.get("postcode") or ""))
            if len(digits) == 6 and digits[0] != "0":
                return digits
    except Exception:
        pass
    return ""


def _nominatim_search(place: str, query: str) -> list[dict]:
    try:
        res = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "addressdetails": 1,
                "limit": 8,
                "countrycodes": "in",
            },
            headers=_NOMINATIM_HEADERS,
            timeout=12,
        )
        if res.status_code == 200:
            return res.json() or []
    except Exception as exc:
        log_reasoning(f"Nominatim search failed for {place!r} ({query!r}): {exc}")
    return []


def _nominatim_geocode_india(
    city: str,
    *,
    state_hint: str = "",
    country: str = "India",
) -> dict[str, Any]:
    """Pan-India geocode — multiple query shapes + reverse lookup when needed."""
    place = (city or "").strip()
    if not place:
        return {}

    queries: list[str] = []
    if state_hint:
        queries.append(f"{place}, {state_hint}, {country}")
    queries.append(f"{place}, {country}")
    if "," not in place:
        queries.append(place)

    seen_q: set[str] = set()
    for query in queries:
        qn = query.strip().lower()
        if not qn or qn in seen_q:
            continue
        seen_q.add(qn)
        for row in _nominatim_search(place, query):
            out = _row_to_geocode_out(row, place, state_hint, "nominatim")
            if out:
                return out
            try:
                lat = float(row.get("lat") or 0)
                lng = float(row.get("lon") or 0)
            except (TypeError, ValueError):
                lat, lng = 0.0, 0.0
            if lat and lng:
                pin = _nominatim_reverse_pincode(lat, lng)
                if pin:
                    out = {
                        "pincode": pin,
                        "city_label": place,
                        "state_hint": state_hint or "",
                        "lat": lat,
                        "lng": lng,
                        "display_name": str(row.get("display_name") or "")[:200],
                        "confidence": "medium",
                        "source": "nominatim_reverse",
                    }
                    return out
    return {}


def _geocode_google_maps(
    city: str,
    *,
    state_hint: str = "",
    country: str = "India",
    api_key: str = "",
) -> dict[str, Any]:
    """Google Geocoding API — optional primary when GOOGLE_MAPS_API_KEY is set."""
    if not api_key or not (city or "").strip():
        return {}
    query = f"{city.strip()}, {state_hint.strip()}, {country}".strip(" ,")
    try:
        res = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": query, "key": api_key, "region": "in"},
            timeout=12,
        )
        if res.status_code != 200:
            return {}
        payload = res.json() or {}
        if (payload.get("status") or "").upper() not in ("OK", "ZERO_RESULTS"):
            log_reasoning(f"Google geocode status={payload.get('status')} for {city!r}")
        for row in payload.get("results") or []:
            pin = ""
            city_label = city.strip()
            state = state_hint or ""
            for comp in row.get("address_components") or []:
                types = comp.get("types") or []
                if "postal_code" in types:
                    digits = re.sub(r"\D", "", str(comp.get("long_name") or ""))
                    if len(digits) == 6 and digits[0] != "0":
                        pin = digits
                if "locality" in types or "sublocality" in types:
                    city_label = str(comp.get("long_name") or city_label)
                if "administrative_area_level_1" in types:
                    state = str(comp.get("long_name") or state)
            if pin:
                loc = row.get("geometry", {}).get("location") or {}
                return {
                    "pincode": pin,
                    "city_label": city_label,
                    "state_hint": state,
                    "lat": loc.get("lat"),
                    "lng": loc.get("lng"),
                    "display_name": str(row.get("formatted_address") or "")[:200],
                    "confidence": "high",
                    "source": "google_maps",
                }
    except Exception as exc:
        log_reasoning(f"Google geocode failed for {city!r}: {exc}")
    return {}


def geocode_city_to_pincode(
    city: str,
    *,
    state_hint: str = "",
    country: str = "India",
) -> dict[str, Any]:
    """
    Map city/area/locality name → representative 6-digit PIN (anywhere in India).
    Order: cache → Google Maps (if key) → Nominatim pan-India → major-city offline fallback.
    """
    key = _norm_city_key(city)
    if not key or len(key) < 2:
        return {}
    cache_key = f"{key}|{(state_hint or '').lower()}|{country.lower()}"
    now = time.time()
    cached = _CITY_GEO_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _CITY_GEO_TTL_SEC:
        return dict(cached[1])

    out: dict[str, Any] = {}
    gkey = (
        (os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_GEOCODING_API_KEY") or "")
        .strip()
    )
    if gkey:
        out = _geocode_google_maps(city, state_hint=state_hint, country=country, api_key=gkey)
        if out.get("pincode"):
            _CITY_GEO_CACHE[cache_key] = (now, out)
            log_reasoning(f"City geocode (Google): {city!r} → PIN {out['pincode']}")
            return out

    out = _nominatim_geocode_india(city, state_hint=state_hint, country=country)
    if out.get("pincode"):
        _CITY_GEO_CACHE[cache_key] = (now, out)
        log_reasoning(f"City geocode (Nominatim): {city!r} → PIN {out['pincode']}")
        return out

    fuzzy_key = _fuzzy_major_city_typo(city)
    if fuzzy_key:
        fallback = _MAJOR_CITY_PINS.get(fuzzy_key)
        if fallback:
            out = {
                "pincode": fallback,
                "city_label": fuzzy_key.title(),
                "state_hint": state_hint or "",
                "lat": None,
                "lng": None,
                "display_name": f"{fuzzy_key.title()}, India",
                "confidence": "medium",
                "source": "city_typo_fuzzy",
            }
            _CITY_GEO_CACHE[cache_key] = (now, out)
            return out

    fallback = _MAJOR_CITY_PINS.get(key)
    if fallback:
        out = {
            "pincode": fallback,
            "city_label": city.strip(),
            "state_hint": state_hint or "",
            "lat": None,
            "lng": None,
            "display_name": f"{city.strip()}, India",
            "confidence": "medium",
            "source": "city_fallback",
        }
        _CITY_GEO_CACHE[cache_key] = (now, out)
        log_reasoning(f"City geocode (offline fallback): {city!r} → PIN {fallback}")
        return out

    return {}


def _delivery_turn_cache_get(key: str) -> Optional[DeliveryTurnUnderstanding]:
    store = getattr(_DELIVERY_TURN_CACHE, "by_key", None)
    if not store:
        return None
    return store.get(key)


def _delivery_turn_cache_put(key: str, value: DeliveryTurnUnderstanding) -> None:
    store = getattr(_DELIVERY_TURN_CACHE, "by_key", None)
    if store is None:
        store = {}
        _DELIVERY_TURN_CACHE.by_key = store
    store[key] = value


def _delivery_turn_cache_key(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
) -> str:
    tail = (conversation_context or "")[-1200:]
    return f"{original_msg}|{msg_en}|{tail}"


def _pincode_thread_context_hints(conversation_context: str) -> list[str]:
    hints: list[str] = []
    try:
        from utils.helpers import (
            _conversation_bot_asked_for_pincode,
            _conversation_in_pincode_delivery_flow,
        )

        if _conversation_in_pincode_delivery_flow(conversation_context):
            hints.append(
                "Recent chat was about Welfog DELIVERY SERVICEABILITY / pincode check."
            )
        if _conversation_bot_asked_for_pincode(conversation_context):
            hints.append("Bot recently asked the user for a 6-digit delivery PIN.")
    except ImportError:
        pass
    return hints


def ai_understand_delivery_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> Optional[DeliveryTurnUnderstanding]:
    """
    Micro-classifier: delivery/serviceability intent + location (any language).
    Uses RECENT CHAT when the latest message is short or refers back.
    Cached per request — safe to call from routing and live API paths.
    """
    comb = _combined(original_msg, msg_en)
    if not comb:
        return None

    try:
        from services.chat_flow_telemetry import should_skip_micro_classifier_llm

        if should_skip_micro_classifier_llm():
            return DeliveryTurnUnderstanding(source="deferred_brain")
    except ImportError:
        pass

    cache_key = _delivery_turn_cache_key(original_msg, msg_en, conversation_context)
    cached = _delivery_turn_cache_get(cache_key)
    if cached is not None:
        return cached

    from services.ai_service import (
        _compact_conversation_context,
        _llm_json_with_provider_fallback,
        _llm_classifier_provider_chain,
        _trim_text_mid,
    )
    from services.translation_service import language_reply_instruction, resolve_customer_reply_lang

    providers = _llm_classifier_provider_chain()
    if not providers:
        return None

    rl = resolve_customer_reply_lang(original_msg or msg_en, reply_lang)
    compact_ctx = _compact_conversation_context(conversation_context or "", 2200)
    user_line = _trim_text_mid(comb, 560)
    route_hint = ""
    if isinstance(ai_route, dict):
        route_hint = (
            f"Main router intent={ai_route.get('intent') or '-'} "
            f"meaning={(ai_route.get('user_meaning') or '')[:140]}"
        )
    ctx_hints = _pincode_thread_context_hints(conversation_context)
    hint_block = "\n".join(ctx_hints) if ctx_hints else ""

    system_prompt = f"""You understand Welfog CUSTOMER support messages about DELIVERY SERVICEABILITY:
whether Welfog can deliver / provide service to a place (PIN, city, locality, area).

This is NOT tracking an existing order shipment, NOT refund, NOT product search.

Use the LATEST user message AND RECENT CONVERSATION when:
- The latest message is short or refers back ("aur jagatpura me?", "wahan?", "same place", "302023")
- User continues a prior delivery-check thread with another area name
- Language is Hinglish, Hindi, Tamil, English, or mixed — infer MEANING, never match fixed phrase lists

Return ONLY valid JSON:
{{
  "user_meaning": "one English sentence — what they want THIS turn",
  "is_delivery_serviceability": true or false,
  "is_area_followup": true or false,
  "location_kind": "none" | "pincode" | "city" | "ask_pin",
  "pincode": "",
  "city_name": "",
  "state_hint": "",
  "confidence": 0.0 to 1.0
}}

is_delivery_serviceability=true: live check if Welfog serves that delivery address/area.
is_area_followup=true: new place/area continuing a delivery check already discussed in chat.
location_kind:
- pincode: clear 6-digit Indian PIN (may be alone after bot asked)
- city: ANY named Indian city/town/locality/area (Mumbai, Indore, Jagatpura, Kochi, etc.) — we geocode to PIN
- ask_pin: ONLY when place is too vague/random OR user did not name any place (half question) — then ask 6-digit PIN
- none: not a delivery serviceability turn

Use location_kind=city for all recognizable Indian place names (any state).
Use location_kind=ask_pin ONLY if no place can be inferred from message + recent chat.

is_delivery_serviceability=false: order tracking, refund, invoice, wishlist, order history,
product catalog, company FAQ, seller, payment issue, shipping policy duration (general rules).

{language_reply_instruction(rl)}"""

    user_payload = f"ROUTER HINT: {route_hint or '-'}"
    if hint_block:
        user_payload += f"\nTHREAD HINT:\n{hint_block}"
    user_payload += f"\n\nLATEST USER MESSAGE:\n{user_line}"
    if compact_ctx:
        user_payload = f"RECENT CONVERSATION:\n{compact_ctx}\n\n{user_payload}"

    try:
        data = _llm_json_with_provider_fallback(
            providers,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            max_tokens=240,
            timeout_sec=16,
            max_attempts=2,
            temperature=0.15,
        )
    except Exception as exc:
        log_reasoning(f"Delivery-turn LLM error (non-fatal): {exc}")
        return None
    if not data:
        return None

    kind = (data.get("location_kind") or "none").strip().lower()
    if kind not in ("pincode", "city", "ask_pin", "none"):
        kind = "none"
    pin = re.sub(r"\D", "", str(data.get("pincode") or ""))
    if not (len(pin) == 6 and pin[0] != "0"):
        pin = ""

    out = DeliveryTurnUnderstanding(
        is_serviceability=bool(data.get("is_delivery_serviceability")),
        is_area_followup=bool(data.get("is_area_followup")),
        location_kind=kind,
        pincode=pin,
        city_name=str(data.get("city_name") or "").strip(),
        state_hint=str(data.get("state_hint") or "").strip(),
        user_meaning=str(data.get("user_meaning") or "").strip(),
        confidence=float(data.get("confidence") or 0.0),
        source="ai_classifier",
    )
    log_reasoning(
        f"Delivery-turn AI: service={out.is_serviceability} followup={out.is_area_followup} "
        f"kind={out.location_kind} city={out.city_name or '-'} pin={out.pincode or '-'} "
        f"conf={out.confidence:.2f}"
    )
    _delivery_turn_cache_put(cache_key, out)
    return out


def ai_extract_delivery_location(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
) -> Optional[dict[str, Any]]:
    """Backward-compatible location dict from unified delivery-turn classifier."""
    understood = ai_understand_delivery_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=reply_lang,
    )
    if not understood:
        return None
    return {
        "user_meaning": understood.user_meaning,
        "location_kind": understood.location_kind,
        "pincode": understood.pincode,
        "city_name": understood.city_name,
        "state_hint": understood.state_hint,
        "confidence": understood.confidence,
        "is_delivery_serviceability": understood.is_serviceability,
        "is_area_followup": understood.is_area_followup,
    }


def _city_label_from_ai_route(ai_route: dict | None) -> str:
    """Place name from brain route — no extra LLM."""
    if not isinstance(ai_route, dict):
        return ""
    for key in ("extracted_location", "extracted_city", "search_query"):
        val = str(ai_route.get(key) or "").strip()
        if not val or len(val) < 2:
            continue
        if re.search(r"\b[1-9]\d{5}\b", val):
            continue
        return val
    return ""


def _brain_route_implies_area_followup(ai_route: dict | None) -> bool:
    """
    Structural signals from brain route JSON only — no English/Hinglish phrase lists.
    Language-specific follow-up meaning comes from ai_understand_delivery_turn().
    """
    if not isinstance(ai_route, dict):
        return False
    if ai_route.get("continue_previous_topic"):
        return True
    reuse = (ai_route.get("reuse_user_value_from_chat") or "").strip().lower()
    if reuse == "pincode":
        return True
    intent = (ai_route.get("intent") or "").strip().lower()
    handler = (ai_route.get("route_handler") or "").strip().lower()
    if intent != "pincode_check" and handler != "pincode_delivery_api":
        return False
    pin = re.sub(r"\D", "", str(ai_route.get("extracted_pincode") or ""))
    if len(pin) == 6 and pin[0] != "0":
        return False
    if _city_label_from_ai_route(ai_route):
        return False
    try:
        from services.ai_route_semantics import ai_meaning_describes_delivery_serviceability

        return ai_meaning_describes_delivery_serviceability(ai_route)
    except ImportError:
        return True


def _understanding_from_ai_route(ai_route: dict | None) -> Optional[DeliveryTurnUnderstanding]:
    if not isinstance(ai_route, dict):
        return None
    handler = (ai_route.get("route_handler") or "").strip().lower()
    intent = (ai_route.get("intent") or "").strip().lower()
    try:
        from services.ai_route_semantics import ai_meaning_describes_delivery_serviceability

        meaning_ok = ai_meaning_describes_delivery_serviceability(ai_route)
    except ImportError:
        meaning_ok = False
    if handler != "pincode_delivery_api" and intent != "pincode_check" and not meaning_ok:
        return None
    pin = re.sub(r"\D", "", str(ai_route.get("extracted_pincode") or ""))
    city = _city_label_from_ai_route(ai_route)
    if len(pin) == 6 and pin[0] != "0":
        kind = "pincode"
    elif city:
        kind = "city"
    elif meaning_ok:
        kind = "ask_pin"
    else:
        kind = "none"
    area_followup = _brain_route_implies_area_followup(ai_route)
    return DeliveryTurnUnderstanding(
        is_serviceability=True,
        is_area_followup=area_followup,
        location_kind=kind,
        pincode=pin if kind == "pincode" else "",
        city_name=city if kind == "city" else "",
        user_meaning=str(ai_route.get("user_meaning") or "").strip(),
        confidence=0.88,
        source="ai_route",
    )


def _understanding_llm_down_failsafe(
    original_msg: str,
    msg_en: str,
    conversation_context: str,
    ai_route: dict | None = None,
) -> Optional[DeliveryTurnUnderstanding]:
    """Structural signals only — when all LLM providers are unavailable."""
    comb = _combined(original_msg, msg_en)
    bare = _bare_pin_submission_in_pincode_thread(
        original_msg, conversation_context, ai_route=ai_route
    )
    if bare:
        return DeliveryTurnUnderstanding(
            is_serviceability=True,
            location_kind="pincode",
            pincode=bare,
            confidence=0.9,
            source="bare_pin_thread",
        )
    if not comb:
        return None
    m = _PIN_RE.search(comb)
    if m:
        try:
            from utils.helpers import (
                _conversation_bot_asked_for_pincode,
                _conversation_in_pincode_delivery_flow,
            )

            if (
                _conversation_in_pincode_delivery_flow(conversation_context)
                or _conversation_bot_asked_for_pincode(conversation_context)
            ):
                return DeliveryTurnUnderstanding(
                    is_serviceability=True,
                    location_kind="pincode",
                    pincode=m.group(1),
                    confidence=0.75,
                    source="failsafe_pin_in_thread",
                )
        except ImportError:
            pass
    return None


def resolve_delivery_turn(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> DeliveryTurnUnderstanding:
    """
    AI-first delivery turn understanding — one classifier per turn (cached).
    Keyword lists are NOT used; LLM-down path uses structural signals only.
    """
    if allow_llm:
        try:
            from services.chat_flow_telemetry import should_skip_micro_classifier_llm

            if should_skip_micro_classifier_llm():
                allow_llm = False
        except ImportError:
            pass

    cache_key = _delivery_turn_cache_key(original_msg, msg_en, conversation_context)
    cached = _delivery_turn_cache_get(cache_key)
    if cached is not None and cached.source != "pending":
        return cached

    route_u = _understanding_from_ai_route(ai_route)
    route_needs_place_nlu = bool(
        route_u
        and not route_u.pincode
        and route_u.location_kind in ("ask_pin", "none", "city")
        and not (route_u.city_name and route_u.location_kind == "city")
    )

    if allow_llm and route_needs_place_nlu:
        try:
            from services.product_catalog_resolver import product_catalog_route_is_locked

            if product_catalog_route_is_locked(ai_route):
                if route_u:
                    _delivery_turn_cache_put(cache_key, route_u)
                    return route_u
        except ImportError:
            pass
        ai_u = ai_understand_delivery_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            reply_lang=reply_lang,
        )
        if ai_u and (
            ai_u.is_serviceability
            or ai_u.is_area_followup
            or ai_u.location_kind in ("pincode", "city", "ask_pin")
        ):
            _delivery_turn_cache_put(cache_key, ai_u)
            return ai_u

    if route_u:
        _delivery_turn_cache_put(cache_key, route_u)
        return route_u

    failsafe = _understanding_llm_down_failsafe(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )

    if allow_llm:
        try:
            from services.product_catalog_resolver import product_catalog_route_is_locked

            if product_catalog_route_is_locked(ai_route):
                empty = DeliveryTurnUnderstanding(source="product_catalog_locked")
                _delivery_turn_cache_put(cache_key, empty)
                return empty
        except ImportError:
            pass
        ai_u = ai_understand_delivery_turn(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            reply_lang=reply_lang,
        )
        if ai_u:
            _delivery_turn_cache_put(cache_key, ai_u)
            return ai_u

    if failsafe:
        _delivery_turn_cache_put(cache_key, failsafe)
        return failsafe

    empty = DeliveryTurnUnderstanding(source="none")
    _delivery_turn_cache_put(cache_key, empty)
    return empty


def _resolved_location_from_understanding(
    understood: DeliveryTurnUnderstanding,
    *,
    reply_lang: str = "",
) -> ResolvedDeliveryLocation:
    """Map classifier output → geocoded PIN target for live API."""
    empty = ResolvedDeliveryLocation(kind="none")
    if not understood:
        return empty

    conf = understood.confidence
    service_ok = understood.is_serviceability and conf >= _DELIVERY_AI_CONF_SERVICEABILITY
    followup_ok = understood.is_area_followup and conf >= _DELIVERY_AI_CONF_FOLLOWUP
    if not service_ok and not followup_ok:
        if understood.location_kind == "none":
            return empty
        if conf < _DELIVERY_AI_CONF_LOCATION:
            return empty

    kind = understood.location_kind
    if kind == "pincode" and understood.pincode:
        return ResolvedDeliveryLocation(
            kind="pincode",
            pincode=understood.pincode,
            city_label=understood.city_name,
            source=understood.source or "ai_classifier",
            confidence="high" if conf >= 0.65 else "medium",
            reasoning=understood.user_meaning[:200],
        )

    if kind == "city" and understood.city_name and conf >= _DELIVERY_AI_CONF_LOCATION:
        city = understood.city_name.strip()
        state = understood.state_hint.strip()
        geo = geocode_city_to_pincode(city, state_hint=state)
        if geo.get("pincode"):
            return ResolvedDeliveryLocation(
                kind="city_geocoded",
                pincode=str(geo["pincode"]),
                city_label=str(geo.get("city_label") or city),
                state_hint=str(geo.get("state_hint") or state),
                source=str(geo.get("source") or "geocode"),
                confidence=str(geo.get("confidence") or "medium"),
                geocode_display=str(geo.get("display_name") or ""),
                lat=geo.get("lat"),
                lng=geo.get("lng"),
                reasoning=understood.user_meaning[:200],
            )
        return ResolvedDeliveryLocation(
            kind="ask_pin",
            city_label=city,
            state_hint=state,
            source="city_unresolved",
            confidence="medium",
            reasoning=f"Could not map {city} to PIN — ask user.",
        )

    if kind == "ask_pin" and (service_ok or followup_ok):
        place = (understood.city_name or "").strip()
        if place:
            geo = geocode_city_to_pincode(place, state_hint=understood.state_hint)
            if geo.get("pincode"):
                return ResolvedDeliveryLocation(
                    kind="city_geocoded",
                    pincode=str(geo["pincode"]),
                    city_label=str(geo.get("city_label") or place),
                    state_hint=str(geo.get("state_hint") or understood.state_hint),
                    source=str(geo.get("source") or "geocode"),
                    confidence=str(geo.get("confidence") or "medium"),
                    geocode_display=str(geo.get("display_name") or ""),
                    lat=geo.get("lat"),
                    lng=geo.get("lng"),
                    reasoning=understood.user_meaning[:200],
                )
        return ResolvedDeliveryLocation(
            kind="ask_pin",
            city_label=understood.city_name,
            source=understood.source or "ai_classifier",
            confidence="medium",
            reasoning=understood.user_meaning[:200],
        )

    return empty


def _try_geocode_place_from_message(
    comb: str,
    *,
    city_hint: str = "",
    state_hint: str = "",
) -> ResolvedDeliveryLocation:
    """Last resort: extract place name from message and geocode before asking PIN."""
    empty = ResolvedDeliveryLocation(kind="none")
    if not (comb or "").strip():
        return empty

    candidates: list[tuple[str, str]] = []
    if (city_hint or "").strip():
        candidates.append((city_hint.strip(), state_hint or ""))
    extracted = extract_place_query_from_delivery_message(comb)
    if extracted:
        norm_ext = _norm_city_key(extracted)
        if not any(_norm_city_key(c[0]) == norm_ext for c in candidates):
            candidates.append((extracted, state_hint or ""))

    for place, state in candidates:
        geo = geocode_city_to_pincode(place, state_hint=state)
        if geo.get("pincode"):
            log_reasoning(
                f"Place geocode from message: {place!r} → PIN {geo['pincode']} ({geo.get('source')})"
            )
            return ResolvedDeliveryLocation(
                kind="city_geocoded",
                pincode=str(geo["pincode"]),
                city_label=str(geo.get("city_label") or place),
                state_hint=str(geo.get("state_hint") or state),
                source=str(geo.get("source") or "geocode"),
                confidence=str(geo.get("confidence") or "medium"),
                geocode_display=str(geo.get("display_name") or ""),
                lat=geo.get("lat"),
                lng=geo.get("lng"),
                reasoning=f"Geocoded place {place} from delivery question.",
            )
    return empty


def _embedded_pincode_in_delivery_thread(
    text: str,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """Explicit pincode + 6-digit in active delivery thread — not Order ID."""
    raw = (text or "").strip()
    if not raw or not re.search(r"\b(?:pincode|pin\s*code)\b", raw, re.I):
        return ""
    m = _PIN_RE.search(raw)
    if not m:
        return ""
    try:
        from utils.helpers import (
            _conversation_bot_asked_for_pincode,
            _conversation_in_pincode_delivery_flow,
        )
    except ImportError:
        return ""
    in_thread = (
        _conversation_in_pincode_delivery_flow(conversation_context)
        or _conversation_bot_asked_for_pincode(conversation_context)
    )
    if not in_thread and isinstance(ai_route, dict):
        in_thread = (
            (ai_route.get("intent") or "").strip().lower() == "pincode_check"
            or (ai_route.get("numeric_context") or "").strip().lower() == "pincode"
        )
    return m.group(1) if in_thread else ""


def _bare_pin_submission_in_pincode_thread(
    text: str,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> str:
    """6-digit alone when bot asked PIN or pincode thread — not order ID."""
    raw = (text or "").strip()
    embedded = _embedded_pincode_in_delivery_thread(text, conversation_context, ai_route=ai_route)
    if embedded:
        return embedded
    if not re.fullmatch(r"[1-9]\d{5}\s*\??", raw):
        return ""
    try:
        from utils.helpers import (
            _conversation_bot_asked_for_pincode,
            _conversation_in_pincode_delivery_flow,
        )
    except ImportError:
        return ""
    if _conversation_bot_asked_for_pincode(conversation_context):
        return raw[:6]
    if _conversation_in_pincode_delivery_flow(conversation_context):
        return raw[:6]
    if isinstance(ai_route, dict):
        if (ai_route.get("intent") or "").strip().lower() == "pincode_check":
            return raw[:6]
        if (ai_route.get("numeric_context") or "").strip().lower() == "pincode":
            return raw[:6]
    return ""


def resolve_delivery_check_target(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    reply_lang: str = "",
    *,
    allow_llm: bool = True,
) -> ResolvedDeliveryLocation:
    """
    Resolve PIN for live API — unified AI classifier + geocode.
    Structural failsafe only when LLM unavailable.
    """
    empty = ResolvedDeliveryLocation(kind="none")

    bare = _bare_pin_submission_in_pincode_thread(
        original_msg, conversation_context, ai_route=ai_route
    )
    if bare:
        return ResolvedDeliveryLocation(
            kind="pincode",
            pincode=bare,
            source="bare_pin_thread",
            confidence="high",
            reasoning="Bare PIN after bot asked / pincode thread.",
        )

    try:
        from utils.helpers import resolve_pincode_for_check

        pin = resolve_pincode_for_check(
            original_msg,
            conversation_context,
            msg_en=msg_en,
            ai_extracted=(ai_route or {}).get("extracted_pincode") if isinstance(ai_route, dict) else "",
            ai_route=ai_route,
        )
        if pin:
            return ResolvedDeliveryLocation(
                kind="pincode",
                pincode=pin,
                source="resolve_pincode_for_check",
                confidence="high",
            )
    except ImportError:
        pass

    understood = resolve_delivery_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        reply_lang=reply_lang,
        allow_llm=allow_llm,
    )
    loc = _resolved_location_from_understanding(understood, reply_lang=reply_lang)
    if loc.kind == "ask_pin":
        geo_loc = _try_geocode_place_from_message(
            _combined(original_msg, msg_en),
            city_hint=understood.city_name or loc.city_label,
            state_hint=understood.state_hint,
        )
        if geo_loc.kind == "city_geocoded":
            return geo_loc
        return loc
    if loc.kind != "none":
        return loc

    failsafe = _understanding_llm_down_failsafe(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    if failsafe:
        loc_fs = _resolved_location_from_understanding(failsafe, reply_lang=reply_lang)
        if loc_fs.kind != "none":
            return loc_fs

    comb = _combined(original_msg, msg_en)
    geo_loc = _try_geocode_place_from_message(
        comb,
        city_hint=understood.city_name,
        state_hint=understood.state_hint,
    )
    if geo_loc.kind != "none":
        return geo_loc

    if _delivery_question_incomplete(comb, understood) and (
        understood.is_serviceability or understood.is_area_followup
    ):
        log_reasoning("Delivery: incomplete question — need PIN (no place named).")
        return ResolvedDeliveryLocation(
            kind="ask_pin",
            source="incomplete_delivery_question",
            confidence="medium",
            reasoning=understood.user_meaning[:200] or "Delivery check without place.",
        )

    return empty


def turn_continues_pincode_area_check(
    text: str,
    conversation_context: str = "",
    ai_route: dict | None = None,
) -> bool:
    """
    Active pincode thread + user names another place/area to check.
    AI-first with recent chat; structural blockers for topic switches only.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    try:
        from utils.helpers import (
            _conversation_in_pincode_delivery_flow,
            _text_is_live_order_lookup_intent,
            _text_is_order_tracking_intent,
            message_mentions_wishlist_topic,
        )
    except ImportError:
        return False
    if not _conversation_in_pincode_delivery_flow(conversation_context):
        return False
    if _PIN_RE.search(raw):
        return True
    if _text_is_order_tracking_intent(raw) or _text_is_live_order_lookup_intent(raw):
        return False
    if message_mentions_wishlist_topic(raw):
        return False
    tl = f" {raw.lower()} "
    if any(
        x in tl
        for x in (
            "order history",
            "my orders",
            "wishlist",
            "track order",
            "order id",
            "refund",
        )
    ):
        return False

    understood = resolve_delivery_turn(
        raw,
        "",
        conversation_context,
        ai_route=ai_route,
        allow_llm=True,
    )
    if understood.is_area_followup and understood.confidence >= _DELIVERY_AI_CONF_FOLLOWUP:
        return True
    if (
        understood.is_serviceability
        and understood.confidence >= _DELIVERY_AI_CONF_SERVICEABILITY
        and understood.location_kind in ("city", "ask_pin", "pincode")
    ):
        return True

    if isinstance(ai_route, dict):
        try:
            from services.ai_route_semantics import ai_meaning_describes_delivery_serviceability

            if ai_meaning_describes_delivery_serviceability(ai_route):
                return True
        except ImportError:
            pass

    if _short_area_followup_in_pincode_thread(raw):
        return True

    return False


def _short_area_followup_in_pincode_thread(text: str) -> bool:
    """Short 'aur X me?' / another-area follow-up while checking delivery — structural."""
    raw = (text or "").strip()
    if not raw or len(raw.split()) > 12:
        return False
    if _PIN_RE.search(raw):
        return True
    place = extract_place_query_from_delivery_message(raw)
    if place and len(place) >= 3 and not _place_name_looks_gibberish(place):
        return True
    tl = f" {raw.lower()} "
    if re.search(r"\b(?:aur|and|also|waha|vaha|wahan|vahan)\b", tl):
        if re.search(
            r"\b(?:area|locality|ghar|jagah|ilaka|pincode|pin\s*code)\b", tl
        ):
            return True
        if re.search(
            r"\b(?:dost|friend|bhai|behan|papa|mummy|office|college|relative)\b",
            tl,
        ):
            return True
    if re.search(r"\barea\b", tl) and re.search(r"\b(?:me|mein|par|pe|per)\b", tl):
        return True
    if re.search(
        r"(?:\b(?:aur|and|also)\s+)?\w{3,15}\s*(?:me|mein|par|pe|per)\s*\??",
        raw,
        re.I,
    ):
        return bool(extract_place_query_from_delivery_message(raw)) or bool(
            _place_is_relational_or_vague_reference(
                extract_place_query_from_delivery_message(raw)
            )
        )
    return False


def pincode_delivery_route_is_locked(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
    *,
    allow_llm: bool = True,
) -> bool:
    """Delivery/serviceability route must not be overridden by order-id promotions."""
    if not route:
        return False
    handler = (route.get("route_handler") or "").strip().lower()
    intent = (route.get("intent") or "").strip().lower()
    nc = (route.get("numeric_context") or "").strip().lower()
    if handler == "pincode_delivery_api" or intent == "pincode_check" or nc == "pincode":
        return True
    try:
        from services.semantic_intent import ai_route_requests_pincode_delivery

        if ai_route_requests_pincode_delivery(route):
            return True
    except ImportError:
        pass
    if original_msg or msg_en or conversation_context:
        return turn_requests_delivery_serviceability(
            original_msg,
            msg_en,
            conversation_context,
            route,
            allow_llm=allow_llm,
        )
    return False


def promote_pincode_delivery_on_route(
    route: dict | None,
    original_msg: str = "",
    msg_en: str = "",
    conversation_context: str = "",
) -> dict:
    """Lock live pincode API when this turn is delivery/serviceability (beats KB embedding)."""
    out = dict(route or {})
    handler = (out.get("route_handler") or "").strip().lower()
    intent = (out.get("intent") or "").strip().lower()
    if handler == "pincode_delivery_api" or intent == "pincode_check":
        return out
    if not turn_requests_delivery_serviceability(
        original_msg,
        msg_en,
        conversation_context,
        out,
        allow_llm=True,
    ):
        return out
    out["intent"] = "pincode_check"
    out["data_channel"] = "live_api"
    out["needs_order_id"] = False
    out["numeric_context"] = "pincode"
    out["run_catalog_search"] = False
    out["search_query"] = ""
    out["order_lookup_kind"] = "none"
    out["route_handler"] = "pincode_delivery_api"
    out["is_welfog_related"] = True
    log_reasoning(
        "Route fix: delivery/serviceability — pincode_delivery_api (KB override blocked)."
    )
    return out


def turn_requests_delivery_serviceability(
    original_msg: str,
    msg_en: str = "",
    conversation_context: str = "",
    ai_route: dict | None = None,
    *,
    allow_llm: bool = True,
) -> bool:
    """
    Current turn is delivery/PIN serviceability (beats order-id / KB misroutes).
    AI-first micro-classifier + recent chat; keywords not used.
    """
    try:
        from services.product_catalog_resolver import product_catalog_route_is_locked

        if product_catalog_route_is_locked(ai_route):
            return False
    except ImportError:
        pass
    try:
        from services.order_details_flow import turn_blocks_delivery_for_single_order_lookup

        if turn_blocks_delivery_for_single_order_lookup(
            original_msg, msg_en, conversation_context, ai_route=ai_route
        ):
            return False
    except ImportError:
        pass
    if _bare_pin_submission_in_pincode_thread(
        original_msg, conversation_context, ai_route=ai_route
    ):
        return True

    understood = resolve_delivery_turn(
        original_msg,
        msg_en,
        conversation_context,
        ai_route=ai_route,
        allow_llm=allow_llm,
    )
    if understood.is_serviceability and understood.confidence >= _DELIVERY_AI_CONF_SERVICEABILITY:
        return True
    if understood.is_area_followup and understood.confidence >= _DELIVERY_AI_CONF_FOLLOWUP:
        return True
    if (
        understood.location_kind in ("pincode", "city", "ask_pin")
        and understood.confidence >= _DELIVERY_AI_CONF_LOCATION
        and (understood.is_serviceability or understood.is_area_followup)
    ):
        return True

    comb = _combined(original_msg, msg_en)
    try:
        from utils.helpers import _conversation_in_pincode_delivery_flow

        if _conversation_in_pincode_delivery_flow(conversation_context) and _PIN_RE.search(comb):
            return True
        if turn_continues_pincode_area_check(comb, conversation_context, ai_route=ai_route):
            return True
    except ImportError:
        pass

    if allow_llm:
        loc = resolve_delivery_check_target(
            original_msg,
            msg_en,
            conversation_context,
            ai_route=ai_route,
            allow_llm=True,
        )
        return loc.kind in ("pincode", "city_geocoded", "ask_pin")

    failsafe = _understanding_llm_down_failsafe(
        original_msg, msg_en, conversation_context, ai_route=ai_route
    )
    return bool(failsafe and failsafe.is_serviceability)

