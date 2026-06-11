"""Today's deals and category list — live API cards for chat."""
from __future__ import annotations

from services.kb_service import sysmsg
from services.translation_service import localized_sysmsg_for_customer
from services.welfog_api import (
    customer_sale_price,
    ensure_expanded_categories_map_for_ctx,
    fetch_nav_categories,
    fetch_today_deals,
    format_customer_price_display,
)
from utils.reasoning_log import log_reasoning


def build_today_deals_reply_html(original_msg: str, *, reply_lang: str = "en") -> str:
    deals = fetch_today_deals()
    items = []
    if isinstance(deals, dict):
        for key in ("data", "products", "result", "today_deal"):
            if isinstance(deals.get(key), list):
                items = deals.get(key)
                break
    elif isinstance(deals, list):
        items = deals

    if not items:
        log_reasoning("Today deals API returned no items.")
        return sysmsg("deals_unavailable") or "Today's deals are not available right now."

    image_base = "https://d1f02fefkbso7w.cloudfront.net/"
    title = (deals.get("title") if isinstance(deals, dict) else None) or sysmsg("default_deals_name")
    response_text = sysmsg("deals_title", title=title)
    response_text += "<div class='wf-product-rail'>"

    shown = 0
    for p in items:
        if shown >= 5:
            break
        if not isinstance(p, dict):
            continue
        name = p.get("name") or p.get("product_name") or sysmsg("default_deal_card_title")
        new_price = format_customer_price_display(p, sysmsg("na_price"))
        old_price = p.get("stroked_price") or p.get("old_price") or p.get("unit_price")
        if old_price and customer_sale_price(p) and float(old_price) <= float(customer_sale_price(p)):
            old_price = None
        slug = p.get("slug") or ""
        thumb = p.get("thumbnail_img") or p.get("thumbnail_image") or p.get("image") or ""
        image = (image_base + str(thumb).lstrip("/")) if thumb else ""
        link = f"https://welfog.com/product_details/{slug}" if slug else "https://welfog.com"

        response_text += "<div class='wf-product-card'>"
        if image:
            response_text += (
                f"<div style='width: 100%; height: 130px; background-color: #f9f9f9; "
                f"border-radius: 8px; overflow: hidden; margin-bottom: 10px; display: flex; "
                f"align-items: center; justify-content: center; border: 1px solid #f0f0f0;'>"
                f"<img src='{image}' alt='{name}' style='max-width: 100%; max-height: 100%; "
                f"object-fit: contain; display: block;'></div>"
            )
        else:
            response_text += (
                f"<div style='width: 100%; height: 130px; background: #f0f0f0; border-radius: 8px; "
                f"margin-bottom: 10px; display: flex; align-items: center; justify-content: center; "
                f"color: #999; font-size: 12px; border: 1px solid #e0e0e0;'>{sysmsg('no_image')}</div>"
            )

        name_short = name[:38] + "..." if len(name) > 38 else name
        response_text += (
            f"<div style='font-size: 13px; font-weight: 600; color: #333; margin-bottom: 6px; "
            f"height: 34px; overflow: hidden; line-height: 1.3; display: -webkit-box; "
            f"-webkit-line-clamp: 2; -webkit-box-orient: vertical;'>{name_short}</div>"
        )
        if old_price and str(old_price).strip() and str(old_price) != str(new_price):
            response_text += "<div style='margin-bottom: 10px; margin-top: auto;'>"
            response_text += f"<span style='font-size: 15px; font-weight: bold; color: #ff7a00;'>₹{new_price}</span> "
            response_text += (
                f"<span style='font-size: 12px; color: #888; text-decoration: line-through;'>₹{old_price}</span>"
            )
            response_text += "</div>"
        else:
            response_text += (
                f"<div style='font-size: 15px; font-weight: bold; color: #ff7a00; "
                f"margin-bottom: 12px; margin-top: auto;'>₹{new_price}</div>"
            )
        response_text += (
            f"<a href='{link}' target='_blank' rel='noopener noreferrer'>{sysmsg('view_deal')}</a>"
        )
        response_text += "</div>"
        shown += 1

    response_text += "</div>"
    response_text += (
        localized_sysmsg_for_customer("deals_carousel_footer", original_msg, reply_lang=reply_lang)
        or sysmsg("deals_carousel_footer")
        or ""
    )
    log_reasoning(f"Today deals reply built ({shown} cards).")
    return response_text


def build_categories_list_reply_html(ctx: dict, original_msg: str = "", *, reply_lang: str = "en") -> str:
    cats = fetch_nav_categories()
    if not cats:
        log_reasoning("Nav categories API unavailable.")
        return sysmsg("categories_unavailable") or "Categories are not available right now."

    items = []
    if isinstance(cats, dict):
        for key in ("data", "categories", "result"):
            if isinstance(cats.get(key), list):
                items = cats.get(key)
                break
    elif isinstance(cats, list):
        items = cats

    shown = []
    for it in items[:20]:
        if not isinstance(it, dict):
            continue
        cid = it.get("id") or it.get("category_id") or it.get("cat_id")
        name = it.get("name") or it.get("title") or it.get("category_name")
        if cid and name:
            shown.append((cid, name))

    if not shown:
        return sysmsg("categories_parse_failed") or "Could not load category list."

    ctx.setdefault("data", {})
    ensure_expanded_categories_map_for_ctx(ctx)
    response_text = sysmsg("categories_title")
    response_text += sysmsg("categories_list_wrap_start")
    for cid, name in shown:
        response_text += f"• <b>{name}</b> (id: {cid})<br>"
    response_text += sysmsg("categories_list_wrap_end") + sysmsg("categories_footer")
    log_reasoning(f"Categories list reply ({len(shown)} categories).")
    return response_text
