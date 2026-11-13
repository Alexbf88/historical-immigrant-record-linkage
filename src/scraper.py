#!/usr/bin/env python3
"""
Asynchronous Web Scraper & Relational ETL Pipeline
Target: São Paulo Immigration Museum Digital Archive (Museu da Imigração de SP)
Author: Alexandre Borges
"""

import asyncio
import logging
import os
import random
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
from playwright.async_api import (
    BrowserContext,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.museudaimigracao.org.br"
SEARCH_URL = f"{BASE_URL}/acervo/registro-de-matricula"

SURNAME_VARIATIONS = [
    "Rastini",
    "Rastine",
    "Restini",
    "Pastini",
    "Pastine",
    "Pestini",
    "Rastin",
    "Restin",
    "Rastino",
    "Bastini",
    "Rastelli",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_FULL_CSV = os.path.join(DATA_DIR, "resultados_completos.csv")
OUTPUT_FILTERED_CSV = os.path.join(DATA_DIR, "resultados_filtrados.csv")

MAX_RETRIES = 3
DELAY_MIN = 1.5
DELAY_MAX = 3.5


def normalize_text(text: str) -> str:
    """Strip accents, extra whitespace, and convert to lowercase."""
    if not text or not isinstance(text, str):
        return ""
    nfkd_form = unicodedata.normalize("NFKD", text)
    without_accents = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return without_accents.strip().lower()


async def random_delay(min_s: float = DELAY_MIN, max_s: float = DELAY_MAX) -> None:
    """Rate-limiting jitter to avoid overwhelming the archive server."""
    await asyncio.sleep(random.uniform(min_s, max_s))


async def save_debug_artifacts(page: Page, context_label: str) -> None:
    """Save full-page screenshot and DOM snapshot when selectors or timeouts fail."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", context_label)
    screenshot_path = os.path.join(DEBUG_DIR, f"debug_{safe_label}_{timestamp}.png")
    html_path = os.path.join(DEBUG_DIR, f"debug_{safe_label}_{timestamp}.html")

    try:
        await page.screenshot(path=screenshot_path, full_page=True)
        html_content = await page.content()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.warning(f"Saved debug artifacts: {screenshot_path} | {html_path}")
    except Exception as exc:
        logger.error(f"Failed to capture debug artifacts ({context_label}): {exc}")


async def navigate_with_retry(
    page: Page, url: str, wait_selector: str, context_label: str
) -> bool:
    """Navigate to a URL with exponential/jittered backoff retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"[{context_label}] Navigating to {url} (Attempt {attempt}/{MAX_RETRIES})"
            )
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_selector(wait_selector, timeout=15000)
            return True
        except PlaywrightTimeoutError as exc:
            logger.warning(
                f"[{context_label}] Timeout on attempt {attempt}/{MAX_RETRIES}: {exc}"
            )
            if attempt == MAX_RETRIES:
                await save_debug_artifacts(page, context_label)
                return False
            await random_delay(2.0, 4.5)
        except Exception as exc:
            logger.warning(
                f"[{context_label}] Unexpected error on attempt {attempt}/{MAX_RETRIES}: {exc}"
            )
            if attempt == MAX_RETRIES:
                await save_debug_artifacts(page, context_label)
                return False
            await random_delay(2.0, 4.5)
    return False


def parse_onclick_detalhes(onclick_attr: str) -> Tuple[str, str, str, str]:
    """Parse (id, book, page, family) from onclick="detalhes('1052753','078','122','12490')"."""
    if not onclick_attr:
        return "", "", "", ""
    matches = re.findall(r"['\"]([^'\"]*)['\"]", onclick_attr)
    if len(matches) >= 4:
        return (
            matches[0].strip(),
            matches[1].strip(),
            matches[2].strip(),
            matches[3].strip(),
        )
    return "", "", "", ""


async def extract_details_and_family(
    context: BrowserContext,
    detalhes_url: str,
    pdf_url: str,
    search_term: str,
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    Fetch the relational details view for a record to extract exact Book Page,
    Steamship (Vapor), Destination Municipality, and all linked family members.
    """
    page = await context.new_page()
    extra_fields = {"Página": "", "Vapor/Navio": "", "Destino": ""}
    related_records: List[Dict[str, str]] = []

    try:
        await random_delay()
        ok = await navigate_with_retry(
            page, detalhes_url, "#registroInfo", f"details_{search_term}"
        )
        if not ok:
            return extra_fields, related_records

        async def get_span_text(selector: str) -> str:
            loc = page.locator(selector).first
            return (await loc.inner_text()).strip() if await loc.count() > 0 else ""

        extra_fields["Página"] = await get_span_text("#txt_pag")
        extra_fields["Vapor/Navio"] = await get_span_text("#txt_vapor")
        extra_fields["Destino"] = await get_span_text("#txt_destino")
        head_surname = await get_span_text("#txt_sobrenome")

        rows = page.locator("tbody#registrosRelacionados tr")
        row_count = await rows.count()
        for i in range(row_count):
            cells = rows.nth(i).locator("td")
            if await cells.count() >= 11:
                livro = (await cells.nth(0).inner_text()).strip()
                pagina = (await cells.nth(1).inner_text()).strip()
                familia = (await cells.nth(2).inner_text()).strip()
                chegada = (await cells.nth(3).inner_text()).strip()
                sobrenome = (await cells.nth(4).inner_text()).strip() or head_surname
                nome = (await cells.nth(5).inner_text()).strip()
                idade = (await cells.nth(6).inner_text()).strip()
                parentesco = (await cells.nth(8).inner_text()).strip()
                nacionalidade = (await cells.nth(9).inner_text()).strip()
                vapor = (await cells.nth(10).inner_text()).strip()

                related_records.append(
                    {
                        "Termo Busca": search_term,
                        "Origem Registro": "Familiar Relacionado",
                        "Livro": livro,
                        "Página": pagina,
                        "Família": familia,
                        "Nome": nome,
                        "Sobrenome": sobrenome,
                        "Parentesco": parentesco,
                        "Idade": idade,
                        "Nacionalidade": nacionalidade,
                        "Vapor/Navio": vapor or extra_fields["Vapor/Navio"],
                        "Data de Chegada": chegada,
                        "Destino": extra_fields["Destino"],
                        "URL da Imagem Digitalizada": pdf_url,
                        "URL Detalhes": detalhes_url,
                    }
                )
    except Exception as exc:
        logger.warning(f"Error extracting details from {detalhes_url}: {exc}")
        await save_debug_artifacts(page, f"error_details_{search_term}")
    finally:
        await page.close()

    return extra_fields, related_records


async def scrape_surname(
    context: BrowserContext,
    page: Page,
    surname: str,
    details_cache: Dict[str, Tuple[Dict[str, str], List[Dict[str, str]]]],
) -> List[Dict[str, str]]:
    """Query a surname variation and paginate through all results."""
    logger.info(f"--- Querying surname variation: '{surname}' ---")
    collected: List[Dict[str, str]] = []

    await random_delay()
    if not await navigate_with_retry(
        page, SEARCH_URL, "input#sobrenome", f"search_{surname}"
    ):
        return collected

    try:
        await page.locator("input#sobrenome").fill(surname)
        async with page.expect_navigation(
            wait_until="domcontentloaded", timeout=30000
        ):
            await page.locator("button.pesquisar[type='submit']").click()
        await page.wait_for_selector("div#teste", timeout=15000)
    except Exception as exc:
        logger.error(f"[{surname}] Search form submission failed: {exc}")
        await save_debug_artifacts(page, f"submit_fail_{surname}")
        return collected

    page_num = 1
    while True:
        header_loc = page.locator("div#teste h4").first
        header_text = (
            (await header_loc.inner_text()).strip()
            if await header_loc.count() > 0
            else ""
        )
        if page_num == 1:
            logger.info(f"[{surname}] Archive response: {header_text}")

        if "0 REGISTRO" in header_text.upper():
            break

        cards = page.locator("div#teste > div.row.border-bottom")
        card_count = await cards.count()
        for i in range(card_count):
            card = cards.nth(i)
            p_tags = card.locator("p")
            card_data: Dict[str, str] = {}
            for j in range(await p_tags.count()):
                txt = (await p_tags.nth(j).inner_text()).strip()
                if ":" in txt:
                    k, v = txt.split(":", 1)
                    card_data[k.strip()] = v.strip()

            pdf_loc = card.locator("a.visualizar")
            pdf_href = ""
            if await pdf_loc.count() > 0:
                href = await pdf_loc.first.get_attribute("href")
                if href:
                    pdf_href = urljoin(BASE_URL, href.strip())

            ver_mais_loc = card.locator("a[onclick*='detalhes']")
            reg_id, livro_js, pagina_js, familia_js = "", "", "", ""
            detalhes_url = ""
            if await ver_mais_loc.count() > 0:
                onclick_attr = (
                    await ver_mais_loc.first.get_attribute("onclick") or ""
                )
                reg_id, livro_js, pagina_js, familia_js = parse_onclick_detalhes(
                    onclick_attr
                )
                if reg_id:
                    detalhes_url = (
                        f"{BASE_URL}/acervo/registro-de-matricula/detalhes"
                        f"?id={reg_id}&livro={livro_js}&pagina={pagina_js}&familia={familia_js}"
                    )

            livro_val = card_data.get("Livro", livro_js)
            familia_val = card_data.get("Família", familia_js)
            pagina_val = pagina_js

            extra_fields = {"Página": pagina_val, "Vapor/Navio": "", "Destino": ""}
            related_records: List[Dict[str, str]] = []

            if detalhes_url:
                cache_key = f"{livro_val}_{pagina_val}_{familia_val}"
                if cache_key in details_cache:
                    extra_fields, related_records = details_cache[cache_key]
                else:
                    extra_fields, related_records = await extract_details_and_family(
                        context, detalhes_url, pdf_href, surname
                    )
                    details_cache[cache_key] = (extra_fields, related_records)

            record = {
                "Termo Busca": surname,
                "Origem Registro": "Busca Direta",
                "Livro": livro_val,
                "Página": extra_fields.get("Página") or pagina_val,
                "Família": familia_val,
                "Nome": card_data.get("Nome", ""),
                "Sobrenome": card_data.get("Sobrenome", surname.upper()),
                "Parentesco": card_data.get("Parentesco", ""),
                "Idade": card_data.get("Idade", ""),
                "Nacionalidade": card_data.get("Nacionalidade", ""),
                "Vapor/Navio": extra_fields.get("Vapor/Navio", ""),
                "Data de Chegada": card_data.get("Chegada", ""),
                "Destino": extra_fields.get("Destino", ""),
                "URL da Imagem Digitalizada": pdf_href,
                "URL Detalhes": detalhes_url,
            }
            collected.append(record)
            collected.extend(related_records)

        next_btn = page.locator(
            "ul.pagination li.page-item:not(.disabled) a.page-link[aria-label='Próximo']"
        )
        if await next_btn.count() > 0:
            next_href = await next_btn.first.get_attribute("href")
            if next_href:
                page_num += 1
                await random_delay()
                if not await navigate_with_retry(
                    page,
                    urljoin(BASE_URL, next_href),
                    "div#teste",
                    f"{surname}_p{page_num}",
                ):
                    break
            else:
                break
        else:
            break

    return collected


def extract_year(date_str: str) -> Optional[int]:
    if not date_str or not isinstance(date_str, str):
        return None
    m = re.search(r"\b(18\d{2}|19\d{2})\b", date_str)
    return int(m.group(1)) if m else None


def evaluate_target_criteria(row: pd.Series) -> Tuple[bool, str]:
    """Label records matching target historical window, given names, or Franco-Belgian origin."""
    reasons = []
    year = extract_year(str(row.get("Data de Chegada", "")))
    if year is not None and 1895 <= year <= 1910:
        reasons.append(f"Ano({year})")

    nome_norm = normalize_text(str(row.get("Nome", "")))
    for t_name in ["augusto", "auguste", "angelina", "angeline", "jose", "joseph"]:
        if t_name in nome_norm:
            reasons.append(f"Nome({t_name})")
            break

    nac_norm = normalize_text(str(row.get("Nacionalidade", "")))
    for t_nac in ["franc", "belg"]:
        if t_nac in nac_norm:
            reasons.append(f"Nacionalidade({row.get('Nacionalidade', '')})")
            break

    return (len(reasons) > 0, "; ".join(reasons))


async def main() -> None:
    all_records: List[Dict[str, str]] = []
    details_cache: Dict[str, Tuple[Dict[str, str], List[Dict[str, str]]]] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        for surname in SURNAME_VARIATIONS:
            records = await scrape_surname(context, page, surname, details_cache)
            all_records.extend(records)

        await browser.close()

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning("No records extracted.")
        return

    df["_priority"] = df["Origem Registro"].apply(
        lambda x: 0 if x == "Busca Direta" else 1
    )
    df = df.sort_values(by=["_priority"]).drop(columns=["_priority"])
    df = df.drop_duplicates(subset=["Livro", "Página", "Nome"], keep="first").reset_index(
        drop=True
    )

    evaluated = df.apply(evaluate_target_criteria, axis=1)
    df["match_provavel"] = [r[0] for r in evaluated]
    df["motivo_match"] = [r[1] for r in evaluated]

    df.to_csv(OUTPUT_FULL_CSV, index=False, encoding="utf-8-sig")
    df[df["match_provavel"]].to_csv(
        OUTPUT_FILTERED_CSV, index=False, encoding="utf-8-sig"
    )
    logger.info(
        f"Saved {len(df)} unique records to {OUTPUT_FULL_CSV}."
    )


if __name__ == "__main__":
    asyncio.run(main())
