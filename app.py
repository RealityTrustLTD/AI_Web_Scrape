#!/usr/bin/env python3
import os
import re
import json
import logging
import threading
import queue
import atexit
import signal
import asyncio
import requests
from datetime import datetime
from flask import Flask, request, redirect, url_for, render_template_string, jsonify
from flask_sqlalchemy import SQLAlchemy
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import hashlib
import spacy
import difflib

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------------
# Environment Variables & Model Setup
# ----------------------------
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
MODEL = "llama3.2:latest"  # Model for extraction/refinement

HR_OLLAMA_API_URL = os.environ.get("HR_OLLAMA_API_URL", "http://localhost:11434/api/generate")
HR_MODEL = "llama3.2:latest"  # Model for human-readable summary

def load_embedding_model() -> SentenceTransformer:
    logging.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'")
    return SentenceTransformer('all-MiniLM-L6-v2')

_embedding_model = load_embedding_model()

def generate_embedding(text: str) -> np.ndarray:
    logging.debug("Generating embedding for text of length %d", len(text))
    embedding = _embedding_model.encode(text, convert_to_numpy=True)
    logging.debug("Generated embedding with shape %s", embedding.shape)
    return embedding

# ----------------------------
# Global Regex Patterns & spaCy Setup for Non-LLM Extraction
# ----------------------------
PHONE_REGEX = re.compile(r'(?:(?:\+?\d{1,3})?[-.\s()]*)?(?:\d{3}[-.\s()]*)?\d{3}[-.\s()]*\d{4}')
EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
TITLE_REGEX = re.compile(
    r'\b(?:Assessor|Attorney|Clerk|Supervisor|Manager|Director|Officer|CEO|CFO|COO|CTO|Founder|President|Vice President|VP|Partner|Representative|Administrator|Coordinator|Consultant|Engineer|Architect|Analyst|Specialist|Advisor|Executive|Owner|HR|Sales|Marketing|Operations|General Manager|Regional Manager|Business Development|Account Manager|Project Manager|Operations Manager|Product Manager|IT Manager|Finance Manager|Human Resources|Customer Service|Technical Support|Procurement|Compliance|Risk|Legal|Communications|Public Relations|Strategic)\b',
    re.IGNORECASE
)
nlp = spacy.load("en_core_web_sm")

# ----------------------------
# Utility Functions for Candidate Merging and Similarity
# ----------------------------
def similar(a: str, b: str) -> float:
    """Compute a similarity ratio between two strings."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def merge_candidate(existing: dict, new_candidate: dict) -> dict:
    """Merge new candidate information into an existing candidate."""
    # Merge entity_type if empty.
    if not existing.get("entity_type") and new_candidate.get("entity_type"):
        existing["entity_type"] = new_candidate["entity_type"]
    # Merge names using fuzzy matching.
    if new_candidate.get("name") and existing.get("name"):
        if similar(existing["name"].strip().lower(), new_candidate["name"].strip().lower()) < 0.8:
            existing["name"] = existing["name"] + " / " + new_candidate["name"]
    elif new_candidate.get("name"):
        existing["name"] = new_candidate["name"]

    # Merge contact details.
    def merge_field(existing_value, new_value):
        if new_value and new_value not in existing_value:
            if existing_value:
                return existing_value + ", " + new_value
            else:
                return new_value
        return existing_value

    existing["contact_details"]["phone"] = merge_field(existing["contact_details"].get("phone", ""), new_candidate["contact_details"].get("phone", ""))
    existing["contact_details"]["email"] = merge_field(existing["contact_details"].get("email", ""), new_candidate["contact_details"].get("email", ""))
    existing["contact_details"]["address"] = merge_field(existing["contact_details"].get("address", ""), new_candidate["contact_details"].get("address", ""))

    # Merge operational description.
    if new_candidate.get("operational_description") and new_candidate["operational_description"] not in existing.get("operational_description", ""):
        existing["operational_description"] = existing.get("operational_description", "") + " " + new_candidate["operational_description"]

    # Merge digital presence (only website).
    if new_candidate.get("digital_presence", {}).get("website") and not existing.get("digital_presence", {}).get("website"):
        existing_dp = existing.get("digital_presence", {})
        existing_dp["website"] = new_candidate["digital_presence"]["website"]
        existing["digital_presence"] = existing_dp

    # Merge promotional keywords.
    existing_keywords = set(existing.get("promotional_keywords", []))
    new_keywords = set(new_candidate.get("promotional_keywords", []))
    existing["promotional_keywords"] = list(existing_keywords.union(new_keywords))

    # Average extraction confidence.
    existing["extraction_confidence"] = (existing.get("extraction_confidence", 0) + new_candidate.get("extraction_confidence", 0)) / 2.0

    return existing

# ----------------------------
# Candidate Extraction Functions
# ----------------------------
def extract_entities_from_text(text: str) -> dict:
    """Extract candidate details from text using regex and spaCy."""
    phones = PHONE_REGEX.findall(text)
    emails = EMAIL_REGEX.findall(text)
    titles = TITLE_REGEX.findall(text)
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]
    
    candidate = {
        "entity_type": "",
        "name": names[0] if names else "",
        "contact_details": {
            "phone": phones[0] if phones else "",
            "email": emails[0] if emails else "",
            "address": ""
        },
        "operational_description": text.strip()[:200],
        "digital_presence": {
            "website": "",
            "social_media": [],
            "engagement_cues": ""
        },
        "locality": "",
        "promotional_keywords": list(set(titles)),
        "extraction_confidence": 1.0
    }
    return candidate

def extract_candidates(html: str) -> list:
    """Extract candidates using plain text blocks."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    candidates = []
    for block in blocks:
        candidate = extract_entities_from_text(block)
        if candidate["name"] or candidate["contact_details"]["email"] or candidate["contact_details"]["phone"]:
            candidates.append(candidate)
    return candidates

# ----------------------------
# New Extraction Functions (Additional Methods)
# ----------------------------
def additional_plain_text_extraction(html: str) -> list:
    """Split the entire page text into blocks and extract candidates."""
    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(separator="\n")
    candidates = []
    for block in full_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        candidate = extract_entities_from_text(block)
        if candidate["name"] or candidate["contact_details"]["email"] or candidate["contact_details"]["phone"]:
            candidates.append(candidate)
    return candidates

def extract_structured_data_from_jsonld(html: str) -> list:
    """Extract candidates from JSON-LD blocks."""
    candidates = []
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                for item in data:
                    candidate = structured_data_to_candidate(item)
                    if candidate:
                        candidates.append(candidate)
            else:
                candidate = structured_data_to_candidate(data)
                if candidate:
                    candidates.append(candidate)
        except Exception as e:
            logging.debug("Error parsing JSON-LD: %s", e)
    return candidates

def structured_data_to_candidate(data: dict) -> dict:
    """Convert a JSON-LD dict to a candidate profile."""
    candidate = {
        "entity_type": data.get("@type", ""),
        "name": data.get("name", ""),
        "contact_details": {
            "phone": data.get("telephone", ""),
            "email": data.get("email", ""),
            "address": ""
        },
        "operational_description": data.get("description", "")[:200],
        "digital_presence": {
            "website": data.get("url", ""),
            "social_media": [],
            "engagement_cues": ""
        },
        "locality": "",
        "promotional_keywords": [],
        "extraction_confidence": 1.0
    }
    if "address" in data:
        if isinstance(data["address"], dict):
            addr_parts = [str(val) for key, val in data["address"].items() if val]
            candidate["contact_details"]["address"] = " ".join(addr_parts)
        else:
            candidate["contact_details"]["address"] = data["address"]
    return candidate

def enhanced_dom_extraction(html: str) -> list:
    """Extract candidates from specific DOM tags."""
    candidates = []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["p", "li", "span"]):
        text = tag.get_text(separator=" ", strip=True)
        if any(keyword in text.lower() for keyword in ["phone", "email", "contact", "tel"]):
            candidate = extract_entities_from_text(text)
            if candidate["name"] or candidate["contact_details"]["email"] or candidate["contact_details"]["phone"]:
                candidates.append(candidate)
    return candidates

def extract_microdata(html: str) -> list:
    """Extract candidates from microdata (itemprop attributes)."""
    candidates = []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(attrs={"itemprop": "telephone"}):
        candidate = {
            "entity_type": "",
            "name": "",
            "contact_details": {
                "phone": tag.get_text(strip=True),
                "email": "",
                "address": ""
            },
            "operational_description": tag.parent.get_text(separator=" ", strip=True)[:200],
            "digital_presence": {"website": "", "social_media": [], "engagement_cues": ""},
            "locality": "",
            "promotional_keywords": [],
            "extraction_confidence": 1.0
        }
        candidates.append(candidate)
    for tag in soup.find_all(attrs={"itemprop": "email"}):
        candidate = {
            "entity_type": "",
            "name": "",
            "contact_details": {
                "phone": "",
                "email": tag.get_text(strip=True),
                "address": ""
            },
            "operational_description": tag.parent.get_text(separator=" ", strip=True)[:200],
            "digital_presence": {"website": "", "social_media": [], "engagement_cues": ""},
            "locality": "",
            "promotional_keywords": [],
            "extraction_confidence": 1.0
        }
        candidates.append(candidate)
    return candidates

def extract_social_media_links(html: str) -> list:
    """Extract candidates from anchor tags with social media URLs."""
    candidates = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if any(domain in href for domain in ["linkedin.com", "twitter.com", "instagram.com", "facebook.com", "fb.com"]):
            candidate = {
                "entity_type": "",
                "name": a.get_text(strip=True),
                "contact_details": {"phone": "", "email": "", "address": ""},
                "operational_description": a.parent.get_text(separator=" ", strip=True)[:200],
                "digital_presence": {"website": "", "social_media": [href], "engagement_cues": ""},
                "locality": "",
                "promotional_keywords": [],
                "extraction_confidence": 1.0
            }
            candidates.append(candidate)
    return candidates

def extract_anchor_contact_info(html: str) -> list:
    """Extract candidates from anchor tags with 'tel:' or 'mailto:' links."""
    candidates = []
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("tel:"):
            candidate = {
                "entity_type": "",
                "name": a.get_text(strip=True),
                "contact_details": {"phone": href[4:], "email": "", "address": ""},
                "operational_description": a.parent.get_text(separator=" ", strip=True)[:200],
                "digital_presence": {"website": "", "social_media": [], "engagement_cues": ""},
                "locality": "",
                "promotional_keywords": [],
                "extraction_confidence": 1.0
            }
            candidates.append(candidate)
        elif href.startswith("mailto:"):
            candidate = {
                "entity_type": "",
                "name": a.get_text(strip=True),
                "contact_details": {"phone": "", "email": href[7:], "address": ""},
                "operational_description": a.parent.get_text(separator=" ", strip=True)[:200],
                "digital_presence": {"website": "", "social_media": [], "engagement_cues": ""},
                "locality": "",
                "promotional_keywords": [],
                "extraction_confidence": 1.0
            }
            candidates.append(candidate)
    return candidates

def normalize_candidate(candidate: dict) -> dict:
    """Normalize candidate data: trim, lower-case and standardize contact info."""
    # Normalize phone: remove non-digits.
    phone = candidate.get("contact_details", {}).get("phone", "")
    candidate["contact_details"]["phone"] = re.sub(r'\D', '', phone)
    # Normalize email to lower-case.
    email = candidate.get("contact_details", {}).get("email", "")
    candidate["contact_details"]["email"] = email.lower().strip()
    # Normalize name: trim spaces and lower-case for comparison (keep original for display).
    candidate["name"] = candidate.get("name", "").strip()
    # Normalize social media URLs.
    social_list = candidate.get("digital_presence", {}).get("social_media", [])
    normalized = []
    for url in social_list:
        if not url.startswith("http"):
            url = "https://" + url
        normalized.append(url)
    candidate["digital_presence"]["social_media"] = normalized
    return candidate

def generate_fingerprint(candidate: dict) -> str:
    """Generate a unique fingerprint based on normalized name, phone, and email."""
    norm_candidate = normalize_candidate(candidate.copy())
    name = norm_candidate.get("name", "").strip().lower()
    phone = norm_candidate.get("contact_details", {}).get("phone", "").strip().lower()
    email = norm_candidate.get("contact_details", {}).get("email", "").strip().lower()
    fingerprint_source = f"{name}|{phone}|{email}"
    return hashlib.md5(fingerprint_source.encode("utf-8")).hexdigest()

# ----------------------------
# Functions for LLM API Calls and Refinement
# ----------------------------
async def call_hr_llm(model_name: str, prompt: str, max_tokens: int = 512,
                      temperature: float = 0.3, ctx_size: int = 4096, retries: int = 3,
                      task_label: str = "HumanDescription", api_url: str = HR_OLLAMA_API_URL) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ctx_size": ctx_size,
        "stream": False
    }
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=120) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response_text = data.get("response", "").strip()
                        logging.debug(f"[{task_label} Agent] HR Response (first 8000 chars): '{response_text[:8000]}...'")
                        return response_text if response_text else "No response."
                    else:
                        logging.error(f"[{task_label} Agent] HR LLM returned status {resp.status}")
                        if 500 <= resp.status < 600:
                            raise aiohttp.ClientError(f"Server error: {resp.status}")
                        return f"[ERR] HR LLM returned status {resp.status}"
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logging.warning(f"[{task_label} Agent] HR Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                logging.error(f"[{task_label} Agent] All HR retry attempts failed.")
                return "[ERR] HR LLM request failed after retries."
        except Exception as e:
            logging.exception(f"[{task_label} Agent] Unexpected HR error: {e}")
            return "[ERR] Unexpected HR error occurred."
    return "[ERR] HR LLM request failed after final attempt."

def query_local_llm(prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    # Directly call call_llm without nesting asyncio.run()
    return call_llm(MODEL, prompt, max_tokens=max_tokens,
                    temperature=temperature, ctx_size=4096,
                    retries=3, task_label="Refinement", api_url=OLLAMA_API_URL)

def call_llm(model_name: str, prompt: str, max_tokens: int = 4096,
             temperature: float = 0.1, ctx_size: int = 8192, retries: int = 3,
             task_label: str = "General", api_url: str = OLLAMA_API_URL) -> str:
    return asyncio.run(_call_llm_async(model_name, prompt, max_tokens, temperature, ctx_size, retries, task_label, api_url))

async def _call_llm_async(model_name: str, prompt: str, max_tokens: int,
                          temperature: float, ctx_size: int, retries: int,
                          task_label: str, api_url: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ctx_size": ctx_size,
        "stream": False
    }
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload, timeout=90) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        response_text = data.get("response", "").strip()
                        logging.debug(f"[{task_label} Agent] Response (first 8000 chars): '{response_text[:8000]}...'")
                        return response_text if response_text else "No response."
                    else:
                        logging.error(f"[{task_label} Agent] LLM returned status {resp.status}")
                        if 500 <= resp.status < 600:
                            raise aiohttp.ClientError(f"Server error: {resp.status}")
                        return f"[ERR] LLM returned status {resp.status}"
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logging.warning(f"[{task_label} Agent] Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                logging.error(f"[{task_label} Agent] All retry attempts failed.")
                return "[ERR] LLM request failed after retries."
        except Exception as e:
            logging.exception(f"[{task_label} Agent] Unexpected error: {e}")
            return "[ERR] Unexpected error occurred."
    return "[ERR] LLM request failed after final attempt."

def extract_json_from_response(response: str) -> str:
    json_match = re.search(r'(\{.*\})', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        logging.debug("Extracted JSON substring: %s", json_str[:200])
        return json_str
    return response

def refine_candidate(candidate: dict, job_url: str) -> dict:
    summary_text = (
        f"Entity Type: {candidate.get('entity_type', '')}\n"
        f"Name: {candidate.get('name', '')}\n"
        f"Contact Details: {json.dumps(candidate.get('contact_details', {}))}\n"
        f"Operational Description: {candidate.get('operational_description', '')}\n"
        f"Digital Presence: {json.dumps(candidate.get('digital_presence', {}))}\n"
        f"Locality: {candidate.get('locality', '')}\n"
        f"Promotional Keywords: {', '.join(candidate.get('promotional_keywords', []))}\n"
    )
    prompt = (
        "You are an intelligent data extraction system. Your task is to validate and enrich the following candidate entity profile, "
        "using only information explicitly stated in the provided text. Do not hallucinate or invent any details. "
        "Return a JSON object with the following keys exactly: 'entity_type', 'name', 'contact_details', 'operational_description', "
        "'digital_presence', 'locality', 'promotional_keywords', and 'extraction_confidence' (a float between 0 and 1). "
        "Only output data that is directly supported by the input text. Here is the candidate profile:\n\n" +
        summary_text +
        "\nExample output: {\"entity_type\": \"Business\", \"name\": \"Example Inc.\", \"contact_details\": {\"phone\": \"123-456-7890\", \"email\": \"contact@example.com\", \"address\": \"123 Main St\"}, \"operational_description\": \"Provides example services.\", \"digital_presence\": {\"website\": \"https://example.com\"}, \"locality\": \"\", \"promotional_keywords\": [\"innovation\", \"quality\"], \"extraction_confidence\": 0.85}"
    )
    response = query_local_llm(prompt, max_tokens=512, temperature=0.0)
    if response.startswith("[ERR]"):
        logging.error("LLM response error in refine_candidate: %s", response)
        candidate['extraction_confidence'] = 0.0
        return candidate
    json_response = extract_json_from_response(response)
    try:
        refined = json.loads(json_response)
        refined['extraction_confidence'] = float(refined.get('extraction_confidence', 0.8))
        for key in ['name', 'operational_description', 'locality']:
            if key in candidate and candidate[key] and refined.get(key, "") not in candidate[key]:
                refined[key] = candidate[key]
        return refined
    except Exception as e:
        logging.error("Error refining candidate with LLM: %s", e)
        logging.error("LLM response: %s", response)
        candidate['extraction_confidence'] = 0.0
        return candidate

def adjust_locality(candidate: dict, job_url: str) -> dict:
    if not candidate.get("locality"):
        parsed = urlparse(job_url)
        parts = [part for part in parsed.path.split("/") if part and part.isalpha() and len(part) > 3]
        if parts:
            candidate["locality"] = parts[-1].replace("-", " ").title()
    return candidate

def predict_marketability(candidate: dict) -> Tuple[float, int]:
    summary_text = (
        f"Entity Type: {candidate.get('entity_type', '')}\n"
        f"Operational Description: {candidate.get('operational_description', '')}\n"
        f"Digital Presence: {json.dumps(candidate.get('digital_presence', {}))}\n"
        f"Locality: {candidate.get('locality', '')}\n"
        f"Promotional Keywords: {', '.join(candidate.get('promotional_keywords', []))}\n"
    )
    prompt = (
        "You are a marketability evaluator. Analyze the following entity profile summary and assess its market potential. "
        "Return only a JSON object with exactly two keys: 'score' (a float between 0 and 1) and 'cluster' (an integer). "
        "Do not include any extra text. Here is the profile summary:\n\n" +
        summary_text +
        "\nExample output: {\"score\": 0.85, \"cluster\": 2}"
    )
    response = query_local_llm(prompt)
    if response.startswith("[ERR]"):
        logging.error("LLM response error in predict_marketability: %s", response)
        return 0.0, 0
    json_response = extract_json_from_response(response)
    try:
        result = json.loads(json_response)
        score = float(result.get("score", 0.0))
        cluster = int(result.get("cluster", 0))
        return score, cluster
    except Exception as e:
        logging.error("Error parsing marketability LLM response: %s", e)
        return 0.0, 0

# ----------------------------
# Original Candidate Extraction and Clustering
# ----------------------------
def extract_candidates_original(html: str) -> list:
    """Original heuristic extraction using div elements."""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    divs = soup.find_all("div")
    phone_pattern = re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
    name_pattern = re.compile(r'(?:Name|Contact|CEO|Manager)[:\-]\s*([\w\s\.]+)', re.IGNORECASE)
    for div in divs:
        text = div.get_text(separator=" ", strip=True)
        phones = phone_pattern.findall(text)
        emails = email_pattern.findall(text)
        names = name_pattern.findall(text)
        if phones or emails or names:
            candidate = {
                "entity_type": "",
                "name": names[0].strip() if names else "",
                "contact_details": {
                    "phone": phones[0] if phones else "",
                    "email": emails[0] if emails else "",
                    "address": ""
                },
                "operational_description": text[:200],
                "digital_presence": {},
                "locality": "",
                "promotional_keywords": [],
                "extraction_confidence": 0.7
            }
            candidates.append(candidate)
    return candidates

def cluster_candidates(candidates: list, eps: float = 0.5, min_samples: int = 1) -> list:
    if not candidates:
        return []
    texts = [candidate.get("name", "") + " " + candidate.get("operational_description", "") for candidate in candidates]
    embeddings = _embedding_model.encode(texts)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_
    merged = {}
    for idx, label in enumerate(labels):
        if label == -1:
            label = idx
        if label not in merged:
            merged[label] = candidates[idx].copy()
        else:
            for key in ["name", "operational_description"]:
                if len(candidates[idx].get(key, "")) > len(merged[label].get(key, "")):
                    merged[label][key] = candidates[idx][key]
            for field in ["phone", "email", "address"]:
                if not merged[label]["contact_details"].get(field) and candidates[idx]["contact_details"].get(field):
                    merged[label]["contact_details"][field] = candidates[idx]["contact_details"][field]
            merged[label]["extraction_confidence"] = (merged[label]["extraction_confidence"] + candidates[idx]["extraction_confidence"]) / 2.0
    return list(merged.values())

# ----------------------------
# NEW: Iterative Extraction Workflow
# ----------------------------
def iterative_extraction_workflow(html: str, job_url: str, initial_candidates: list = None, max_iterations: int = 5) -> list:
    """
    Iteratively extract candidate blocks:
      1. Start with initial candidates.
      2. Run additional extraction methods.
      3. Normalize, deduplicate, and refine using LLM.
      4. Iterate until no new unique candidates are found.
    """
    final_candidates = initial_candidates if initial_candidates is not None else []
    previous_fingerprints = {generate_fingerprint(c) for c in final_candidates}
    iteration = 0

    while iteration < max_iterations:
        logging.info("Iteration %d of iterative extraction", iteration + 1)
        additional_candidates = []
        additional_candidates.extend(additional_plain_text_extraction(html))
        additional_candidates.extend(extract_microdata(html))
        additional_candidates.extend(extract_social_media_links(html))
        additional_candidates.extend(extract_anchor_contact_info(html))
        additional_candidates.extend(extract_structured_data_from_jsonld(html))
        additional_candidates.extend(enhanced_dom_extraction(html))

        normalized_candidates = [normalize_candidate(c) for c in additional_candidates]

        new_candidates = []
        for candidate in normalized_candidates:
            fingerprint = generate_fingerprint(candidate)
            if fingerprint not in previous_fingerprints:
                new_candidates.append(candidate)
                previous_fingerprints.add(fingerprint)

        if not new_candidates:
            logging.info("No new candidates found in iteration %d", iteration + 1)
            break

        refined_candidates = []
        for candidate in new_candidates:
            refined = refine_candidate(candidate, job_url)
            refined = adjust_locality(refined, job_url)
            score, cluster = predict_marketability(refined)
            refined["marketability_score"] = score
            refined["market_cluster"] = str(cluster)
            refined_candidates.append(refined)

        for candidate in refined_candidates:
            existing = find_existing_entity_in_list(final_candidates, candidate)
            if existing:
                merge_candidate(existing, candidate)
            else:
                final_candidates.append(candidate)

        iteration += 1

    return final_candidates

def find_existing_entity_in_list(candidate_list: list, candidate: dict) -> dict:
    """Find a candidate in a list that matches by normalized phone, email, or similar name."""
    for existing in candidate_list:
        if candidate.get("contact_details", {}).get("phone") and existing.get("contact_details", {}).get("phone"):
            if candidate["contact_details"]["phone"].strip() == existing["contact_details"]["phone"].strip():
                return existing
        if candidate.get("contact_details", {}).get("email") and existing.get("contact_details", {}).get("email"):
            if candidate["contact_details"]["email"].strip().lower() == existing["contact_details"]["email"].strip().lower():
                return existing
        if candidate.get("name") and existing.get("name"):
            if similar(candidate["name"].strip().lower(), existing["name"].strip().lower()) > 0.8:
                return existing
    return None

# ----------------------------
# Flask App & Database Setup
# ----------------------------
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scraped_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {"pool_size": 20, "max_overflow": 10}
db = SQLAlchemy(app)

class ScrapedData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String, nullable=False)
    markdown = db.Column(db.Text)
    extracted_content = db.Column(db.Text)
    status = db.Column(db.String, default="queued")
    follow_links = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    marketability_score = db.Column(db.Float, default=None)
    market_cluster = db.Column(db.String, default=None)

class EntityDossier(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    scraped_data_id = db.Column(db.Integer, db.ForeignKey('scraped_data.id'), nullable=False)
    entity_type = db.Column(db.String, default="")
    name = db.Column(db.String, default="")
    phone = db.Column(db.String, default="")
    email = db.Column(db.String, default="")
    address = db.Column(db.String, default="")
    operational_description = db.Column(db.String, default="")
    digital_presence = db.Column(db.Text, default="{}")
    locality = db.Column(db.String, default="")
    promotional_keywords = db.Column(db.Text, default="[]")
    extraction_confidence = db.Column(db.Float, default=0.0)
    marketability_score = db.Column(db.Float, default=None)
    market_cluster = db.Column(db.String, default="")
    human_description = db.Column(db.Text, default="")  # Human-readable summary
    fingerprint = db.Column(db.String, default="")  # Candidate fingerprint

with app.app_context():
    db.create_all()

job_queue = queue.Queue()

# ----------------------------
# Worker and Job Processing Functions
# ----------------------------
def process_job(job_id: int, follow_links: bool = False):
    try:
        with app.app_context():
            job = ScrapedData.query.get(job_id)
            if not job or job.status == "cancelled":
                logging.info("Job %d is not available or has been cancelled; skipping.", job_id)
                return
            logging.info("Processing job %d", job_id)
            job.status = "running"
            db.session.commit()

            if follow_links:
                logging.info("Following internal links for job %d", job_id)
                page_html = quick_fetch(job.url)
                if page_html:
                    links = extract_internal_links(page_html, job.url)
                    for link in [l for l in links if l != job.url]:
                        enqueue_job(link, follow_links=False)

            logging.info("Starting dynamic crawl for job %d on URL: %s", job_id, job.url)
            main_result = run_crawl_main(job.url)
            job.markdown = main_result.markdown
            job.extracted_content = main_result.extracted_content
            db.session.commit()

            logging.debug("Crawl markdown: %s", main_result.markdown)
            logging.debug("Crawl extracted JSON: %s", main_result.extracted_content)

            try:
                extracted_profiles = json.loads(job.extracted_content)
                if not isinstance(extracted_profiles, list):
                    extracted_profiles = [extracted_profiles]
            except Exception as e:
                logging.error("Error parsing primary extracted content for job %d: %s", job_id, e)
                extracted_profiles = []

            heuristic_candidates = extract_candidates(job.markdown)
            logging.info("Primary heuristic extraction found %d candidates", len(heuristic_candidates))
            initial_candidates = extracted_profiles + heuristic_candidates

            final_candidates = iterative_extraction_workflow(job.markdown, job.url, initial_candidates=initial_candidates)

            for candidate in final_candidates:
                fingerprint = generate_fingerprint(candidate)
                existing = EntityDossier.query.filter_by(scraped_data_id=job.id, fingerprint=fingerprint).first()
                if existing:
                    merge_candidate(existing, candidate)
                else:
                    dossier = EntityDossier(
                        scraped_data_id=job.id,
                        entity_type=candidate.get("entity_type", ""),
                        name=candidate.get("name", ""),
                        phone=candidate.get("contact_details", {}).get("phone", ""),
                        email=candidate.get("contact_details", {}).get("email", ""),
                        address=candidate.get("contact_details", {}).get("address", ""),
                        operational_description=candidate.get("operational_description", ""),
                        digital_presence=json.dumps(candidate.get("digital_presence", {})),
                        locality=candidate.get("locality", ""),
                        promotional_keywords=json.dumps(candidate.get("promotional_keywords", [])),
                        extraction_confidence=candidate.get("extraction_confidence", 0.0),
                        marketability_score=candidate.get("marketability_score"),
                        market_cluster=candidate.get("market_cluster", ""),
                        fingerprint=fingerprint
                    )
                    db.session.add(dossier)
            db.session.commit()

            for dossier in EntityDossier.query.filter_by(scraped_data_id=job.id).all():
                if not dossier.human_description:
                    hr_entity = {
                        "entity_type": dossier.entity_type,
                        "name": dossier.name,
                        "contact_details": {"phone": dossier.phone, "email": dossier.email, "address": dossier.address},
                        "operational_description": dossier.operational_description,
                        "digital_presence": json.loads(dossier.digital_presence),
                        "locality": dossier.locality,
                        "promotional_keywords": json.loads(dossier.promotional_keywords)
                    }
                    description = asyncio.run(call_hr_llm(
                        model_name=HR_MODEL,
                        prompt=(
                            "You are a summarization assistant. Your task is to generate a concise, human-readable description of the webpage content. "
                            "Use only the information provided below. Do not add any information that is not present. Summarize the key details about the page "
                            "and the entity. Output a plain text description.\n\n"
                            "Page Markdown:\n" + job.markdown + "\n\n"
                            "Entity Details:\n" + json.dumps(hr_entity, indent=2) + "\n\n"
                            "Please output a concise description."
                        ),
                        max_tokens=512,
                        temperature=0.0,
                        ctx_size=4096,
                        retries=3,
                        task_label="HumanDescription",
                        api_url=HR_OLLAMA_API_URL
                    ))
                    dossier.human_description = description
            db.session.commit()

            if final_candidates:
                first = final_candidates[0]
                job.marketability_score = first.get("marketability_score")
                job.market_cluster = first.get("market_cluster")
            job.status = "completed"
            db.session.commit()
    except Exception as e:
        with app.app_context():
            job = ScrapedData.query.get(job_id)
            if job:
                job.status = "failed"
                db.session.commit()
        logging.error("Job %d failed: %s", job_id, e)
    finally:
        with app.app_context():
            db.session.remove()

def enqueue_job(url: str, follow_links: bool = False):
    with app.app_context():
        job = ScrapedData(url=url, follow_links=follow_links, status="queued")
        db.session.add(job)
        db.session.commit()
        job_queue.put((job.id, follow_links))
        logging.info("Enqueued job %d for URL: %s", job.id, url)

def worker():
    while True:
        job_id, follow_links = job_queue.get()
        logging.info("Worker picked job %d (follow_links=%s)", job_id, follow_links)
        try:
            process_job(job_id, follow_links)
        except Exception as e:
            logging.error("Worker error on job %d: %s", job_id, e)
        finally:
            job_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

# ----------------------------
# Crawl4AI Wrapper Functions
# ----------------------------
def get_browser_config():
    return BrowserConfig(
        headless=True,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.140 Safari/537.36",
        viewport_width=1280,
        viewport_height=720,
        java_script_enabled=True
    )

def get_run_config():
    extraction_schema = {
        "entity_type": "string (Business, Government, Non-profit, Person, Other)",
        "name": "string",
        "contact_details": {
            "phone": "string",
            "email": "string",
            "address": "string"
        },
        "operational_description": "string (brief summary of services or purpose)",
        "digital_presence": {
            "website": "string",
            "social_media": "array of strings (URLs)",
            "engagement_cues": "string"
        },
        "locality": "string (city or region)",
        "promotional_keywords": "array of strings",
        "extraction_confidence": "float (0 to 1)"
    }
    extraction_instruction = (
        "Extract a complete entity profile from the webpage content using the following JSON schema exactly: " +
        f"{json.dumps(extraction_schema)}. For missing fields, use empty strings or empty arrays. Output only valid JSON."
    )
    llm_strategy = LLMExtractionStrategy(
        provider="ollama/openchat:latest",
        api_token=None,
        schema=extraction_schema,
        extraction_type="schema",
        instruction=extraction_instruction,
        chunk_token_threshold=1000,
        apply_chunking=True,
        input_format="markdown",
        extra_args={"temperature": 0.0, "max_tokens": 8192}
    )
    return CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        page_timeout=60000
    )

def run_crawl_main(url: str):
    async def crawl():
        browser_cfg = get_browser_config()
        run_cfg = get_run_config()
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await crawler.arun(url=url, config=run_cfg)
            return result
    try:
        logging.info("Starting asynchronous crawl for URL: %s", url)
        result = asyncio.run(crawl())
        logging.info("Finished asynchronous crawl for URL: %s", url)
        return result
    except Exception as e:
        logging.error("Error during asynchronous crawl for URL %s: %s", url, e)
        raise

def extract_internal_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    base_domain = urlparse(base_url).netloc
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        absolute = urljoin(base_url, href)
        if urlparse(absolute).netloc == base_domain:
            links.add(absolute)
    logging.debug("Extracted %d internal links from %s", len(links), base_url)
    return list(links)

def quick_fetch(url: str):
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            logging.debug("Fetched URL %s with response length %d", url, len(response.text))
            return response.text
        else:
            logging.error("Failed to fetch URL %s with status code %d", url, response.status_code)
    except Exception as e:
        logging.error("quick_fetch error for %s: %s", url, e)
    return None

# ----------------------------
# Search, Dashboard, and API Endpoints
# ----------------------------
@app.route('/search', methods=['GET', 'POST'])
def search_entities():
    query = ""
    results = []
    if request.method == 'POST':
        query = request.form.get('query', "").lower()
        results = EntityDossier.query.filter(
            (EntityDossier.name.ilike(f"%{query}%")) |
            (EntityDossier.operational_description.ilike(f"%{query}%"))
        ).all()
    search_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Entity Search</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
      <style>
        .entity { margin-bottom: 1rem; padding: 1rem; border: 1px solid #dee2e6; border-radius: .375rem; }
      </style>
    </head>
    <body class="container">
      <h1>Search Entities</h1>
      <form method="post">
        <div class="mb-3">
          <input type="text" name="query" class="form-control" placeholder="Enter entity name or keywords" value="{{ query }}">
        </div>
        <button type="submit" class="btn btn-primary">Search</button>
      </form>
      <hr>
      {% if results %}
        <h2>Search Results ({{ results|length }})</h2>
        {% for entity in results %}
          <div class="entity">
            <h3>{{ entity.name }} ({{ entity.entity_type }})</h3>
            <p><strong>Contact:</strong> {{ entity.phone }} | {{ entity.email }} | {{ entity.address }}</p>
            <p><strong>Description:</strong> {{ entity.operational_description }}</p>
            <p><strong>Digital Presence:</strong> {{ entity.digital_presence }}</p>
            <p><strong>Locality:</strong> {{ entity.locality }}</p>
            <p><strong>Promotional Keywords:</strong> {{ entity.promotional_keywords }}</p>
            <p><strong>Summary Description:</strong> {{ entity.human_description }}</p>
          </div>
        {% endfor %}
      {% else %}
        <p>No entities found matching your query.</p>
      {% endif %}
      <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Dashboard</a>
    </body>
    </html>
    """
    return render_template_string(search_template, query=query, results=results)

dashboard_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Live Scraped Data Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { padding-top: 2rem; }
    .job { padding: 1rem; margin-bottom: 1rem; border: 1px solid #dee2e6; border-radius: 0.375rem; }
    .job h5 { margin-top: 0; }
    .status { font-weight: bold; }
    .top-buttons { margin-bottom: 1rem; }
    .flagged { background-color: #fff3cd; }
  </style>
</head>
<body class="container">
  <div class="top-buttons d-flex justify-content-between align-items-center mb-4">
    <div>
      <a href="/api/jobs" class="btn btn-info">View API Jobs</a>
      <a href="/api/dossiers" class="btn btn-secondary">View All Dossiers</a>
      <a href="/search" class="btn btn-primary">Search Entities</a>
      <a href="/review" class="btn btn-warning">Review Flagged Entries</a>
    </div>
  </div>
  <h1 class="mb-4">Live Scraped Data Dashboard</h1>
  <form id="jobForm" method="post" action="{{ url_for('index') }}" class="mb-4">
    <div class="mb-3">
      <label for="url" class="form-label">Enter URL to scrape:</label>
      <input type="text" class="form-control" name="url" id="url" placeholder="https://example.com" required>
    </div>
    <div class="form-check mb-3">
      <input class="form-check-input" type="checkbox" name="follow_links" id="follow_links">
      <label class="form-check-label" for="follow_links">Follow internal links (1 page deep)</label>
    </div>
    <button type="submit" class="btn btn-primary">Enqueue Job</button>
  </form>
  <div id="jobList"></div>
  <script>
    async function fetchJobs() {
      const response = await fetch("{{ url_for('api_jobs') }}");
      const jobs = await response.json();
      const jobList = document.getElementById("jobList");
      jobList.innerHTML = "";
      jobs.forEach(job => {
        const jobDiv = document.createElement("div");
        jobDiv.className = "job";
        let cancelButtonHTML = "";
        if (job.status === "queued") {
          cancelButtonHTML = `<button onclick="cancelJob(${job.id})" class="btn btn-danger btn-sm">Cancel</button>`;
        }
        jobDiv.innerHTML = `
          <h5>${job.url}</h5>
          <p>Status: <span class="status">${job.status}</span></p>
          <p>Enqueued at: ${job.created_at}</p>
          <p>Marketability Score: ${job.marketability_score !== null ? job.marketability_score.toFixed(2) : "N/A"}</p>
          <p>Market Cluster: ${job.market_cluster || "N/A"}</p>
          ${ job.status === "completed" ? `<a href="/job/${job.id}" class="btn btn-success btn-sm">View Details</a>` : cancelButtonHTML }
        `;
        jobList.appendChild(jobDiv);
      });
    }
    async function cancelJob(jobId) {
      if (confirm("Are you sure you want to cancel this job?")) {
        const response = await fetch(`/cancel_job/${jobId}`, { method: "POST" });
        const result = await response.json();
        alert(result.message);
        fetchJobs();
      }
    }
    fetchJobs();
    setInterval(fetchJobs, 5000);
  </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form.get('url')
        follow_links = request.form.get('follow_links') == "on"
        if url:
            enqueue_job(url, follow_links)
            return redirect(url_for('index'))
    return render_template_string(dashboard_template)

@app.route('/job/<int:job_id>')
def job_detail(job_id: int):
    with app.app_context():
        job = ScrapedData.query.get_or_404(job_id)
        dossiers = EntityDossier.query.filter_by(scraped_data_id=job.id).all()
        detail_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>Job Detail - {{ job.id }}</title>
          <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
          <style>
            pre { white-space: pre-wrap; word-wrap: break-word; }
            .metadata-table th, .metadata-table td { vertical-align: middle; }
          </style>
        </head>
        <body class="container mt-4">
          <h1>Job Detail - {{ job.id }}</h1>
          <table class="table table-bordered metadata-table">
            <tr>
              <th>URL</th>
              <td><a href="{{ job.url }}" target="_blank">{{ job.url }}</a></td>
            </tr>
            <tr>
              <th>Status</th>
              <td>{{ job.status }}</td>
            </tr>
            <tr>
              <th>Scraped At</th>
              <td>{{ job.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</td>
            </tr>
            <tr>
              <th>Overall Marketability Score</th>
              <td>{{ job.marketability_score if job.marketability_score is not none else "N/A" }}</td>
            </tr>
            <tr>
              <th>Overall Market Cluster</th>
              <td>{{ job.market_cluster if job.market_cluster else "N/A" }}</td>
            </tr>
          </table>
          <h3>Extracted Markdown</h3>
          <div class="card mb-4">
            <div class="card-body">
              <pre>{{ job.markdown }}</pre>
            </div>
          </div>
          <h3>Entity Dossiers</h3>
          {% for dossier in dossiers %}
          <div class="card mb-3 {% if dossier.extraction_confidence < 0.6 %}flagged{% endif %}">
            <div class="card-body">
              <h5>{{ dossier.name }} ({{ dossier.entity_type }})</h5>
              <p><strong>Contact:</strong> {{ dossier.phone }} | {{ dossier.email }} | {{ dossier.address }}</p>
              <p><strong>Description:</strong> {{ dossier.operational_description }}</p>
              <p><strong>Digital Presence:</strong> {{ dossier.digital_presence }}</p>
              <p><strong>Locality:</strong> {{ dossier.locality }}</p>
              <p><strong>Promotional Keywords:</strong> {{ dossier.promotional_keywords }}</p>
              <p><strong>Extraction Confidence:</strong> {{ dossier.extraction_confidence }}</p>
              <p><strong>Marketability:</strong> {{ dossier.marketability_score }} (Cluster: {{ dossier.market_cluster }})</p>
              <p><strong>Summary Description:</strong> {{ dossier.human_description }}</p>
              <a href="/edit/{{ dossier.id }}" class="btn btn-primary btn-sm">Edit</a>
            </div>
          </div>
          {% endfor %}
          <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Dashboard</a>
        </body>
        </html>
        """
        return render_template_string(detail_template, job=job, dossiers=dossiers)

@app.route('/edit/<int:dossier_id>', methods=['GET', 'POST'])
def edit_dossier(dossier_id: int):
    with app.app_context():
        dossier = EntityDossier.query.get_or_404(dossier_id)
        if request.method == 'POST':
            dossier.name = request.form.get('name', dossier.name)
            dossier.entity_type = request.form.get('entity_type', dossier.entity_type)
            dossier.phone = request.form.get('phone', dossier.phone)
            dossier.email = request.form.get('email', dossier.email)
            dossier.address = request.form.get('address', dossier.address)
            dossier.operational_description = request.form.get('operational_description', dossier.operational_description)
            dossier.digital_presence = request.form.get('digital_presence', dossier.digital_presence)
            dossier.locality = request.form.get('locality', dossier.locality)
            dossier.promotional_keywords = request.form.get('promotional_keywords', dossier.promotional_keywords)
            db.session.commit()
            return redirect(url_for('job_detail', job_id=dossier.scraped_data_id))
        edit_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>Edit Dossier - {{ dossier.id }}</title>
          <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="container mt-4">
          <h1>Edit Dossier - {{ dossier.id }}</h1>
          <form method="post">
            <div class="mb-3">
              <label class="form-label">Name</label>
              <input type="text" name="name" class="form-control" value="{{ dossier.name }}">
            </div>
            <div class="mb-3">
              <label class="form-label">Entity Type</label>
              <input type="text" name="entity_type" class="form-control" value="{{ dossier.entity_type }}">
            </div>
            <div class="mb-3">
              <label class="form-label">Phone</label>
              <input type="text" name="phone" class="form-control" value="{{ dossier.phone }}">
            </div>
            <div class="mb-3">
              <label class="form-label">Email</label>
              <input type="text" name="email" class="form-control" value="{{ dossier.email }}">
            </div>
            <div class="mb-3">
              <label class="form-label">Address</label>
              <input type="text" name="address" class="form-control" value="{{ dossier.address }}">
            </div>
            <div class="mb-3">
              <label class="form-label">Operational Description</label>
              <textarea name="operational_description" class="form-control">{{ dossier.operational_description }}</textarea>
            </div>
            <div class="mb-3">
              <label class="form-label">Digital Presence (JSON)</label>
              <textarea name="digital_presence" class="form-control">{{ dossier.digital_presence }}</textarea>
            </div>
            <div class="mb-3">
              <label class="form-label">Locality</label>
              <input type="text" name="locality" class="form-control" value="{{ dossier.locality }}">
            </div>
            <div class="mb-3">
              <label class="form-label">Promotional Keywords (JSON array)</label>
              <textarea name="promotional_keywords" class="form-control">{{ dossier.promotional_keywords }}</textarea>
            </div>
            <button type="submit" class="btn btn-primary">Save Changes</button>
            <a href="{{ url_for('job_detail', job_id=dossier.scraped_data_id) }}" class="btn btn-secondary">Cancel</a>
          </form>
        </body>
        </html>
        """
        return render_template_string(edit_template, dossier=dossier)

@app.route('/review')
def review_flagged():
    with app.app_context():
        flagged = EntityDossier.query.filter(EntityDossier.extraction_confidence < 0.6).all()
        review_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <title>Flagged Dossiers for Review</title>
          <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="container mt-4">
          <h1>Flagged Dossiers for Review</h1>
          {% for dossier in dossiers %}
          <div class="card mb-3">
            <div class="card-body">
              <h5>{{ dossier.name }} ({{ dossier.entity_type }})</h5>
              <p><strong>Extraction Confidence:</strong> {{ dossier.extraction_confidence }}</p>
              <a href="/edit/{{ dossier.id }}" class="btn btn-primary btn-sm">Edit</a>
            </div>
          </div>
          {% endfor %}
          <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Dashboard</a>
        </body>
        </html>
        """
        return render_template_string(review_template, dossiers=flagged)

@app.route('/api/dossiers', methods=['GET'])
def api_dossiers():
    with app.app_context():
        dossiers = EntityDossier.query.all()
        dossiers_list = [{
            "id": d.id,
            "scraped_data_id": d.scraped_data_id,
            "entity_type": d.entity_type,
            "name": d.name,
            "contact_details": {
                "phone": d.phone,
                "email": d.email,
                "address": d.address
            },
            "operational_description": d.operational_description,
            "digital_presence": json.loads(d.digital_presence),
            "locality": d.locality,
            "promotional_keywords": json.loads(d.promotional_keywords),
            "extraction_confidence": d.extraction_confidence,
            "marketability_score": d.marketability_score,
            "market_cluster": d.market_cluster,
            "human_description": d.human_description
        } for d in dossiers]
        return jsonify(dossiers_list)

@app.route('/cancel_job/<int:job_id>', methods=['POST'])
def cancel_job(job_id: int):
    with app.app_context():
        job = ScrapedData.query.get_or_404(job_id)
        if job.status == "queued":
            job.status = "cancelled"
            db.session.commit()
            return jsonify({"message": f"Job {job_id} cancelled."})
        return jsonify({"message": "Job cannot be cancelled (it may be already running or completed)."}), 400

@app.route('/api/jobs', methods=['GET'])
def api_jobs():
    with app.app_context():
        try:
            jobs = ScrapedData.query.order_by(ScrapedData.created_at.desc()).all()
            jobs_list = [{
                "id": job.id,
                "url": job.url,
                "status": job.status,
                "created_at": job.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                "marketability_score": job.marketability_score,
                "market_cluster": job.market_cluster
            } for job in jobs]
            return jsonify(jobs_list)
        except Exception as e:
            logging.error("Error in /api/jobs endpoint: %s", e)
            return jsonify({"error": str(e)}), 500

def cleanup_tasks():
    with app.app_context():
        tasks = ScrapedData.query.filter(
            ScrapedData.status.in_(["queued", "cancelled", "running"])
        ).all()
        for task in tasks:
            db.session.delete(task)
        db.session.commit()
        logging.info("Cleanup complete: Removed queued, cancelled, and running tasks.")

atexit.register(cleanup_tasks)

def handle_shutdown(signum, frame):
    logging.info("Received termination signal. Cleaning up tasks.")
    cleanup_tasks()
    exit(0)

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

if __name__ == '__main__':
    logging.info("Starting Flask app on port 8080")
    app.run(port=8080, debug=False)