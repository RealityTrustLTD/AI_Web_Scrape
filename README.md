# Web Scraper & Entity Extraction Workflow

A Python-based web scraping, data extraction, and analysis workflow. This project uses [Flask](https://flask.palletsprojects.com/), [SQLAlchemy](https://www.sqlalchemy.org/), and [crawl4ai](https://github.com/sammwyy/crawl4ai) to:

1. **Scrape web pages** (optionally following internal links),
2. **Extract entities** (e.g., businesses or people) using both heuristic parsers and LLM-driven strategies,
3. **Refine extracted information** via local Large Language Model (LLM) API calls,
4. **Store results** in a local SQLite database,
5. **Provide a simple dashboard** to manage scraping jobs and view/edit extracted entities,
6. **Allow searching** across stored entities.

> **Note**: The script currently references an LLM endpoint via [Ollama](https://github.com/jmorganca/ollama), but you can adapt it to other local or remote LLM services if needed.

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Endpoints & Dashboard](#endpoints--dashboard)
- [License](#license)

## Features

1. **Async Web Crawler**  
   Uses `crawl4ai.AsyncWebCrawler` to fetch and render pages (including JavaScript), returning a Markdown representation and an LLM-extracted JSON object.

2. **Heuristic Extraction**  
   - Regex-based searches for phone numbers, emails, job titles, etc.  
   - Parsing of JSON-LD, microdata (`itemprop`), social media links, `tel:` or `mailto:` anchors, and more.

3. **Iterative Refinement**  
   - Candidates are combined, normalized, and refined by an LLM prompt to minimize missing or incorrect fields.  
   - Each candidate is scored for “marketability” (0.0–1.0).

4. **Flask Server & SQLite**  
   - Queued jobs are stored in a `ScrapedData` table.  
   - Extracted entities in an `EntityDossier` table.  
   - A background worker thread processes queued URLs one by one.

5. **Dashboard & Search**  
   - Web-based dashboard to enqueue new scraping jobs and track their status.  
   - Search page to query extracted entities by name or keywords.  
   - A “flagged” review page for low-confidence extractions.

## How It Works

1. **Queue a Job**  
   You submit a URL (via the dashboard or API). A new `ScrapedData` record is created, and the job is “queued.”

2. **Worker Thread**  
   The job queue (a `queue.Queue`) feeds a dedicated worker thread.  
   The worker sets the job status to “running,” fetches the page(s), and calls the crawler.

3. **Crawl & Extract**  
   `crawl4ai` opens the page in a headless browser, returning two key items:
   - A **Markdown** representation of the final rendered HTML  
   - An **LLM-based extraction** in JSON format, following a defined schema

4. **Heuristic Pipeline**  
   Additional extraction steps parse the HTML for phone numbers, emails, social links, JSON-LD, microdata, etc.

5. **Normalization & Refinement**  
   Each extracted “candidate” is normalized (cleaning phone numbers, deduplicating email, standardizing names).  
   A fingerprint (`md5(name|phone|email)`) ensures duplicates aren’t stored repeatedly.  
   A local LLM prompt “refines” each candidate to confirm consistency.

6. **Marketability Score**  
   Another prompt evaluates an entity’s “marketability” (numeric score) and assigns a “cluster” category.

7. **Storage & Human Summary**  
   The final candidate is written to the `EntityDossier` table with contact info, short description, extraction confidence, etc.  
   A separate LLM call produces a human-readable summary for display.

8. **Dashboard**  
   The user can view job status, see results, edit entity details, and search across all stored dossiers.

## Requirements

- **Python 3.7+**  
  (Uses async/await, type hints, etc.)

- **Python Packages**:
  - Flask  
  - Flask-SQLAlchemy  
  - requests  
  - beautifulsoup4  
  - aiohttp  
  - spacy  
  - sentence-transformers  
  - scikit-learn  
  - numpy  
  - crawl4ai

  Also install spaCy’s model by running:
      python -m spacy download en_core_web_sm

- **Local LLM Endpoint** (by default, Ollama on `localhost:11434`):  
  Set two environment variables (optional if defaults work):
      export OLLAMA_API_URL="http://localhost:11434/api/generate"
      export HR_OLLAMA_API_URL="http://localhost:11434/api/generate"

  Adjust as needed for your LLM setup.

## Installation

1. **Clone or Download** this repository.

2. **Create a Virtual Environment** (optional, but recommended):
    python3 -m venv venv
    source venv/bin/activate  # Mac/Linux
    # or:
    .\venv\Scripts\activate   # Windows

3. **Install Required Packages**:
    pip install flask flask-sqlalchemy requests beautifulsoup4 aiohttp spacy sentence-transformers scikit-learn numpy crawl4ai

4. **Download spaCy Model**:
    python -m spacy download en_core_web_sm

5. **(Optional) Set Environment Variables** if your LLM is on a different endpoint:
    export OLLAMA_API_URL="http://your-llm-host:port/api/generate"
    export HR_OLLAMA_API_URL="http://your-llm-host:port/api/generate"

## Usage

1. **Run the Script**:
    python test2.py

   This starts a Flask server on `http://localhost:8080/`.

2. **Open the Dashboard**:
   - Navigate to [http://localhost:8080/](http://localhost:8080/).
   - Use the form to enqueue a URL for scraping.
   - Check the list of jobs below to see queued, running, or completed jobs.

3. **Explore the Results**:
   - Once a job completes, click “View Details” to see the extracted Markdown plus all “Entity Dossiers.”
   - Each dossier shows contact info, promotional keywords, confidence, marketability, etc.
   - You can edit any dossier from the job details page.

4. **Search**:
   - Go to [http://localhost:8080/search](http://localhost:8080/search) to query by entity name or keywords.

5. **Review Flagged**:
   - Visit [http://localhost:8080/review](http://localhost:8080/review) to see entities with low confidence (< 0.6).

6. **Canceling Jobs**:
   - For a job in “queued” state, click “Cancel” to stop it before it starts. (If already running, it won’t terminate immediately.)

## Endpoints & Dashboard

| **Route**                  | **Method** | **Purpose**                                                                                                                     |
|----------------------------|------------|-------------------------------------------------------------------------------------------------------------------------------|
| `/` (index)               | GET/POST   | Main dashboard. Shows a form to enqueue a URL for scraping. Lists all jobs with statuses.                                     |
| `/job/<int:job_id>`       | GET        | Detailed view for a specific job: scraped Markdown, list of extracted `EntityDossier` records.                                |
| `/edit/<int:dossier_id>`  | GET/POST   | Form to edit an `EntityDossier` entry (e.g., fix phone/email).                                                                |
| `/review`                 | GET        | Shows a list of dossiers flagged for low extraction confidence.                                                               |
| `/search`                 | GET/POST   | Search form to query `EntityDossier` by name or description.                                                                  |
| `/api/jobs`               | GET        | Returns a JSON list of all jobs (ID, URL, status, timestamps, etc.).                                                         |
| `/api/dossiers`           | GET        | Returns a JSON array of all `EntityDossier` records in the database.                                                         |
| `/cancel_job/<int:job_id>`| POST       | Cancels a job if it’s still in the “queued” state.                                                                            |

The dashboard auto-refreshes job statuses every few seconds.

## License

No explicit license is provided in this repository. If you intend to use this code, please consult the author or repository owner for any applicable terms and conditions.
