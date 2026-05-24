#!/usr/bin/env python3
"""RAG Ingestion Pipeline — Google Drive → Pinecone
=====================================================

Python equivalent of the n8n workflow "Internal Data Ingestion - RAG - Internal Docs".

Pipeline Steps (mirrors n8n nodes 1-for-1)
------------------------------------------
  n8n Node                              Python Equivalent
  ─────────────────────────────────────────────────────────────────────────────
  1. Ingest Data at Midnight            APScheduler CronTrigger (00:00 UTC/local)
  2. Search for files in Google Drive   Google Drive API – files.list() in folder
  3. Download file                      Google Drive API – files.get_media()
  4. Extract PDF Content                pypdf – join all pages (joinPages=true)
  5. Prepare Text for Vector Store      Build Document(page_content, metadata)
  6. Recursive Character Text Splitter  RecursiveCharacterTextSplitter (overlap=200)
  7. Embeddings OpenAI                  OpenAIEmbeddings (text-embedding-3-large, dim=1024)
  8. Pinecone Vector Store              PineconeVectorStore.from_documents (insert mode)

n8n Workflow Settings Preserved
--------------------------------
  Google Drive folder : 1tydRGOQQXgr_AxVImcG6EpvNVW5nh5Zx  (Internal Docs - 22 May)
  Pinecone index      : internal-docs
  Pinecone namespace  : internal-docs
  Embedding model     : text-embedding-3-large
  Embedding dimensions: 1024
  Chunk overlap       : 200

Google Drive Authentication
----------------------------
Option A – Service Account (recommended for automated/scheduled runs):
  1. Create a Service Account in Google Cloud Console
  2. Download the JSON key file
  3. Set GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/key.json in .env
  4. Share the Google Drive folder with the service account email

Option B – OAuth 2.0 (browser-based, one-time authorisation):
  1. Create OAuth 2.0 credentials in Google Cloud Console
  2. Download as credentials.json (or set GOOGLE_CREDENTIALS_FILE=path)
  3. On first run a browser window opens; after approval token.json is saved
  4. Subsequent runs auto-refresh the token — no browser needed

Environment Variables (.env or shell)
---------------------------------------
  OPENAI_API_KEY                  Required – OpenAI API key
  PINECONE_API_KEY                Required – Pinecone API key
  GOOGLE_SERVICE_ACCOUNT_JSON     Path to service account JSON (Option A)
  GOOGLE_CREDENTIALS_FILE         Path to OAuth credentials JSON (Option B, default: credentials.json)
  GOOGLE_TOKEN_FILE               Path to OAuth token cache (Option B, default: token.json)
  GDRIVE_FOLDER_ID                Google Drive folder ID (default: n8n workflow value)
  PINECONE_INDEX_NAME             Pinecone index name (default: internal-docs)
  PINECONE_NAMESPACE              Pinecone namespace (default: internal-docs)
  EMBEDDING_MODEL                 OpenAI embedding model (default: text-embedding-3-large)
  EMBEDDING_DIMENSIONS            Embedding vector dimensions (default: 1024)
  CHUNK_SIZE                      Text chunk size in characters (default: 1000)
  CHUNK_OVERLAP                   Text chunk overlap in characters (default: 200)
  SCHEDULER_TIMEZONE              Timezone for midnight trigger (default: UTC)
  CLEAR_NAMESPACE_BEFORE_INSERT   Clear Pinecone namespace before each run (default: true)

Usage
------
  # One-shot manual run (runs pipeline once and exits):
  python rag_pipeline_gdrive_pinecone.py --run-now

  # Start daily midnight scheduler only (blocks until Ctrl+C):
  python rag_pipeline_gdrive_pinecone.py --schedule

  # Run immediately AND keep the scheduler alive afterwards:
  python rag_pipeline_gdrive_pinecone.py --run-now --schedule

  # Programmatic use from another module:
  from rag_pipeline_gdrive_pinecone import run_pipeline
  run_pipeline()
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import signal
import ssl
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Load .env early so all os.getenv() calls below pick up the values.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# SSL Certificate Configuration  (ported from audio_bot_agent_with_rag.py)
#
# ROOT CAUSE: The corporate proxy uses its own CA certificate.  Libraries
# like tiktoken (used by OpenAIEmbeddings to download tokenizer files from
# openaipublic.blob.core.windows.net) and httpx create their own SSL
# contexts and do not automatically pick up certifi's Mozilla CA bundle,
# causing "certificate verify failed: unable to get local issuer certificate".
#
# Three-layer defence so EVERY SSL connection uses the correct CA bundle:
#
#   Layer 1 – os.environ  : covers requests / urllib / any env-var-aware lib
#   Layer 2 – ssl patch   : patches ssl.create_default_context() globally so
#                           libraries that create their own SSL contexts also
#                           get certifi's bundle appended automatically.
#   Layer 3 – explicit ctx: _make_http_client() returns an httpx.Client whose
#                           SSLContext is built by _make_ssl_context() and
#                           passed to every httpx.Client we construct — this
#                           is the only fully reliable path in httpx ≥ 0.27.
# ---------------------------------------------------------------------------
import certifi
import httpx

_CORP_CERT = Path(__file__).parent / "corp-certs.pem"
_CERTIFI_PATH: str = certifi.where()

# Layer 1 — environment variables (covers requests, urllib3, tiktoken, etc.)
os.environ["REQUESTS_CA_BUNDLE"] = _CERTIFI_PATH
os.environ["SSL_CERT_FILE"]      = _CERTIFI_PATH
os.environ["CURL_CA_BUNDLE"]     = _CERTIFI_PATH

# Layer 2 — patch ssl.create_default_context() globally so any library that
# creates its own SSLContext (e.g. tiktoken's internal urllib calls) also
# trusts certifi's Mozilla CAs.  We ADD certifi on top of whatever the
# caller requested so corporate proxy certs and public CAs are both trusted.
_orig_ssl_create_default_context = ssl.create_default_context


def _certifi_ssl_context(
    purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
    *,
    cafile: str | None = None,
    capath: str | None = None,
    cadata: str | None = None,
) -> ssl.SSLContext:
    ctx = _orig_ssl_create_default_context(
        purpose, cafile=cafile, capath=capath, cadata=cadata
    )
    if cafile != _CERTIFI_PATH:
        try:
            ctx.load_verify_locations(cafile=_CERTIFI_PATH)
        except Exception:
            pass
    return ctx


ssl.create_default_context = _certifi_ssl_context  # type: ignore[assignment]


# Layer 3 — explicit SSLContext factory used for every httpx.Client we build.
def _make_ssl_context() -> ssl.SSLContext:
    """Return an SSLContext that always trusts certifi's Mozilla CA bundle,
    plus corporate proxy certs when corp-certs.pem is present locally."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_verify_locations(cafile=_CERTIFI_PATH)
    if _CORP_CERT.exists():
        try:
            ctx.load_verify_locations(cafile=str(_CORP_CERT))
        except Exception:
            pass
    return ctx


def _make_http_client() -> httpx.Client:
    """Create a fresh httpx.Client with an explicit, certifi-backed SSL context."""
    return httpx.Client(verify=_make_ssl_context())


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rag_pipeline")

# ---------------------------------------------------------------------------
# Configuration — mirrors n8n workflow node settings exactly
# ---------------------------------------------------------------------------

# ── Google Drive ────────────────────────────────────────────────────────────
# n8n node: "Search for files in google drive"
# folderId: 1tydRGOQQXgr_AxVImcG6EpvNVW5nh5Zx  (Internal Docs - 22 May)
GDRIVE_FOLDER_ID = os.getenv(
    "GDRIVE_FOLDER_ID", "1tydRGOQQXgr_AxVImcG6EpvNVW5nh5Zx"
)
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
GOOGLE_TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")
GOOGLE_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ── Pinecone ────────────────────────────────────────────────────────────────
# n8n node: "Pinecone Vector Store" — mode: insert, index: internal-docs, ns: internal-docs
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "internal-docs")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "internal-docs")

# ── OpenAI Embeddings ───────────────────────────────────────────────────────
# n8n node: "Embeddings OpenAI" — model: text-embedding-3-large, dimensions: 1024
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# ── Text Splitter ───────────────────────────────────────────────────────────
# n8n node: "Recursive Character Text Splitter" — chunkOverlap: 200
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Scheduler ───────────────────────────────────────────────────────────────
# n8n node: "Ingest Data at Midnight" — scheduleTrigger at midnight
SCHEDULER_TIMEZONE = os.getenv("SCHEDULER_TIMEZONE", "UTC")

# ── Pipeline behaviour ──────────────────────────────────────────────────────
# When True, deletes all vectors in the namespace before re-inserting.
# Guarantees the index always reflects the current state of the Drive folder.
CLEAR_NAMESPACE_BEFORE_INSERT = (
    os.getenv("CLEAR_NAMESPACE_BEFORE_INSERT", "true").lower() == "true"
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_config() -> None:
    """Raise EnvironmentError if required API keys are missing."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    has_service_account = bool(GOOGLE_SERVICE_ACCOUNT_JSON)
    has_oauth_creds = Path(GOOGLE_CREDENTIALS_FILE).exists()
    has_oauth_token = Path(GOOGLE_TOKEN_FILE).exists()
    if not has_service_account and not has_oauth_creds and not has_oauth_token:
        missing.append(
            "Google Drive credentials: set GOOGLE_SERVICE_ACCOUNT_JSON "
            "or provide credentials.json / token.json for OAuth"
        )
    if missing:
        raise EnvironmentError(
            "Missing required configuration:\n  "
            + "\n  ".join(missing)
            + "\n\nSee module docstring for setup instructions."
        )


# ---------------------------------------------------------------------------
# Step 2 — Google Drive: Authenticate
# ---------------------------------------------------------------------------

def _get_gdrive_service():
    """Return an authorized Google Drive v3 API service client.

    Tries Service Account first (non-interactive, best for scheduled jobs),
    then falls back to OAuth 2.0 with browser-based first-time authorisation.
    """
    try:
        from googleapiclient.discovery import build  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "google-api-python-client is not installed. "
            "Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        ) from exc

    if GOOGLE_SERVICE_ACCOUNT_JSON:
        logger.info("Authenticating with Google Drive via Service Account …")
        from google.oauth2 import service_account  # noqa: PLC0415

        key_path = Path(GOOGLE_SERVICE_ACCOUNT_JSON)
        if not key_path.exists():
            raise FileNotFoundError(
                f"Service account JSON not found: {GOOGLE_SERVICE_ACCOUNT_JSON}"
            )
        creds = service_account.Credentials.from_service_account_file(
            str(key_path), scopes=GOOGLE_DRIVE_SCOPES
        )
        return build("drive", "v3", credentials=creds)

    # OAuth 2.0 fallback
    logger.info("Authenticating with Google Drive via OAuth 2.0 …")
    from google.auth.transport.requests import Request  # noqa: PLC0415
    from google.oauth2.credentials import Credentials  # noqa: PLC0415

    creds: Optional[Credentials] = None
    token_path = Path(GOOGLE_TOKEN_FILE)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(
            str(token_path), GOOGLE_DRIVE_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired OAuth token …")
            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: PLC0415

            creds_path = Path(GOOGLE_CREDENTIALS_FILE)
            if not creds_path.exists():
                raise FileNotFoundError(
                    f"OAuth credentials file not found: {GOOGLE_CREDENTIALS_FILE}\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            logger.info(
                "Opening browser for one-time Google Drive authorisation …"
            )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(creds_path), GOOGLE_DRIVE_SCOPES
            )
            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json())
        logger.info(f"OAuth token saved to {token_path}")

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Step 2 — Google Drive: List files in folder
# ---------------------------------------------------------------------------

def _list_drive_files(service, folder_id: str) -> list[dict]:
    """Return metadata for all non-trashed files in the specified Drive folder.

    Handles pagination transparently — mirrors n8n's "Search for files in
    google drive" node with filter.folderId set.
    """
    files: list[dict] = []
    page_token: Optional[str] = None
    query = f"'{folder_id}' in parents and trashed = false"

    logger.info(f"Listing files in Drive folder: {folder_id}")
    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageToken=page_token,
                pageSize=100,
            )
            .execute()
        )
        batch = response.get("files", [])
        files.extend(batch)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    logger.info(f"Found {len(files)} file(s) in folder.")
    return files


# ---------------------------------------------------------------------------
# Step 3 — Google Drive: Download file
# ---------------------------------------------------------------------------

def _download_file(service, file_id: str, file_name: str) -> bytes:
    """Download a single file from Google Drive and return its raw bytes.

    Mirrors n8n "Download file" node (operation: download, fileId: $json.id).
    Google Docs/Sheets/Slides are exported to PDF automatically since they
    cannot be downloaded in their native format via the Drive API.
    """
    from googleapiclient.errors import HttpError  # noqa: PLC0415
    from googleapiclient.http import MediaIoBaseDownload  # noqa: PLC0415

    # Google Workspace files must be exported — use PDF as the universal target.
    workspace_export_map = {
        "application/vnd.google-apps.document": "application/pdf",
        "application/vnd.google-apps.spreadsheet": "application/pdf",
        "application/vnd.google-apps.presentation": "application/pdf",
    }

    try:
        meta = service.files().get(fileId=file_id, fields="mimeType").execute()
        mime_type = meta.get("mimeType", "")

        if mime_type in workspace_export_map:
            export_mime = workspace_export_map[mime_type]
            logger.info(
                f"    Exporting Google Workspace file as PDF: {file_name}"
            )
            request = service.files().export_media(
                fileId=file_id, mimeType=export_mime
            )
        else:
            request = service.files().get_media(fileId=file_id)

        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()

    except HttpError as exc:
        raise RuntimeError(
            f"Drive API error downloading {file_name} ({file_id}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Step 4 — Extract PDF Content
# ---------------------------------------------------------------------------

def _extract_text(raw_bytes: bytes, file_name: str) -> str:
    """Extract plain text from file bytes.

    Mirrors n8n "Extract PDF Content" node:
      - operation: pdf
      - joinPages: true  → pages are joined into a single string
      - keepSource: json

    Also handles plain-text files (e.g. .txt, .md) which were not in the
    original n8n workflow but are a natural extension.
    """
    name_lower = file_name.lower()

    if name_lower.endswith(".pdf"):
        try:
            import pypdf  # noqa: PLC0415

            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            # joinPages: true — same behaviour as n8n node
            return "\n".join(pages)
        except Exception as exc:
            raise RuntimeError(
                f"pypdf could not parse {file_name}: {exc}"
            ) from exc

    # Plain-text fallback
    return raw_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Step 5 — Prepare Text for Vector Store
# ---------------------------------------------------------------------------

def _prepare_document(text: str, file_name: str, file_id: str):
    """Build a LangChain Document with the metadata fields set in the n8n
    "Prepare Text for Vector Store" node:
      - data     → page_content (the extracted text)
      - fileName → metadata["fileName"]
      - fileId   → metadata["fileId"]
    """
    from langchain_core.documents import Document  # noqa: PLC0415

    return Document(
        page_content=text,
        metadata={
            "fileName": file_name,   # n8n: $json.name
            "fileId": file_id,       # n8n: $json.id
            "source": file_name,     # used by retrieve_from_documents tool
            "ingested_at": datetime.utcnow().isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Step 6 — Recursive Character Text Splitter
# ---------------------------------------------------------------------------

def _split_documents(documents) -> list:
    """Chunk documents with RecursiveCharacterTextSplitter.

    Mirrors n8n "Recursive Character Text Splitter" node:
      - chunkOverlap: 200  (CHUNK_OVERLAP env var)
      - chunkSize uses project default (CHUNK_SIZE env var)
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# Steps 7 & 8 — Embed + Upsert into Pinecone
# ---------------------------------------------------------------------------

def _ensure_pinecone_index(pc) -> None:
    """Create the Pinecone index if it does not already exist.

    Uses cosine similarity with the same dimension as the embedding model
    (1024 for text-embedding-3-large at dimension=1024).
    """
    from pinecone import ServerlessSpec  # noqa: PLC0415

    existing = {idx.name for idx in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing:
        logger.info(
            f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist — creating …"
        )
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        logger.info(f"Index '{PINECONE_INDEX_NAME}' created.")
    else:
        logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists.")


def _clear_pinecone_namespace(pc) -> None:
    """Delete all vectors in the target namespace before re-inserting.

    This ensures the index always reflects the current state of the Drive
    folder — stale vectors from deleted/renamed files are removed.
    Mirrors n8n "Pinecone Vector Store" insert-mode behaviour where the
    node replaces the content on every workflow execution.
    """
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(
            f"Clearing namespace '{PINECONE_NAMESPACE}' in index "
            f"'{PINECONE_INDEX_NAME}' …"
        )
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        logger.info("Namespace cleared.")
    except Exception as exc:
        logger.warning(f"Could not clear namespace (may be empty): {exc}")


def _upsert_to_pinecone(chunks: list) -> None:
    """Embed chunks with OpenAI and upsert into Pinecone.

    Mirrors n8n nodes:
      - "Embeddings OpenAI": model=text-embedding-3-large, dimensions=1024
      - "Pinecone Vector Store": mode=insert, index=internal-docs,
                                 namespace=internal-docs
    """
    try:
        from pinecone import Pinecone  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pinecone is not installed. Run: pip install pinecone"
        ) from exc

    try:
        from langchain_pinecone import PineconeVectorStore  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "langchain-pinecone is not installed. Run: pip install langchain-pinecone"
        ) from exc

    from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

    # Step 7 — OpenAI Embeddings (text-embedding-3-large, dim=1024)
    # Each client gets its own httpx.Client with an explicit, certifi-backed
    # SSLContext so tiktoken's tokenizer download and the embedding API call
    # both go through the correct CA chain (fixes SSL cert verify failures
    # on networks with a corporate HTTPS proxy).
    logger.info(
        f"Initialising OpenAI embeddings "
        f"(model={EMBEDDING_MODEL}, dimensions={EMBEDDING_DIMENSIONS}) …"
    )
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        openai_api_key=OPENAI_API_KEY,
        http_client=_make_http_client(),
    )

    # Step 8 — Pinecone Vector Store (insert / upsert mode)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    _ensure_pinecone_index(pc)

    if CLEAR_NAMESPACE_BEFORE_INSERT:
        _clear_pinecone_namespace(pc)

    logger.info(
        f"Upserting {len(chunks)} chunk(s) into "
        f"'{PINECONE_INDEX_NAME}' / ns='{PINECONE_NAMESPACE}' …"
    )
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE,
    )
    logger.info("Upsert complete.")


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline() -> dict:
    """Execute the full RAG ingestion pipeline end-to-end.

    Returns a summary dict with keys:
      files_found    – number of files discovered in Drive folder
      files_indexed  – number of files successfully indexed
      chunks_indexed – total number of text chunks inserted into Pinecone
      elapsed_s      – wall-clock seconds for the entire run
      started_at     – ISO-8601 UTC timestamp of pipeline start
      errors         – list of (file_name, error_message) tuples
    """
    started_at = datetime.utcnow()
    logger.info(
        "=" * 70
        + f"\nRAG Ingestion Pipeline STARTED at {started_at.isoformat()} UTC\n"
        + "=" * 70
    )

    _validate_config()

    summary: dict = {
        "files_found": 0,
        "files_indexed": 0,
        "chunks_indexed": 0,
        "elapsed_s": 0.0,
        "started_at": started_at.isoformat(),
        "errors": [],
    }

    # ── Step 2: Authenticate + list files ───────────────────────────────────
    service = _get_gdrive_service()
    files = _list_drive_files(service, GDRIVE_FOLDER_ID)
    summary["files_found"] = len(files)

    if not files:
        logger.warning("No files found in folder — pipeline finished with no work.")
        summary["elapsed_s"] = (datetime.utcnow() - started_at).total_seconds()
        return summary

    logger.info(f"Files to process: {[f['name'] for f in files]}")

    # ── Steps 3–6: Download → Extract → Prepare → Split ────────────────────
    all_chunks: list = []

    for file_info in files:
        file_id = file_info["id"]
        file_name = file_info["name"]
        logger.info(f"\nProcessing [{file_name}] …")

        # Step 3 — Download
        try:
            raw_bytes = _download_file(service, file_id, file_name)
            logger.info(f"  Downloaded {len(raw_bytes):,} bytes.")
        except Exception as exc:
            msg = f"Download failed: {exc}"
            logger.error(f"  {msg}")
            summary["errors"].append((file_name, msg))
            continue

        # Step 4 — Extract PDF Content (joinPages: true)
        try:
            text = _extract_text(raw_bytes, file_name)
            logger.info(f"  Extracted {len(text):,} characters of text.")
        except Exception as exc:
            msg = f"Text extraction failed: {exc}"
            logger.error(f"  {msg}")
            summary["errors"].append((file_name, msg))
            continue

        if not text.strip():
            msg = "No text extracted — file may be image-only or empty."
            logger.warning(f"  {msg}")
            summary["errors"].append((file_name, msg))
            continue

        # Step 5 — Prepare Text for Vector Store
        doc = _prepare_document(text, file_name, file_id)

        # Step 6 — Recursive Character Text Splitter (chunkOverlap=200)
        chunks = _split_documents([doc])
        logger.info(f"  Split into {len(chunks)} chunk(s).")
        all_chunks.extend(chunks)
        summary["files_indexed"] += 1

    if not all_chunks:
        logger.warning("No chunks to index — all files may have been empty or failed.")
        summary["elapsed_s"] = (datetime.utcnow() - started_at).total_seconds()
        return summary

    logger.info(f"\nTotal chunks to embed and index: {len(all_chunks)}")

    # ── Steps 7 & 8: Embed + Upsert into Pinecone ───────────────────────────
    try:
        _upsert_to_pinecone(all_chunks)
        summary["chunks_indexed"] = len(all_chunks)
    except Exception as exc:
        msg = f"Pinecone upsert failed: {exc}"
        logger.error(msg)
        summary["errors"].append(("pinecone_upsert", msg))

    # ── Summary ──────────────────────────────────────────────────────────────
    summary["elapsed_s"] = round(
        (datetime.utcnow() - started_at).total_seconds(), 2
    )
    logger.info(
        "\n"
        + "=" * 70
        + f"\nRAG Ingestion Pipeline COMPLETE\n"
        + f"  Files found    : {summary['files_found']}\n"
        + f"  Files indexed  : {summary['files_indexed']}\n"
        + f"  Chunks indexed : {summary['chunks_indexed']}\n"
        + f"  Elapsed        : {summary['elapsed_s']}s\n"
        + (f"  Errors         : {len(summary['errors'])}\n" if summary["errors"] else "")
        + "=" * 70
    )
    if summary["errors"]:
        for name, err in summary["errors"]:
            logger.warning(f"  Error [{name}]: {err}")

    return summary


# ---------------------------------------------------------------------------
# Scheduler — mirrors n8n "Ingest Data at Midnight" scheduleTrigger
# ---------------------------------------------------------------------------

def _build_scheduler():
    """Build and return an APScheduler BlockingScheduler configured to fire
    run_pipeline() every day at 00:00 (midnight) in SCHEDULER_TIMEZONE.

    Misfire grace time is 3600 s so the job still runs if the machine was
    briefly offline at midnight (matching n8n's retry behaviour).
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: PLC0415
        from apscheduler.triggers.cron import CronTrigger  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "apscheduler is not installed. Run: pip install apscheduler"
        ) from exc

    scheduler = BlockingScheduler(timezone=SCHEDULER_TIMEZONE)
    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(hour=0, minute=0, timezone=SCHEDULER_TIMEZONE),
        id="rag_midnight_ingest",
        name="RAG Midnight Ingestion (mirrors n8n scheduleTrigger)",
        misfire_grace_time=3600,
        replace_existing=True,
    )
    return scheduler


def start_scheduler(run_now: bool = False) -> None:
    """Start the daily midnight scheduler.  Blocks until Ctrl+C / SIGTERM.

    Parameters
    ----------
    run_now : bool
        When True, run_pipeline() is called once immediately before the
        scheduler enters its blocking loop — so you don't have to wait
        until midnight for the first ingestion.
    """
    scheduler = _build_scheduler()

    # Graceful shutdown on SIGTERM (e.g. systemd / Docker stop)
    def _sigterm_handler(signum, frame):  # noqa: ANN001
        logger.info("SIGTERM received — shutting down scheduler …")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    if run_now:
        logger.info("Running pipeline immediately (--run-now) before scheduler starts …")
        run_pipeline()

    next_run = scheduler.get_jobs()[0].next_run_time if scheduler.get_jobs() else "N/A"
    logger.info(
        f"Scheduler started — next automatic run: {next_run} "
        f"(timezone={SCHEDULER_TIMEZONE}). Press Ctrl+C to stop."
    )
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — scheduler stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_pipeline_gdrive_pinecone",
        description=(
            "RAG Ingestion Pipeline — Google Drive → Pinecone\n"
            "Python equivalent of the n8n workflow "
            "'Internal Data Ingestion - RAG - Internal Docs'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run once immediately and exit:
  python rag_pipeline_gdrive_pinecone.py --run-now

  # Start daily midnight scheduler (blocks until Ctrl+C):
  python rag_pipeline_gdrive_pinecone.py --schedule

  # Run now, then keep the scheduler alive:
  python rag_pipeline_gdrive_pinecone.py --run-now --schedule

  # Override folder ID at runtime:
  GDRIVE_FOLDER_ID=<folder_id> python rag_pipeline_gdrive_pinecone.py --run-now
""",
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Execute the ingestion pipeline immediately.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help=(
            "Start the daily midnight scheduler. "
            "Combine with --run-now to also ingest immediately."
        ),
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.run_now and not args.schedule:
        parser.print_help()
        print(
            "\nHint: use --run-now for a one-shot manual run, "
            "or --schedule to start the nightly job."
        )
        sys.exit(0)

    if args.schedule:
        # Blocking — keeps process alive until Ctrl+C / SIGTERM
        start_scheduler(run_now=args.run_now)
    else:
        # args.run_now only — one-shot run, then exit
        result = run_pipeline()
        sys.exit(0 if not result["errors"] else 1)
