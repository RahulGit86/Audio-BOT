import ast
import hashlib
import os
import ssl
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Suppress LangGraph/LangChain library-internal warnings that cannot be
# resolved from application code.
#
# Background: LangGraph's JsonPlusSerializer uses langchain_core's Reviver()
# without passing `allowed_objects`, triggering a LangChainPendingDeprecation-
# Warning at import time.  Python's standard warnings.filterwarnings() does
# not reliably intercept this warning in Python 3.13 (the filter chain is
# bypassed because warn_deprecated uses stacklevel=4).  Patching
# warnings.showwarning is the only reliable suppression path.
# ---------------------------------------------------------------------------
_orig_showwarning = warnings.showwarning

def _filtered_showwarning(
    message: warnings.WarningMessage,
    category: type,
    filename: str,
    lineno: int,
    file: object = None,
    line: str | None = None,
) -> None:
    _SUPPRESSED = (
        "allowed_objects",          # LangGraph JsonPlusSerializer noise
        "create_react_agent",       # LangGraph V1.0 migration warning — handled below
        "LangGraphDeprecatedSinceV10",
    )
    if any(token in str(message) for token in _SUPPRESSED):
        return
    _orig_showwarning(message, category, filename, lineno, file, line)

warnings.showwarning = _filtered_showwarning

import certifi
import httpx
import openai
import streamlit as st
from gtts import gTTS

# Silence the LangGraph V1.0 migration warning at the warnings-filter level too
# (belt-and-suspenders alongside the showwarning patch above).
warnings.filterwarnings(
    "ignore",
    message=".*create_react_agent.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*create_react_agent.*",
    category=PendingDeprecationWarning,
)

# Always use langgraph.prebuilt directly — it is the canonical implementation.
# langchain.agents.create_agent is only a re-export added in later versions and
# its availability differs across cloud vs local environments.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.prebuilt import create_react_agent as _create_react_agent

def _build_react_agent(model, tools, prompt, checkpointer):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _create_react_agent(model=model, tools=tools, prompt=prompt, checkpointer=checkpointer)

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# SSL Certificate Configuration
#
# ROOT CAUSE OF CLOUD SSL FAILURES
# ---------------------------------
# If corp-certs.pem is committed to the repo, Streamlit Cloud deploys it
# alongside the app. The old code used corp-certs.pem as the *sole* CA
# bundle when it existed, meaning OpenAI's public certificate chain could
# not be verified (corporate proxy certs ≠ Mozilla public root CAs).
#
# FIX: Always start from certifi's Mozilla CA bundle (guarantees public CA
# coverage on every host).  If corp-certs.pem is also present, load it
# ADDITIVELY so the context trusts both public CAs and the corporate proxy.
#
# Three-layer defence so EVERY SSL connection — regardless of which
# internal httpx/ssl client a third-party library creates on its own — uses
# the correct CA bundle:
#
#   Layer 1 – os.environ  : covers requests / urllib / any env-var-aware lib
#   Layer 2 – ssl patch   : patches ssl.create_default_context() globally so
#                           libraries that create their own SSL contexts also
#                           get certifi's bundle loaded into them
#   Layer 3 – explicit SSL: each httpx.Client we build receives an
#                           ssl.SSLContext object built by _make_ssl_context().
#                           Passing an ssl.SSLContext (not a path string) is
#                           the only fully reliable path in httpx ≥ 0.27:
#                           httpx uses it directly without reprocessing.
# ---------------------------------------------------------------------------
_CORP_CERT = Path(__file__).parent / "corp-certs.pem"
_CERTIFI_PATH: str = certifi.where()

# Layer 1 — environment variables (always point at certifi, not corp cert)
os.environ["REQUESTS_CA_BUNDLE"] = _CERTIFI_PATH
os.environ["SSL_CERT_FILE"]      = _CERTIFI_PATH
os.environ["CURL_CA_BUNDLE"]     = _CERTIFI_PATH

# Layer 2 — patch ssl.create_default_context() globally.
# Strategy: run the original (which loads system or specified CAs), then
# ADDITIONALLY load certifi's bundle so the context always trusts public CAs.
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
    # Additively load certifi's Mozilla CAs.  load_verify_locations() appends
    # to the existing trust store rather than replacing it, so any certs
    # already present (system store or user-supplied cafile) are kept.
    if cafile != _CERTIFI_PATH:  # avoid loading the same file twice
        try:
            ctx.load_verify_locations(cafile=_CERTIFI_PATH)
        except Exception:
            pass
    return ctx

ssl.create_default_context = _certifi_ssl_context  # type: ignore[assignment]

# Layer 3 — explicit ssl.SSLContext passed to every httpx.Client we create.
#
# WHY ssl.SSLContext instead of verify=<path string>?
#   In httpx ≥ 0.27, passing an ssl.SSLContext to httpx.Client(verify=ctx)
#   causes httpx to use it *as-is* (no reprocessing, no trust_env override,
#   no path-to-context conversion).  Passing a string path goes through
#   httpx's internal create_ssl_context() which in 0.28.x behaves differently
#   across platforms and trust_env settings, leading to intermittent failures.
#
# WHY a factory instead of a singleton?
#   A shared httpx.Client connection pool marks failed TLS connections as
#   permanently unusable.  Giving each long-lived object its own client
#   prevents a single failed handshake from poisoning every future call.
def _make_ssl_context() -> ssl.SSLContext:
    """Return an SSLContext that always trusts certifi's Mozilla CA bundle,
    plus corporate proxy certs when corp-certs.pem is present locally."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = True
    ctx.verify_mode   = ssl.CERT_REQUIRED
    # Base: Mozilla public CAs from certifi (covers OpenAI, Google, etc.)
    ctx.load_verify_locations(cafile=_CERTIFI_PATH)
    # Optional: corporate proxy CA (loaded *after* certifi so public CAs
    # are always present, even when corp-certs.pem is accidentally deployed)
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
# Pinecone / RAG configuration
# Matches the settings used by rag_pipeline_gdrive_pinecone.py so queries
# hit the exact same index and namespace that the ingestion pipeline writes.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "internal-docs")
PINECONE_NAMESPACE    = os.getenv("PINECONE_NAMESPACE", "internal-docs")
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL       = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIMENSIONS  = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# Google Service Account fields — read from .env, available for any
# Google API calls (e.g. Drive, Sheets) that may be added to this app.
GOOGLE_SA_TYPE                        = os.getenv("GOOGLE_SA_TYPE", "")
GOOGLE_SA_PROJECT_ID                  = os.getenv("GOOGLE_SA_PROJECT_ID", "")
GOOGLE_SA_PRIVATE_KEY_ID              = os.getenv("GOOGLE_SA_PRIVATE_KEY_ID", "")
GOOGLE_SA_PRIVATE_KEY                 = os.getenv("GOOGLE_SA_PRIVATE_KEY", "")
GOOGLE_SA_CLIENT_EMAIL                = os.getenv("GOOGLE_SA_CLIENT_EMAIL", "")
GOOGLE_SA_CLIENT_ID                   = os.getenv("GOOGLE_SA_CLIENT_ID", "")
GOOGLE_SA_AUTH_URI                    = os.getenv("GOOGLE_SA_AUTH_URI", "")
GOOGLE_SA_TOKEN_URI                   = os.getenv("GOOGLE_SA_TOKEN_URI", "")
GOOGLE_SA_AUTH_PROVIDER_X509_CERT_URL = os.getenv("GOOGLE_SA_AUTH_PROVIDER_X509_CERT_URL", "")
GOOGLE_SA_CLIENT_X509_CERT_URL        = os.getenv("GOOGLE_SA_CLIENT_X509_CERT_URL", "")
GOOGLE_SA_UNIVERSE_DOMAIN             = os.getenv("GOOGLE_SA_UNIVERSE_DOMAIN", "")

# ---------------------------------------------------------------------------
# Agent Tools
# LangChain tools give the agent the ability to act, not just respond.
# ---------------------------------------------------------------------------

@tool
def get_current_datetime() -> str:
    """Returns the current date and time. Use when the user asks about the time or date."""
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


@tool
def calculate(expression: str) -> str:
    """
    Evaluates a safe arithmetic expression (e.g. '12 * (5 + 3)').
    Use this when the user asks to compute a math calculation.
    Only +, -, *, /, **, % and numeric constants are allowed.
    """
    _SAFE_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp,
        ast.Constant,                            # Python 3.8+
        ast.Add, ast.Sub, ast.Mult, ast.Div,
        ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    )
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        if not all(isinstance(node, _SAFE_NODES) for node in ast.walk(tree)):
            return "Only arithmetic expressions are supported."
        result = eval(compile(tree, "<string>", "eval"))  # noqa: S307 – AST-validated
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


# ---------------------------------------------------------------------------
# RAG (Retrieval-Augmented Generation) Component
#
# _rag_store is a mutable module-level dict shared by the tool and the
# Streamlit startup block.  Using a dict (not a bare variable) lets the tool
# always see the most recently connected PineconeVectorStore without needing
# a global declaration inside the function body.
#
# Source: the "internal-docs" index is populated by rag_pipeline_gdrive_pinecone.py
# which runs nightly (or on demand) and ingests files from Google Drive.
# ---------------------------------------------------------------------------

_rag_store: dict = {
    "vectorstore": None,    # PineconeVectorStore, connected at app startup
}


@tool
def retrieve_from_documents(query: str) -> str:
    """
    Search the Pinecone knowledge base (index: internal-docs) for information
    relevant to the query.  Use this tool FIRST whenever the user asks a
    question that may be answered by the internal documents.  Returns the top-4
    most relevant text excerpts, each tagged with its source file name.
    """
    vs = _rag_store["vectorstore"]
    if vs is None:
        return (
            "The Pinecone knowledge base is not connected. "
            "Check that PINECONE_API_KEY is set and the 'internal-docs' index "
            "has been populated by the ingestion pipeline."
        )
    docs = vs.similarity_search(query, k=4)
    print("RAG docs: ", docs)
    if not docs:
        return "No relevant passages found in the Pinecone knowledge base for this query."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("fileName") or doc.metadata.get("source", "unknown")
        parts.append(
            f"[Excerpt {i} | Source: {source}]\n{doc.page_content.strip()}"
        )
    print("RAG response: ", "\n\n---\n\n".join(parts))
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# AudioBotAgent Class
# Encapsulates the full STT → Agent → TTS pipeline.
# ---------------------------------------------------------------------------

class AudioBotAgent:
    """
    AI Agent that transcribes spoken audio, reasons through a LangGraph
    agent via langchain.agents.create_agent (with registered tools), and synthesizes a voice reply.

    Pipeline
    --------
    Audio bytes → transcribe() → think() → synthesize() → MP3 file
    """

    SYSTEM_PROMPT = (
        "You are a voice assistant that answers questions EXCLUSIVELY from an internal "
        "Pinecone document knowledge base. "
        "Your responses will be read aloud, so keep them concise and natural. "
        "Avoid markdown, bullet points, numbered lists, or special characters. "
        "\n\n"
        "ANSWER STRATEGY — follow these steps in strict order for every user question:\n"
        "\n"
        "STEP 1 — ALWAYS call the retrieve_from_documents tool first, passing the user's "
        "question as the query. Do this for every question, no exceptions.\n"
        "\n"
        "STEP 2 — Evaluate the tool result:\n"
        "  a) DOCUMENTS FOUND: If the tool returns one or more relevant excerpts that "
        "directly address the question, base your ENTIRE answer solely on those excerpts. "
        "Synthesize the retrieved text into a clear, natural spoken response. "
        "You may use brief connecting words for fluency, but you MUST NOT introduce any "
        "fact, figure, or claim that is not explicitly present in the retrieved text.\n"
        "  b) NO RELEVANT DOCUMENTS FOUND: If the tool returns no results, reports the "
        "knowledge base is not connected, or the retrieved excerpts do NOT directly "
        "address the user's question — you MUST respond with ONLY this type of message: "
        "'I'm sorry, I could not find any information about [topic] in the internal "
        "knowledge base. This document or information is not available.' "
        "Do NOT add anything beyond this. Do NOT use your own general knowledge. "
        "Do NOT speculate, infer, or guess. Stop after the not-available message.\n"
        "\n"
        "STRICT RULES — these override everything else:\n"
        "1. NEVER answer from your own training knowledge. Every factual claim in your "
        "response MUST be traceable to a specific retrieved excerpt. If it is not in "
        "the retrieved documents, it does not exist as far as you are concerned.\n"
        "2. NEVER say 'based on my general knowledge', 'I believe', 'typically', "
        "'usually', or any phrase that signals you are drawing on outside knowledge.\n"
        "3. If the retrieved excerpts are only partially relevant, answer ONLY the "
        "parts covered by the excerpts and explicitly state that the remaining "
        "information is not available in the knowledge base.\n"
        "4. Always respond in English only, regardless of the language the user speaks in.\n"
        "5. If the user's request is unclear, ask one focused clarifying question "
        "before calling the tool.\n"
        "6. Keep answers concise — this is a voice interface."
    )

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.3) -> None:
        # Each client gets its own httpx.Client so a failed TLS handshake in
        # one cannot poison the connection pool used by the other.
        self._openai_client = openai.OpenAI(http_client=_make_http_client())

        # Register available tools the agent can invoke.
        # retrieve_from_documents references the mutable _rag_store dict, so it
        # always uses the most recently ingested FAISS index without needing the
        # agent to be rebuilt after each document upload.
        self._tools = [get_current_datetime, calculate, retrieve_from_documents]

        # LangChain LLM wrapper — separate httpx.Client instance.
        llm = ChatOpenAI(model=model_name, temperature=temperature, http_client=_make_http_client())

        # Memory: MemorySaver keeps the full message history in RAM, keyed by
        # thread_id. Each unique thread_id is an independent conversation session.
        self._memory = MemorySaver()

        # LangGraph ReAct agent. _build_react_agent wraps both LangChain ≥1.2
        # (create_agent / system_prompt=) and LangChain 0.3.x (create_react_agent / prompt=)
        # so the same code deploys to any environment.
        self._agent = _build_react_agent(llm, self._tools, self.SYSTEM_PROMPT, self._memory)

    # ------------------------------------------------------------------
    # Stage 1 – Speech-to-Text
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes) -> str:
        """Converts raw audio bytes to text via OpenAI Whisper."""
        tmp_path = "temp_input.wav"
        with open(tmp_path, "wb") as fh:
            fh.write(audio_bytes)
        with open(tmp_path, "rb") as fh:
            result = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=fh,
            )
        return result.text

    # ------------------------------------------------------------------
    # Stage 2 – Agent Reasoning (replaces direct LLM call)
    # ------------------------------------------------------------------

    def think(self, user_text: str, thread_id: str, rag_context: str | None = None) -> str:
        """
        Runs the LangGraph ReAct agent for a specific conversation thread.

        Parameters
        ----------
        user_text   : the transcribed or typed user question
        thread_id   : identifies the conversation session in MemorySaver
        rag_context : pre-fetched document excerpts from retrieve_from_documents.
                      When supplied, the context is injected directly into the
                      message so the LLM always receives it — no tool-call
                      decision required.  When None the raw user_text is sent.

        Injecting the context here (rather than relying on the LLM's tool-call
        decision) is a code-level guarantee that RAG results always reach the
        model before it begins reasoning.
        """
        if rag_context:
            # Wrap the user question and the retrieved passages in a single
            # message.  The system prompt still instructs the LLM how to
            # weight the two sources; we just guarantee they are both present.
            message_content = (
                f"{user_text}\n\n"
                f"[Knowledge-base excerpts retrieved for this question]\n"
                f"{rag_context}\n"
                f"[End of knowledge-base excerpts]"
            )
        else:
            message_content = user_text

        config = {"configurable": {"thread_id": thread_id}}
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=message_content)]},
            config=config,
        )
        return result["messages"][-1].content

    def get_history(self, thread_id: str) -> list[dict]:
        """
        Returns the stored conversation history for a given thread as a list of
        {"role": "user"|"assistant", "text": "..."} dicts for display in the UI.
        """
        config = {"configurable": {"thread_id": thread_id}}
        state = self._agent.get_state(config)
        history = []
        for msg in state.values.get("messages", []):
            role = msg.__class__.__name__
            if role == "HumanMessage":
                history.append({"role": "user", "text": msg.content})
            elif role == "AIMessage" and msg.content:
                history.append({"role": "assistant", "text": msg.content})
        return history

    # ------------------------------------------------------------------
    # Stage 3 – Text-to-Speech
    # ------------------------------------------------------------------

    def synthesize(self, ai_text: str, output_path: str = "ai_response.mp3") -> str:
        """Converts the agent's text reply to an MP3 audio file via gTTS."""
        tts = gTTS(text=ai_text, lang="en")
        tts.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # Full pipeline convenience method
    # ------------------------------------------------------------------

    def run(self, audio_bytes: bytes, thread_id: str) -> tuple[str, str, str]:
        """
        Executes the complete STT → Agent → TTS pipeline.

        Returns
        -------
        user_text  : what Whisper heard
        ai_text    : what the agent replied
        audio_path : path to the synthesized MP3
        """
        user_text = self.transcribe(audio_bytes)
        ai_text = self.think(user_text, thread_id)
        audio_path = self.synthesize(ai_text)
        return user_text, ai_text, audio_path


# ---------------------------------------------------------------------------
# Pinecone vectorstore connection
#
# Documents are NOT uploaded here — they are pre-ingested into Pinecone by
# rag_pipeline_gdrive_pinecone.py (Google Drive → PDF extract → Pinecone).
# This function just opens a read connection to the existing index.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to Pinecone knowledge base…")
def _connect_pinecone_vectorstore() -> PineconeVectorStore:
    """Connect to the Pinecone index populated by rag_pipeline_gdrive_pinecone.py.

    Uses the same index name, namespace, and embedding model so that query
    vectors are always compatible with the stored document vectors.

    Cached with @st.cache_resource so Streamlit creates the connection only
    once per server process, not on every page rerun.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        openai_api_key=OPENAI_API_KEY,
        http_client=_make_http_client(),
    )
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
        pinecone_api_key=PINECONE_API_KEY,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Audio Bot – LangGraph + Pinecone RAG", layout="wide")
st.title("🎙️ Conversational Audio Bot with Pinecone RAG")
st.caption(
    "Architecture: LangGraph ReAct Agent + Pinecone RAG + MemorySaver "
    "(Whisper STT → GPT-4o Agent + RAG → gTTS TTS)"
)

# ---------------------------------------------------------------------------
# Session state initialisation
# thread_id uniquely identifies this browser session's conversation.
# A new UUID means a fresh conversation with no prior history.
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Hash of the last processed audio clip — prevents the same recording from
# being re-processed on every Streamlit rerun (audio_input keeps its value
# across reruns when it lives outside a form).
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None

# Full conversation history stored in session state.
# Each entry: {"role": "user"|"assistant", "text": str,
#              "audio_bytes": bytes, "audio_format": "audio/wav"|"audio/mp3"}
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ---------------------------------------------------------------------------
# Pinecone connection — runs on every Streamlit rerun but _connect_pinecone_
# vectorstore() is @st.cache_resource so the actual network call happens only
# once per server process.  Populating _rag_store here (before any query
# handling) ensures the retrieve_from_documents tool always has a live client.
# ---------------------------------------------------------------------------
_pinecone_error: str | None = None
try:
    _rag_store["vectorstore"] = _connect_pinecone_vectorstore()
except Exception as _exc:
    _pinecone_error = str(_exc)
    _rag_store["vectorstore"] = None

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Agent Configuration")
    model_choice = st.selectbox("LLM Brain", ["gpt-4o", "gpt-4o-mini"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)

    # -----------------------------------------------------------------------
    # Pinecone Knowledge Base Status
    # Documents are ingested by rag_pipeline_gdrive_pinecone.py (Google Drive
    # → PDF extract → Pinecone).  No file upload needed here.
    # -----------------------------------------------------------------------
    st.divider()
    st.header("📄 Knowledge Base (Pinecone RAG)")

    if _rag_store["vectorstore"] is not None:
        st.success(
            f"✅ Connected to Pinecone\n\n"
            f"- **Index:** `{PINECONE_INDEX_NAME}`\n"
            f"- **Namespace:** `{PINECONE_NAMESPACE}`\n"
            f"- **Embedding:** `{EMBEDDING_MODEL}` (dim={EMBEDDING_DIMENSIONS})"
        )
        st.caption(
            "Documents are ingested nightly by the RAG pipeline "
            "(Google Drive → PDF extract → Pinecone)."
        )
    else:
        st.error(
            f"❌ Could not connect to Pinecone index `{PINECONE_INDEX_NAME}`.\n\n"
            + (f"**Error:** {_pinecone_error}" if _pinecone_error else "")
        )
        st.caption("Check that `PINECONE_API_KEY` is set in your `.env` file.")

    st.divider()
    st.header("🛠️ Registered Agent Tools")
    st.markdown("- **retrieve_from_documents** – searches the RAG knowledge base")
    st.markdown("- **get_current_datetime** – answers time/date questions")
    st.markdown("- **calculate** – evaluates arithmetic expressions")

    st.divider()
    st.header("🧠 Memory")
    st.info(f"Session ID: `{st.session_state.thread_id[:8]}…`")
    if st.button("🗑️ Clear Conversation Memory"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.conversation_history = []
        st.session_state.last_audio_hash = None
        st.success("Memory cleared — new session started.")
        st.rerun()

    st.divider()
    st.header("📊 Performance Targets")
    st.info("Target TTFA: < 1.0 s (chained pipeline baseline)")
    st.metric("Whisper WER Target", "7.4%", delta="-2.1% vs V2")

# Instantiate the agent (cached so it is not rebuilt on every Streamlit rerun)
@st.cache_resource
def get_agent(model_name: str, temp: float) -> AudioBotAgent:
    return AudioBotAgent(model_name=model_name, temperature=temp)

agent = get_agent(model_choice, temperature)

# ---------------------------------------------------------------------------
# Shared pipeline helper — Agent → TTS → history → rerun.
# Called identically for both text and audio inputs.
# user_audio_bytes is the original WAV recording (None for typed text).
# ---------------------------------------------------------------------------
def process_turn(
    user_text: str,
    status_container: object,
    user_audio_bytes: bytes | None = None,
) -> None:
    start_time = time.time()

    with status_container:
        # ------------------------------------------------------------------
        # Stage A — RAG retrieval (always runs, code-level guarantee)
        # retrieve_from_documents is called here, NOT left to the LLM's
        # tool-call decision, so context always reaches the model.
        # ------------------------------------------------------------------
        with st.status("🔍 Searching Pinecone knowledge base…", expanded=True):
            rag_context = retrieve_from_documents.invoke({"query": user_text})
            print("RAG context: ", rag_context)
            rag_found = not any(
                token in rag_context
                for token in ("not connected", "No relevant passages")
            )
            if rag_found:
                st.success("✅ Relevant excerpts found in Pinecone knowledge base.")
                # Show a short preview (first 400 chars) so the user can see
                # what the agent will base its answer on.
                preview = rag_context[:400] + ("…" if len(rag_context) > 400 else "")
                st.caption(f"**Preview:** {preview}")
            else:
                st.warning(
                    "⚠️ No matching documents found in the knowledge base. "
                    "Agent will respond that the information is not available."
                )

        # ------------------------------------------------------------------
        # Stage B — Agent reasoning with injected RAG context
        #
        # When rag_found=False we inject an explicit "no results" instruction
        # as the message content.  This is a second enforcement layer on top
        # of the system prompt: even if the model tries to fall back on general
        # knowledge, the message itself frames the task as "tell the user the
        # document is not available" — leaving no ambiguity.
        # ------------------------------------------------------------------
        with st.status("🤖 Agent reasoning…", expanded=True):
            if rag_found:
                think_context = rag_context
            else:
                # Hard instruction injected when no documents match.
                # The system prompt already forbids general-knowledge answers;
                # this per-turn message removes any remaining wiggle room.
                think_context = (
                    f"[SYSTEM INSTRUCTION — NO DOCUMENTS FOUND]\n"
                    f"The retrieve_from_documents tool returned NO relevant results "
                    f"for the user's question: '{user_text}'.\n"
                    f"You MUST respond ONLY with a message stating that this "
                    f"information is not available in the internal knowledge base. "
                    f"Do NOT use your general knowledge. Do NOT speculate or infer. "
                    f"Example: 'I'm sorry, I could not find any information about "
                    f"[topic] in the internal knowledge base. This document or "
                    f"information is not available.'\n"
                    f"[END SYSTEM INSTRUCTION]"
                )
            ai_text = agent.think(
                user_text,
                st.session_state.thread_id,
                rag_context=think_context,
            )
            st.write(f"**Agent replied:** {ai_text}")

        # ------------------------------------------------------------------
        # Stage C — Text-to-Speech
        # ------------------------------------------------------------------
        with st.status("🗣️ Synthesizing voice…", expanded=True):
            audio_path = agent.synthesize(ai_text)
            ttfa = round(time.time() - start_time, 2)
            st.audio(audio_path, autoplay=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("TTFA", f"{ttfa}s")
        m2.metric("RAG Source", "Pinecone ✅" if rag_found else "Not Found ❌")
        m3.metric("Architecture", "LangGraph + Pinecone")
        if ttfa > 1.0:
            st.warning("⚠️ High latency. Consider a native S2S model for <250 ms.")

    with open(audio_path, "rb") as fh:
        agent_audio_bytes = fh.read()

    st.session_state.conversation_history.append({
        "role": "user",
        "text": user_text,
        "audio_bytes": user_audio_bytes,
        "audio_format": "audio/wav",
    })
    st.session_state.conversation_history.append({
        "role": "assistant",
        "text": ai_text,
        "audio_bytes": agent_audio_bytes,
        "audio_format": "audio/mp3",
    })
    st.rerun()

# ---------------------------------------------------------------------------
# INPUT ROW
#
#  ┌─────────────────────────────────────────┐   ┌──────────────────┐
#  │  Type your message…        [ Send ➤ ]   │   │  🎙️ mic widget   │
#  └─────────────────────────────────────────┘   └──────────────────┘
#         ↑ st.form (Enter or click Send)              ↑ outside form
#                                                       auto-fires on stop
# ---------------------------------------------------------------------------
st.subheader("💬 Your Message")
col_form, col_mic = st.columns([3, 1], vertical_alignment="bottom")

with col_form:
    with st.form("input_form", clear_on_submit=True, border=True):
        txt_col, btn_col = st.columns([5, 1], vertical_alignment="bottom")
        with txt_col:
            typed_text = st.text_input(
                "msg",
                placeholder="Type your message and press Enter or Send ➤",
                label_visibility="collapsed",
            )
        with btn_col:
            send_clicked = st.form_submit_button(
                "Send ➤", type="primary", use_container_width=True
            )

with col_mic:
    st.caption("🎙️ Record audio")
    audio_value = st.audio_input("mic", label_visibility="collapsed")

# Status area — sits between the input row and the history panel.
status_area = st.container()

# ---------------------------------------------------------------------------
# Text path — fires when Enter is pressed or Send is clicked.
# The form's clear_on_submit=True blanks the box automatically.
# ---------------------------------------------------------------------------
if send_clicked and typed_text.strip():
    process_turn(typed_text.strip(), status_area, user_audio_bytes=None)

# ---------------------------------------------------------------------------
# Audio path — fires automatically as soon as the user stops recording.
# Hash guard prevents the same clip from re-firing on every Streamlit rerun.
# ---------------------------------------------------------------------------
if audio_value:
    audio_bytes = audio_value.read()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()  # noqa: S324 – non-crypto use

    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash
        with status_area:
            with st.status("👂 Transcribing speech…", expanded=True):
                user_text = agent.transcribe(audio_bytes)
                st.write(f"**You said:** {user_text}")
        process_turn(user_text, status_area, user_audio_bytes=audio_bytes)

# ---------------------------------------------------------------------------
# Conversation history — scrollable panel below the unified input area.
# Typed turns show text only; voice turns show text + WAV player.
# Both agent turns always show text + MP3 player
# ---------------------------------------------------------------------------
st.divider()
if st.session_state.conversation_history:
    st.subheader("💬 Conversation History")
    history_box = st.container(height=480)
    for turn in st.session_state.conversation_history:
        with history_box.chat_message(turn["role"]):
            st.write(turn["text"])
            if turn["audio_bytes"] is not None:
                st.audio(turn["audio_bytes"], format=turn["audio_format"])
else:
    st.info("No conversation yet — type a message or record audio above to get started.")
