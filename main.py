import os
import json
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.schemas import ChatRequest, ChatResponse, SearchResponse, ChatChunk, Citation, ActionButton
from rag.index import RAGIndex
from rag.ingest import ingest_pdfs, ingest_transcripts, ingest_tainacan_json
from rag.ethics import detect_sensitive_voice_question, prepend_ethics_disclaimer
from models.provider import generate_with_ollama, format_prompt


load_dotenv()

app = FastAPI(title="Museu Virtual - Agente Cultural")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DATA_DIR = os.getenv("DATA_DIR", "./data")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
TAINACAN_JSON_PATH = os.path.join(DATA_DIR, "metadados-tainacan.json")

INDEX = RAGIndex(STORAGE_DIR)
INDEX.load()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest():
    pdf_chunks = ingest_pdfs(os.path.join(DATA_DIR, "pdfs"))
    tr_chunks = ingest_transcripts(os.path.join(DATA_DIR, "transcripts"))
    tn_chunks = ingest_tainacan_json(TAINACAN_JSON_PATH)
    all_chunks = pdf_chunks + tr_chunks + tn_chunks
    INDEX.build(all_chunks)
    return {"status": "indexed", "chunks": len(all_chunks)}


@app.post("/search", response_model=SearchResponse)
def search(req: ChatRequest):
    results = INDEX.search(req.question, top_k=req.top_k)
    chunks: List[ChatChunk] = []
    for item, _score in results:
        md = item["metadata"]
        chunks.append(ChatChunk(
            text=item["text"],
            citation=Citation(
                source_type=md.get("source_type"),
                title=md.get("title"),
                page=md.get("page"),
                url=md.get("url"),
            )
        ))
    return SearchResponse(chunks=chunks)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    sensitive = detect_sensitive_voice_question(req.question)
    results = INDEX.search(req.question, top_k=req.top_k)
    citations: List[Citation] = []
    context_chunks: List[str] = []
    action_buttons: List[ActionButton] = []

    for item, _score in results:
        md = item["metadata"]
        citations.append(Citation(
            source_type=md.get("source_type"),
            title=md.get("title"),
            page=md.get("page"),
            url=md.get("url"),
        ))
        context_chunks.append(item["text"])
        # ações básicas a partir de metadados
        if md.get("source_type") == "video":
            action_buttons.append(ActionButton(label="Ver vídeo no Museu Virtual", url=md.get("url") or "#"))
        if md.get("source_type") in ("pdf", "tainacan"):
            action_buttons.append(ActionButton(label="Baixar ficha descritiva (PDF)", url=md.get("url") or "#"))

    answer_text = ""
    if sensitive:
        answer_text = sensitive
    else:
        provider = os.getenv("MODEL_PROVIDER", "ollama")
        if provider == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "phi3:latest")
            prompt = format_prompt(req.question, context_chunks)
            try:
                answer_text = generate_with_ollama(base, model, prompt)
            except Exception:
                answer_text = (
                    "Síntese baseada nas fontes: "
                    + " \n\n".join([f"- {c.title or c.source_type} (p. {c.page})" for c in citations])
                )
        else:
            # Fallback simples
            answer_text = (
                "Resumo das fontes consultadas: "
                + " \n\n".join([f"- {c.title or c.source_type} (p. {c.page})" for c in citations])
            )

    answer_text = prepend_ethics_disclaimer(answer_text)
    # garantir botão adicional de turismo, quando relevante
    action_buttons.append(ActionButton(label="Saber mais sobre turismo comunitário em Cunani", url="https://seuwordpress.com/turismo-comunitario-cunani"))

    return ChatResponse(answer=answer_text, citations=citations, actions=action_buttons)