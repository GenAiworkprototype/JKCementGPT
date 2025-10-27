from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from cementGPT_llm_chat.utils.model_loader import ModelLoader
from cementGPT_llm_chat.logger import GLOBAL_LOGGER as log
from cementGPT_llm_chat.exception.custom_exception import DocumentPortalException
import json
import uuid
from datetime import datetime
from cementGPT_llm_chat.utils.file_io import save_uploaded_files
from cementGPT_llm_chat.utils.document_ops import load_documents
import hashlib
import sys

def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"

class ChatIngestor:
    def __init__(
        self,
        temp_base: Optional[str] = "data",
        faiss_base: Optional[str] = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            # Ensure ModelLoader (which may raise) is created first
            self.model_loader = ModelLoader()

            # Normalize and store flags
            self.use_session_dirs = bool(use_session_dirs)

            # Ensure session id exists (generate if None)
            self.session_id = session_id or generate_session_id()

            # Normalize bases to Path and create base directories
            self.temp_base = Path(temp_base or "data")
            self.faiss_base = Path(faiss_base or "faiss_index")

            # Create base directories if missing
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            # Resolve per-session dirs (or use base if sessionization disabled)
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info(
                "ChatIngestor initialized",
                session_id=self.session_id,
                temp_dir=str(self.temp_dir),
                faiss_dir=str(self.faiss_dir),
                sessionized=self.use_session_dirs,
            )
        except Exception as e:
            # Log the real exception but raise a friendly DocumentPortalException
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e

    def _resolve_dir(self, base: Path) -> Path:
        """
        Return directory Path. If sessionization enabled, create base/<session_id>.
        Always ensure the resulting directory exists.
        """
        base = Path(base)
        if self.use_session_dirs:
            if not getattr(self, "session_id", None):
                # should not happen because we set session_id in __init__, but guard anyway
                self.session_id = generate_session_id()
                log.warning("session_id was missing; generated a new one", session_id=self.session_id)
            d = base / self.session_id
        else:
            d = base

        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # fallback to a cwd-based directory so ingestion can continue in restricted environments
            fallback = Path.cwd() / "data" / (self.session_id or generate_session_id())
            try:
                fallback.mkdir(parents=True, exist_ok=True)
                log.warning(
                    "Failed to create configured directory; using fallback",
                    requested=str(d),
                    fallback=str(fallback),
                    error=str(e),
                )
                return fallback.resolve()
            except Exception as e2:
                log.error("Failed to create fallback directory", error=str(e2))
                raise

        return d.resolve()

    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks

    # note: kept original method name but consider renaming to 'build_retriever'
    def built_retriver(
        self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        try:
            # save_uploaded_files should return a list of filesystem paths (strings or Path objects)
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            if not paths:
                raise ValueError("No files saved by save_uploaded_files")

            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # FAISS manager - responsible for load/create/add
            fm = FaissManager(self.faiss_dir, self.model_loader)

            texts = [c.page_content for c in chunks]
            metas = [c.metadata or {} for c in chunks]

            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception as e:
                # Try again once more (original code attempted this). If it fails, raise.
                log.warning("load_or_create failed on first attempt, retrying", error=str(e))
                vs = fm.load_or_create(texts=texts, metadatas=metas)

            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))

            # Configure search parameters based on search type
            search_kwargs = {"k": k}
            if search_type == "mmr":
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult
                log.info("Using MMR search", k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

            return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                # If meta file is corrupt, re-init meta but log
                log.warning("Failed to parse existing ingested_meta.json; reinitializing meta")
                self._meta = {"rows": {}}

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        # check for FAISS artifacts (index files). Names may depend on FAISS implementation.
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            # include row id if present, else just source
            return f"{src}::{rid if rid is not None else ''}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_documents(self, docs: List[Document]):
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents().")

        new_docs: List[Document] = []
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)

        if new_docs:
            # FAISS vectorstore API: add_documents then save_local
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None):
        # Load if existing
        if self._exists():
            # Note: allow_dangerous_deserialization=True is required by some FAISS versions but has security implications.
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs

        # Create if we have data
        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
