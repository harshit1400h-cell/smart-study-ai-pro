import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt", "jpg", "jpeg", "png"}

load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    embedding_provider: str
    openai_api_key: str
    google_api_key: str

    @property
    def llm_api_key(self) -> str:
        return self.openai_api_key if self.llm_provider == "openai" else self.google_api_key

    @property
    def embedding_api_key(self) -> str:
        return self.openai_api_key if self.embedding_provider == "openai" else self.google_api_key

    def missing_llm_key(self) -> bool:
        return not self.llm_api_key

    def missing_embedding_key(self) -> bool:
        return not self.embedding_api_key


def get_config() -> AppConfig:
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower()
    if llm_provider not in {"gemini", "openai"}:
        raise ValueError("LLM_PROVIDER must be either 'gemini' or 'openai'.")
    if embedding_provider not in {"gemini", "openai"}:
        raise ValueError("EMBEDDING_PROVIDER must be either 'gemini' or 'openai'.")
    return AppConfig(
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
    )
