from config import AppConfig


class AIServiceError(Exception):
    """An AI provider request failed without exposing credentials."""


def embed_texts(
    texts: list[str],
    config: AppConfig,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    if not texts:
        return []
    if config.missing_embedding_key():
        provider = config.embedding_provider.title()
        raise AIServiceError(
            f"Add the {provider} API key to the .env file to index and search documents."
        )

    try:
        if config.embedding_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=config.openai_api_key, timeout=60)
            results = []
            for start in range(0, len(texts), 100):
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=texts[start:start + 100],
                )
                results.extend(item.embedding for item in response.data)
            return results

        from google import genai
        from google.genai import types
        client = genai.Client(api_key=config.google_api_key)
        results = []
        for start in range(0, len(texts), 32):
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts[start:start + 32],
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=768,
                    http_options=types.HttpOptions(timeout=60_000),
                ),
            )
            results.extend(item.values for item in response.embeddings)
        return results
    except Exception as exc:
        raise AIServiceError(
            "The embedding provider could not process this request. Check your provider setup, network, and quota."
        ) from exc
