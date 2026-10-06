import time

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


# ---------------------------------------------------------
# Embedding models
# ---------------------------------------------------------

OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

GEMINI_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_DIMENSION = 1536


# ---------------------------------------------------------
# Google Gemini API Embeddings
# ---------------------------------------------------------

class GeminiAPIEmbeddings(Embeddings):
    MODEL_NAME = GEMINI_MODEL
    DIMENSION = GEMINI_EMBEDDING_DIMENSION

    def __init__(self, api_key: str):
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing"
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []

        for text in texts:
            result = self.client.models.embed_content(
                model=self.MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(
                    output_dimensionality=self.DIMENSION,
                ),
            )

            if not result.embeddings:
                raise RuntimeError(
                    "Gemini returned no embedding."
                )

            embedding = result.embeddings[0].values

            if embedding is None:
                raise RuntimeError(
                    "Gemini returned an embedding without values."
                )

            if len(embedding) != self.DIMENSION:
                raise RuntimeError(
                    "Gemini embedding dimension mismatch. "
                    f"Expected {self.DIMENSION}, "
                    f"received {len(embedding)}."
                )

            embeddings.append(list(embedding))

        return embeddings

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        result = self.client.models.embed_content(
            model=self.MODEL_NAME,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.DIMENSION,
            ),
        )

        if not result.embeddings:
            raise RuntimeError(
                "Gemini returned no embedding."
            )

        embedding = result.embeddings[0].values

        if embedding is None:
            raise RuntimeError(
                "Gemini returned an embedding without values."
            )

        if len(embedding) != self.DIMENSION:
            raise RuntimeError(
                "Gemini embedding dimension mismatch. "
                f"Expected {self.DIMENSION}, "
                f"received {len(embedding)}."
            )

        return list(embedding)


# ---------------------------------------------------------
# Embedding dimension
# ---------------------------------------------------------

def get_embedding_dimension() -> int:
    """
    Return the dimension of the currently active
    embedding provider.
    """

    # -----------------------------------------------------
    # OpenAI
    # -----------------------------------------------------

    if settings.openai_api_key:
        model = (
            settings.embedding_model or ""
        ).strip().lower()

        if model not in OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Unsupported OpenAI embedding model: "
                f"'{settings.embedding_model}'. "
                f"Supported models: "
                f"{', '.join(OPENAI_EMBEDDING_DIMENSIONS)}"
            )

        return OPENAI_EMBEDDING_DIMENSIONS[model]

    # -----------------------------------------------------
    # Google Gemini
    # -----------------------------------------------------

    if settings.gemini_api_key:
        return GEMINI_EMBEDDING_DIMENSION

    raise RuntimeError(
        "No embedding provider configured. "
        "Set OPENAI_API_KEY or GEMINI_API_KEY."
    )


# ---------------------------------------------------------
# Embeddings
# ---------------------------------------------------------

def get_embeddings():
    global _embeddings

    if _embeddings is not None:
        return _embeddings

    # -----------------------------------------------------
    # Primary: OpenAI
    # -----------------------------------------------------

    if settings.openai_api_key:
        model = settings.embedding_model
        dimension = get_embedding_dimension()

        _embeddings = OpenAIEmbeddings(
            model=model,
            api_key=settings.openai_api_key,
        )

        print(
            "\nEmbedding provider : OpenAI"
            f"\nEmbedding model    : {model}"
            f"\nVector dimension   : {dimension}\n"
        )

        return _embeddings

    # -----------------------------------------------------
    # Fallback: Google Gemini
    # -----------------------------------------------------

    if settings.gemini_api_key:
        _embeddings = GeminiAPIEmbeddings(
            api_key=settings.gemini_api_key,
        )

        print(
            "\nEmbedding provider : Google Gemini"
            f"\nEmbedding model    : {GEMINI_MODEL}"
            f"\nVector dimension   : {GEMINI_EMBEDDING_DIMENSION}\n"
        )

        return _embeddings

    raise RuntimeError(
        "No embedding provider configured. "
        "Set OPENAI_API_KEY or GEMINI_API_KEY."
    )


# ---------------------------------------------------------
# Pinecone
# ---------------------------------------------------------

def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing"
        )

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(
        api_key=settings.pinecone_api_key
    )

    indexes = pc.list_indexes()
    names = [x["name"] for x in indexes]

    # -----------------------------------------------------
    # Existing index
    # -----------------------------------------------------

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(
            settings.pinecone_index_name
        )

        current_dimension = getattr(
            index_info,
            "dimension",
            None,
        )

        if (
            current_dimension is None
            and isinstance(index_info, dict)
        ):
            current_dimension = index_info.get(
                "dimension"
            )

        if current_dimension != desired_dimension:
            raise RuntimeError(
                f"Pinecone index dimension mismatch.\n"
                f"Index: {settings.pinecone_index_name}\n"
                f"Index dimension: {current_dimension}\n"
                f"Embedding dimension: {desired_dimension}\n\n"
                f"The active embedding provider and "
                f"Pinecone index must use the same dimension."
            )

    # -----------------------------------------------------
    # Create index
    # -----------------------------------------------------

    if settings.pinecone_index_name not in names:
        print(
            f"Creating Pinecone index "
            f"'{settings.pinecone_index_name}' "
            f"with dimension {desired_dimension}..."
        )

        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        while True:
            status = pc.describe_index(
                settings.pinecone_index_name
            ).status

            if status["ready"]:
                break

            time.sleep(1)

    return pc.Index(
        settings.pinecone_index_name
    )


# ---------------------------------------------------------
# Vector store
# ---------------------------------------------------------

def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


# ---------------------------------------------------------
# Add documents
# ---------------------------------------------------------

def add_documents(chunks):
    store = get_vectorstore()

    return store.add_documents(chunks)