import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")


def get_embeddings():
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))
    raise RuntimeError("Defina GOOGLE_API_KEY ou OPENAI_API_KEY no arquivo .env.")


def ingest_pdf():
    if not PDF_PATH:
        raise RuntimeError("Defina PDF_PATH no arquivo .env.")
    if not DATABASE_URL:
        raise RuntimeError("Defina DATABASE_URL no arquivo .env.")
    if not PG_VECTOR_COLLECTION_NAME:
        raise RuntimeError("Defina PG_VECTOR_COLLECTION_NAME no arquivo .env.")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # PGVector rejeita metadados com valores None
    enriched_chunks = [
        Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v is not None},
        )
        for chunk in chunks
    ]

    embeddings = get_embeddings()

    PGVector.from_documents(
        documents=enriched_chunks,
        embedding=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    print(f"Ingestão concluída: {len(enriched_chunks)} chunks armazenados na collection '{PG_VECTOR_COLLECTION_NAME}'.")


if __name__ == "__main__":
    ingest_pdf()