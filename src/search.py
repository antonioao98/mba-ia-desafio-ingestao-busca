import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings():
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"))
    raise RuntimeError("Defina GOOGLE_API_KEY ou OPENAI_API_KEY no arquivo .env.")


def get_llm():
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"), temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL"), temperature=0)
    raise RuntimeError("Defina GOOGLE_API_KEY ou OPENAI_API_KEY no arquivo .env.")


def search_prompt(question=None):
    if not DATABASE_URL:
        raise RuntimeError("Defina DATABASE_URL no arquivo .env.")
    if not PG_VECTOR_COLLECTION_NAME:
        raise RuntimeError("Defina PG_VECTOR_COLLECTION_NAME no arquivo .env.")

    store = PGVector(
        embeddings=get_embeddings(),
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
    prompt = PromptTemplate(input_variables=["contexto", "pergunta"], template=PROMPT_TEMPLATE)
    chain = prompt | get_llm() | StrOutputParser()

    def answer(pergunta: str) -> str:
        resultados = store.similarity_search_with_score(pergunta, k=10)
        contexto = "\n\n".join(doc.page_content for doc, _score in resultados)
        return chain.invoke({"contexto": contexto, "pergunta": pergunta})

    if question is not None:
        return answer(question)

    return answer