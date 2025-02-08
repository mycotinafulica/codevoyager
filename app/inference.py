import chromadb
import app.constants as constants
from openai import OpenAI
from app.utilities import load_api_key

# TODO : Need error handling
collection: chromadb.Collection
openai: OpenAI
current_similar_items: list[str]

def initialize(database_path: str):
    global collection, openai
    chroma_client = chromadb.PersistentClient(path=database_path)
    collection    = chroma_client.get_or_create_collection(name=constants.COLLECTION_NAME)

    api_key = load_api_key()
    openai  = OpenAI(api_key=api_key)

def get_current_similar_items() -> list[str]:
    return current_similar_items


# note that this function modify the list being passed
def inquiry_ai(message: str, chat_history: list[dict[str, str]]):
    global current_similar_items
    current_similar_items = find_similars(message)

    print(chat_history)

    # The original history won't include the RAG docs, while the one that goes to chatgpt will include one
    # This will save us some tokens
    enriched_chat_history = chat_history.copy()
    chat_history.append({"role": "user", "content": message})

    user_inquiry = message + "\nTo help you answer the question, the following files might help:\n"
    for item in current_similar_items:
        user_inquiry += item + "\n=================================================="

    enriched_chat_history.append({"role": "user", "content": user_inquiry})
    response = openai.chat.completions.create(
        model=constants.INFERENCE_MODEL,
        messages=enriched_chat_history
    )
    result = response.choices[0].message.content

    chat_history.append({"role": "assistant", "content": result})

    return result


def get_openai_embeddings(documents: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        input=documents,
        model=constants.EMBEDDING_MODEL
    )

    embeddings = [r.embedding for r in response.data]
    return embeddings

def vector(inquiry) -> list[list[float]]:
    return get_openai_embeddings([inquiry])

def find_similars(inquiry) -> list[str]:
    results = collection.query(query_embeddings=vector(inquiry), n_results=5)
    documents = results['documents'][0][:]
    return documents
