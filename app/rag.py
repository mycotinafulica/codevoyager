import os
import uuid
import chromadb
import tiktoken
import time
import app.constants as constants
from openai import OpenAI
from dataclasses import dataclass
from app.utilities import load_api_key

def create_rag_database(source: str, db_path: str, 
                        directory_filters: list[str] = []) -> None:
    print("Creating rag database from source : " + source)

    api_key = load_api_key()

    openai = OpenAI(api_key=api_key)

    chroma_client = chromadb.PersistentClient(path=db_path)
    collection    = chroma_client.get_or_create_collection(name=constants.COLLECTION_NAME)

    files_to_embed = get_file_for_embeddings(source, directory_filters)

    current_tokens_in_batch = 0
    batch_number = 1
    processed_files = 0
    ids: list[str]       = []
    contents: list[str]  = []
    metadatas: list[str] = []
    for file_path in files_to_embed:
        processed_files += 1
        source_relative_path = file_path.replace(source, ".").replace("\\", "/")

        metadata = "The following content is taken from the following file : " + source_relative_path + "The content is the following:\n\n"
        content_to_embed = metadata + read_file(file_path)

        tokenizer   = tiktoken.encoding_for_model("text-embedding-3-small")
        token_count = len(tokenizer.encode(content_to_embed))

        # Open AI API has limitation for embedding, it allows up to 8192 tokens. Possible improvement is to actually chunk the large file into manageable pieces
        # such as individual methods, but this leave extra works since every language will need different parser, but it can be improved for sure.
        if token_count < 8192:
            if current_tokens_in_batch + token_count < 750000: # There's rate limit of how many tokens you can add in one embedding go, the current limit is 1m
                ids.append(str(uuid.uuid4()))
                contents.append(content_to_embed)
                metadatas.append({"source" : metadata})
                current_tokens_in_batch += token_count
            else:
                print("Processing the " + str(batch_number) + "th batch")
                print("Number of files processed : " + str(processed_files))
                print("The number of tokens is : " + str(current_tokens_in_batch))
                embed_and_save(openai, collection, ids, contents, metadatas)
                ids = []
                contents = []
                metadatas = []
                current_tokens_in_batch = 0
                batch_number += 1
                # Avoiding rate limit. A possible optimization would actually to implement exponential backoff.
                time.sleep(120)
        else:
            print("skipped file : " + file_path)
        
    if len(contents) != 0:
        print("Processing the " + str(batch_number) + " batch")
        print("The number of tokens is : " + str(current_tokens_in_batch))
        embed_and_save(openai, collection, ids, contents, metadatas)

    print("Database saved on : " + db_path)

def embed_and_save(openai: OpenAI, collection: chromadb.Collection, ids: list[str], contents: list[str], metadatas: list[str]):
    vectors = get_openai_embeddings(openai, contents)
    collection.add(
        ids=ids,
        documents=contents,
        embeddings=vectors,
        metadatas=metadatas
    )

def get_openai_embeddings(openai: OpenAI, documents: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(
        input=documents,
        model=constants.EMBEDDING_MODEL
    )

    embeddings = [r.embedding for r in response.data]
    return embeddings

def get_file_for_embeddings(directory: str, directory_filters: list[str]) -> list[str]:
    print("Collecting files for embeddings....")
    file_for_embeddings = []
    for dirpath, dirnames, filenames in os.walk(directory):
        # skip .git & .github as they doesn't contain source code useful for human reading
        filters = [".git", ".github", ".vscode"] + directory_filters

        if all(s not in dirpath for s in filters):
            for filename in filenames:
                # it is best to avoid binary files such as images
                full_file_name = dirpath + os.sep + filename
                if is_file_utf8(full_file_name):
                    file_for_embeddings.append(full_file_name)

    print("Number of files to embed : " + str(len(file_for_embeddings)))

    return file_for_embeddings

def is_file_utf8(file_path: str) -> bool:
    try:
        # Attempt to read the file as text
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read() 
        return True
    except (UnicodeDecodeError, IOError) as e:
        return False

def read_file(file_path):
    try:
        # Attempt to read the file as text
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read() 
        return content
    except (UnicodeDecodeError, IOError) as e:
        print(f"Text reading failed: {e}. Falling back to binary...")
        # Fall back to reading the file as binary if text reading fails
        with open(file_path, 'rb') as file:
            content = file.read()  # Read file as binary
        print("File read as binary : " + file_path)
        return content




@dataclass
class EmbeddingBatch:
    contents: list[str]
    metadatas: list[str]