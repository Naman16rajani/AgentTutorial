import os
import getpass
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from vectorDB.vectorDBInitialization import getVectorDB

load_dotenv()


def addDataToVectorDB(data_path, vector_db):
    """
    Add data from the specified path to the initialized vector database.
    """

    if vector_db is None:
        vector_db = getVectorDB()

    # Ensure the data directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"The data directory {data_path} does not exist. Please check the path."
        )

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

    # Load documents from CSV files and add metadata
    documents = []
    for csv_file in csv_files:
        file_path = os.path.join(data_path, csv_file)
        loader = CSVLoader(file_path)
        csv_docs = loader.load()
        for doc in csv_docs:
            doc.metadata = {"source": csv_file}  # Add source metadata
            documents.append(doc)

    # Split the documents into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # Add the documents to the existing vector database
    print("\n--- Adding data to vector database ---")
    vector_db.add_documents(docs)
    print("\n--- Finished adding data to vector database ---")


# def loadVectorDB(current_dir, data_path, persistent_directory, gemini_embeddings):
#     if not os.path.exists(persistent_directory):
#         print("Persistent directory does not exist. Initializing vector store...")

#         # Ensure the data directory exists
#         if not os.path.exists(data_path):
#             raise FileNotFoundError(
#                 f"The directory {data_path} does not exist. Please check the path."
#             )
#         # List all text files in the directory
#         csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]

#         # Read the text content from each file and store it with metadata
#         documents = []
#         for csv_file in csv_files:
#             file_path = os.path.join(data_path, csv_file)
#             loader = CSVLoader(file_path)
#             book_docs = loader.load()
#             for doc in book_docs:
#                 # Add metadata to each document indicating its source
#                 doc.metadata = {"source": csv_file}
#                 documents.append(doc)
        

#         # Split the documents into chunks
#         text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
#         docs = text_splitter.split_documents(documents)

#         # Display information about the split documents
#         print("\n--- Document Chunks Information ---")
#         print(f"Number of document chunks: {len(docs)}")

#         # Create embeddings
#         print("\n--- Creating embeddings ---")
#         embeddings = gemini_embeddings  # Update to a valid embedding model if needed
#         print("\n--- Finished creating embeddings ---")

#         print("\n--- Creating vector store ---")
#         db = initialiseVectorDB(embeddings, persist_directory=persistent_directory)
#         print("\n--- Finished creating vector store ---")

#         # Persist the vector store
#         # print("\n--- Persisting vector store ---")
#         # db.persist()
#         # print("\n--- Finished persisting vector store ---")

#         # return db
#         # Create the vector store and persist it
#         print("\n--- Creating and persisting vector store ---")
#         db = db.from_documents(
#             docs, embeddings, persist_directory=persistent_directory)
#         print("\n--- Finished creating and persisting vector store ---")

#         return db
    
#     else:
#         print("Loading existing vector store...")
#         db = initialiseVectorDB(embeddings, persist_directory=persistent_directory)

#     return db

