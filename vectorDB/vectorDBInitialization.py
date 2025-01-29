import os

from langchain.vectorstores import Chroma

db = None

def initializeVectorDB(persistent_directory, gemini_embeddings):
    """
    Initialize the Chroma vector database.
    This sets up the database and allows data to be added later.
    """
    if not os.path.exists(persistent_directory):
        os.makedirs(persistent_directory)
        print(f"Created persistent directory: {persistent_directory}")

    db = Chroma(persist_directory=persistent_directory, embedding_function=gemini_embeddings)
    print("Initialized vector database.")
    return db


def getVectorDB(persistent_directory=None, gemini_embeddings=None):
    """
    Retrieve the existing Chroma vector database.
    This is used to load the database if it already exists.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    
    if persistent_directory is None:
        persistent_directory = os.path.join(parent_dir, "db", "chroma")
    
    if gemini_embeddings is None:
        # Set a default embedding function or handle the case where it's not provided
        raise ValueError("gemini_embeddings must be provided")
    global db
    if db is None:
        print("Loading new vector database...")
        db = initializeVectorDB(persistent_directory, gemini_embeddings)
    
    print("Loaded existing vector database.")
    return db