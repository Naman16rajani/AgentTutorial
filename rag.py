from calendar import c
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

from vectorDB.addDataToVectorDB import loadVectorDB
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "dataV1")
persistent_directory = os.path.join(current_dir, "db","chroma")
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def checkAPIKey():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GEMINI_KEY"] = getpass.getpass("Enter your GEMINI API key: ")

def setQuery(query):
    combined_input = (
        "User is asking this question " + query + 
        "provide me query to find the answer from document using vector database \n"
        "data has this columns Index(['PlantVarietyID', 'PlantID', 'PlantVarietyName',"
        "'PlantVarietyDescription', 'SoilTextureID', 'PHRangeID',"
        "'IngredientNutrientID', 'OrganicMatterID', 'SalinityLevelID', 'ZoneID',"
        "'HumidityID', 'WaterRequirementMin', 'WaterRequirementMax', 'SoilTexture_SoilTexture',"
        "'SoilTexture_Description', 'PHRange_PHRange','PHRange_SoilType', 'PHRange_Description',"
        "'OrganicMatter_OrganicMatterContent', 'OrganicMatter_Description',"
        "'OrganicMatter_ImportanceToSoilAndPlants','SalinityLevel_SalinityLevel',"
        "'SalinityLevel_Classification','SalinityLevel_Description', 'SalinityLevel_ImpactonPlants',"
        "'Zone_Zone', 'Zone_TemperatureStartRange', 'Zone_TemperatureEndRange','Humidity_HumidityLevelLow',"
        "'Humidity_HumidityLevelHigh','Humidity_Classification', 'Humidity_Description',"
        "'Humidity_ImpactonPlants'],dtype='object') use context from the user question and form a query for getting the data from the document"
        "add the following in user query with same context as user question that has following information such as weather, temperature, soil type, etc. or any other factors that can help farmers to grow plants and make profit in the query"
        "give me the query in natural language to find the answer from the document"
    )

    print("\n--- Query ---")
    print(combined_input)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    ai_msg = llm.invoke(combined_input)   

    print("\n--- Query Response ---")
    print(ai_msg.content)
    return ai_msg.content

def main():
    """
    Main function to execute the retrieval and response generation process.

    This function performs the following steps:
    1. Checks the API key.
    2. Loads the vector database.
    3. Defines a query about growing conditions in South Oregon during the summer.
    4. Retrieves relevant documents from the vector store based on similarity score threshold.
    5. Displays the relevant documents and their metadata.
    6. Combines the query and relevant document contents.
    7. Uses a language model to generate a response based on the combined input.
    8. Prints the AI-generated response.

    Note: The function assumes the existence of certain variables and functions such as 
    `checkAPIKey`, `loadVectorDB`, `current_dir`, `data_path`, `persistent_directory`, 
    `gemini_embeddings`, `setQuery`, and `ChatGoogleGenerativeAI`.
    """
    checkAPIKey()
    vector_store = loadVectorDB(current_dir, data_path, persistent_directory, gemini_embeddings)
    query = "What is oak leaf?"

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.2}, 
    )

    relevant_docs = retriever.invoke(setQuery(query))

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    print(f"Number of relevant documents: {len(relevant_docs)}")
    # print(f"Query: {query}\n")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    # Combine the query and the relevant document contents
    combined_input = (
        "Here are some documents that might help answer the question: "
        + query
        + "\n\nRelevant Documents:\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    ai_msg = llm.invoke(combined_input)
    print("\n--- AI Response ---")
    print(ai_msg.content)
    # print("Hello from langchain!")


if __name__ == "__main__":
    main()
