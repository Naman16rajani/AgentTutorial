from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langgraph.checkpoint.memory import MemorySaver
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tools.checkTime import get_system_time
from vectorDB.retrieveFromVectorDB import retrieveFromVectorDB
from vectorDB.addDataToVectorDB import addDataToVectorDB
from vectorDB.vectorDBInitialization import getVectorDB
import requests
from langchain.agents import tool

load_dotenv()


@tool
def get_weather(city: str):
    """Returns the weather information of given city"""
    
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    if not WEATHER_API_KEY:
        return "Weather API key is not set. Please check your environment variables."
    
    try:
        response = requests.get(base_url, params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"})
        data = response.json()
        
        if response.status_code == 200:
            temperature = data["main"]["temp"]
            weather_description = data["weather"][0]["description"]
            return f"The current temperature in {city} is {temperature}°C with {weather_description}."
        else:
            return f"Could not fetch weather data for {city}. Error: {data.get('message', 'Unknown error')}."
    except Exception as e:
        return f"An error occurred while fetching weather data: {str(e)}"


def main():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "dataV1")
    persistent_directory = os.path.join(current_dir, "db","chroma")
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    
    vDB = getVectorDB(persistent_directory, gemini_embeddings)

    # addDataToVectorDB(data_path, vDB)

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

    # query = "What is Ventura? Can i grow it in sf?"
    query = "What is oakleaf ? Can i grow in sf?"

    prompt_template = hub.pull("hwchase17/react")
    # prompt_template = PromptTemplate.from_template(
    #     "You are a helpful assistant. You have access to the following tools: {tools}. "
    #     "You can use these tools to answer questions. "
    #     "Your goal is to answer the question as accurately as possible. "
    #     "If you don't know the answer, say 'I don't know'. "
    #     "Question: {input}\n"
    #     "if you ask a question that i can grow this crop in particular area so please look at the weather tool"
    # )

    tools = [get_system_time, get_weather]

    agent = create_react_agent(llm, tools, prompt_template)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    query = retrieveFromVectorDB(query,vDB)

    agent_executor.invoke({"input": query})


if __name__ == "__main__":
    main()
