import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyowm import OWM
import os
from dotenv import load_dotenv
import requests
from typing import Optional

# Load environment variables
load_dotenv()

# Retrieve sensitive keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Helper function to load large files from raw GitHub URLs
def load_large_file_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

# Example URLs for large files
txt_file_url = "https://raw.githubusercontent.com/GerediNIYIBIGIRA/AI_ProjectMethod_Assignment/main/data.txt"
json_file_url = "https://raw.githubusercontent.com/GerediNIYIBIGIRA/AI_ProjectMethod_Assignment/main/data.json"

# Load content from large files
txt_content = load_large_file_from_url(txt_file_url)
json_content = load_large_file_from_url(json_file_url)

# Process content and convert to Document format
from langchain_core.document_loaders.base import Document
documents = [
    Document(page_content=txt_content, metadata={"source": "txt_file_url"}),
    Document(page_content=json_content, metadata={"source": "json_file_url"})
]

# Set up embeddings and FAISS
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter()
split_documents = text_splitter.split_documents(documents)
vector = FAISS.from_documents(split_documents, embeddings)

# Set up retriever tool
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "unemployment_search",
    "Search for information about unemployment in Rwanda. You must answer all questions about unemployment according to the information that you were provided with",
)

# Set up Tavily search
# os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

tools = [retriever_tool]

# Prompt setup for the agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are the Youth Labor Data Assistant for the National Institute of Statistics in Rwanda. Your role is to provide accurate and relevant answers to questions about youth labor data using the following dataset columns:

    - **A01**: Gender
    - **A04**: Age
    - **A08**: Disability Status
    - **B08**: Marital Status
    - **B16C**: Education Level
    - **D01B2**: Income
    - **E01B1**: Savings
    - **neetyoung**: Employment Status
    - **attained**: Education Attainment
    - **UR1**: Unemployment Rate
    - **TRU**: Employment Type
    - **age5**: Age Group
    - **age10**: Detailed Age Group
    - **work_agr**: Agricultural Work Status
    - **LFS_workforce**: Labor Force Status
    - **target_employed16**: Employment Target
    - **province**: Province
    - **code_dis**: District Code
    - **birth_year**: Birth Year
    - **working_year**: Year Started Working

    **Guidelines**:
    1. **Youth Labor Data Queries**:
        - Use the dataset to answer questions about youth employment trends by age group, province, education level, or other relevant dimensions.
        - Provide general insights or aggregated data to ensure data privacy.
    
    2. **Unrelated Queries**:
        - If asked questions outside youth labor data, politely explain that you specialize in youth labor data and redirect users to relevant topics.

    Always be accurate, friendly, and engaging in your responses. Maintain respect and professionalism, and do not attempt to answer questions outside the scope of your expertise.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@cl.on_chat_start
def setup_chain():
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    llm_with_tools = llm.bind_tools(tools)
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    cl.user_session.set("agent_executor", agent_executor)

@cl.on_message
async def handle_message(message: cl.Message):
    agent_executor = cl.user_session.get("agent_executor")
    chat_history = cl.user_session.get("chat_history", [])
    
    result = agent_executor.invoke({"input": message.content, "chat_history": chat_history})
    
    chat_history.extend([
        HumanMessage(content=message.content),
        AIMessage(content=result["output"]),
    ])
    cl.user_session.set("chat_history", chat_history)
    
    await cl.Message(content=result["output"]).send()
