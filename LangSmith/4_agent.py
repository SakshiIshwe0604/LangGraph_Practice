from langchain_groq import ChatGroq
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Fetches current weather for a city"""
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url)
    data = response.json()
    current = data["current_condition"][0]
    return f"Temperature: {current['temp_C']}°C, Feels like: {current['FeelsLikeC']}°C, Weather: {current['weatherDesc'][0]['value']}"

# Free-tier: Groq instead of OpenAI
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True  # groq models can sometimes misformat ReAct output
)

response = agent_executor.invoke({"input": "Identify the birthplace city of Kalpana Chawla (search) and give its current temperature."})
print(response)
print(response['output'])