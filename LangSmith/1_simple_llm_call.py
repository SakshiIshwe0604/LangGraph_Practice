from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate.from_template("{question}")

model = ChatGroq(model='llama-3.3-70b-versatile')
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is the capital of INDIA?"})
print(result)