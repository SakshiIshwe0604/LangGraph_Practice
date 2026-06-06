from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatGroq(model='llama-3.3-70b-versatile')

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
config = {
    'tags': ['llm app','report generation','summarizzation'],
    'metadata':{'model':'llama-3.3-70b-versatile','parser':'stroutputparser'}
}
result = chain.invoke({'topic': 'Unemployment in India'},config = config)

print(result)