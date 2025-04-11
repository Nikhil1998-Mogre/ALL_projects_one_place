from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv

# Load env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Model definition
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt setup
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])
parser = StrOutputParser()

# Chain
chain = prompt_template | model | parser

# FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using Langchain runnable interfaces",
    openapi_url="/openapi.json"
)

# Add LangServe routes
add_routes(app, chain, path="/chain")

# Fix for Pydantic schema generation error
from langserve.pydantic_v1 import rebuild_all_models
rebuild_all_models()

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
