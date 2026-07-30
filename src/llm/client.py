from dotenv import load_dotenv
from openai import OpenAI
from langsmith.wrappers import wrap_openai
import os

load_dotenv()

client = wrap_openai(OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
))  


