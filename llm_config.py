import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# This module is imported before the load_dotenv() calls in raga.py / agent.py
# run, so load .env here rather than relying on an importer to have done it.
load_dotenv()
#####################################################
# Cloudflare Workers AI — OpenAI-compatible endpoint.
# Credentials live in .env (gitignored); never hardcode them here.
ACCOUNT_ID = os.environ["CLOUDFLARE_ACCOUNT_ID"]
API_TOKEN = os.environ["CLOUDFLARE_API_TOKEN"]


model = ChatOpenAI(
    model="@cf/deepseek-ai/deepseek-v4-pro-0813",
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1",
    api_key=API_TOKEN,
)
######################################################################