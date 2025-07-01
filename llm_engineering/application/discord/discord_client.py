import discord
from discord.ext import commands
import os
from loguru import logger
from langchain_openai import ChatOpenAI
from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.model.inference.inference_local import LLMInferenceLocal
from llm_engineering.model.inference.run import InferenceExecutor
from llm_engineering.infrastructure.opik_utils import configure_opik
from langchain_community.llms import VLLM
from opik.integrations.langchain import OpikTracer

VLLM_API_BASE = "http://localhost:8000/v1"
VLLM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=discord.Intents.default())


opik_tracer = OpikTracer()

model_client = ChatOpenAI(
    model=VLLM_MODEL_NAME,
    openai_api_key="EMPTY",  # VLLM server doesn't need a key
    openai_api_base=VLLM_API_BASE,
    max_tokens=1024,
    temperature=0.2,
    model_kwargs={
        "top_p": 0.95,
        #"top_k": 10,
    },
    callbacks=[opik_tracer],
)

"""model = VLLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    trust_remote_code=True,  # Necessary for DeepSeek models
    max_new_tokens=1024,
    top_k=10,
    top_p=0.95,
    temperature=0.7,
    callbacks=[opik_tracer],)"""

retriever = ContextRetriever(model_client, mock=False)
llm = LLMInferenceLocal(model_client)

configure_opik()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')


@bot.event
async def on_message(message):
    """
    This event is called every time a message is sent in a channel
    the bot can access.
    """
    # Important: Ignore messages from the bot itself to prevent infinite loops
    if message.author == bot.user:
        return

    query = message.content.lower()
    logger.info(f"query: {query}")
    query = query.replace('<@1291457952456380478>', '').strip()
    documents = retriever.search(query, k=10)
    context = EmbeddedChunk.to_context(documents)

    answer = InferenceExecutor(llm, query, context=context).execute()

    logger.info(f"{answer}, {message.author}")

    await message.channel.send(answer)

    # --- Process Commands ---
    # If you are also using the commands extension (bot.command()),
    # you MUST include this line to allow your commands to be processed.
    # Without it, on_message will block your commands from working.
    await bot.process_commands(message)


bot.run(DISCORD_TOKEN)