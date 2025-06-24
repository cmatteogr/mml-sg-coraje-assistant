import discord
from discord.ext import commands
import os
from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.model.inference.inference_aws import LLMInferenceSagemakerEndpoint
from llm_engineering.model.inference.inference_local import LLMInferenceLocal
from llm_engineering.model.inference.run import InferenceExecutor
from llm_engineering.settings import settings

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=discord.Intents.default())

retriever = ContextRetriever(mock=False)
llm = LLMInferenceLocal()

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