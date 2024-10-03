from dotenv import load_dotenv

load_dotenv()
import os
import discord
from discord.ext import commands

# import openai

# Load environment variables (ensure you have a .env file or set these variables)
DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize clients
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
client = discord.Client(intents=intents)


# openai.api_key = OPENAI_API_KEY

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    # Prevent the bot from responding to itself
    if message.author == client.user:
        return

    # Formulate the prompt for the LLM
    prompt = message.content

    # Call OpenAI's API
    try:
        """response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        # Extract the assistant's reply
        reply = response.choices[0].message.content.strip()
"""

        # Send the reply back to the channel
        await message.channel.send("FFF")

    except Exception as e:
        print(f"Error: {e}")
        await message.channel.send("Sorry, I couldn't process that request.")


# Run the bot
client.run(DISCORD_TOKEN)
