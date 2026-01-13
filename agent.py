from memory import store_memory, retrieve_memories
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def build_prompt(user_input):
    user_history = retrieve_memories(user_input, mem_type="user")
    agent_history = retrieve_memories(user_input, mem_type="agent")
    reflection_history = retrieve_memories(user_input, mem_type="reflection")

    user_context = "\n".join(f"User said: {u}" for u in user_history)
    agent_context = "\n".join(f"AI replied: {a}" for a in agent_history)
    reflection_context = "\n".join(f"Reflection: {r}" for r in reflection_history)

    prompt = f"""You are a reflective AI. You remember and grow over time.

Recent user thoughts:
{user_context}

Your past responses:
{agent_context}

Your reflections:
{reflection_context}

New input: \"{user_input}\"

Respond thoughtfully, integrating your past experience."""
    return prompt

def chat(user_input):
    prompt = build_prompt(user_input)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    reply = response.text
    store_memory(user_input, mem_type="user")
    store_memory(reply, mem_type="agent")
    return reply
