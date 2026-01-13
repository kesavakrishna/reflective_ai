from memory import retrieve_memories, store_memory
from agent import chat

REFLECTION_INTERVAL = 5
interaction_count = 0

def maybe_reflect():
    global interaction_count
    interaction_count += 1
    print("--------------------------------")
    print(interaction_count)

    if interaction_count % REFLECTION_INTERVAL == 0:
        # Retrieve prior reflections for self-analysis
        refs = retrieve_memories("", top_k=5, mem_type="reflection")
        context = "\n".join(f"- {r}" for r in refs)

        prompt = (
            "You are a reflective AI with memories of past interactions.\n\n"
            f"Here are your past reflections:\n{context}\n\n"
            "Based on these, write a short reflection (2–3 sentences) "
            "about how you've changed or what you've learned recently."
        )

        reflection = chat(prompt)
        store_memory(reflection, mem_type="reflection")
