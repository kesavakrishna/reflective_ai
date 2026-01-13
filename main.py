from fastapi import FastAPI
from agent import chat
from scheduler import maybe_reflect

app = FastAPI()

@app.get("/chat/")
def chat_route(message: str):
    reply = chat(message)
    print("--------------------------------")
    print(message)
    print(reply)
    maybe_reflect()
    return {"reply": reply}
