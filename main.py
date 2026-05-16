from fastapi import FastAPI

from agent import answer

app = FastAPI()


@app.get("/ask/")
def ask_route(question: str):
    reply, _ = answer(question)
    return {"answer": reply}
