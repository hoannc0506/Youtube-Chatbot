from fastapi import FastAPI
from pydantic import BaseModel

import agent_utils


# Define a Pydantic model for the request body
class ChatRequest(BaseModel):
    query: str


app = FastAPI()

agent = agent_utils.get_agent()

@app.post("/chat")
def chat(request: ChatRequest):
    return {"message": f"received {request.query}"}

app = FastAPI()
# agent = get_agent()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
def chat(request: ChatRequest):

    response = agent.chat(request.query)
    
    return {"message": response.response}

# if __name__ == "__main__":
    