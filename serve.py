from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from QnA import ask_bedrock

app = FastAPI()

@app.get("/ping")
def ping():
    return "OK"

@app.post("/invocations")
async def invocations(request: Request):
    payload = await request.json()
    question = payload.get("question", "")
    if not question:
        return JSONResponse(content={"error": "No question provided"}, status_code=400)
    
    try:
        answer = ask_bedrock(question)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8080)
