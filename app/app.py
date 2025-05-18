from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chat import QwenChatbot

app = FastAPI()
# 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 Chatbot
chatbot = QwenChatbot(
        base_model_path=r"D:\CoachBot\SFTuned",
        adapter_a_path=r"D:\CoachBot\DPO_1",
        adapter_b_path=r"D:\CoachBot\DPO_2",
    )
# 请求体与响应体定义
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        reply = chatbot.generate(request.message, enable_thinking=False)
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
