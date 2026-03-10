import os
import dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import tool
from langchain.agents import create_agent
from chess_board_recognizer import recognize_chess_board
dotenv.load_dotenv()

# 初始化模型
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

chat_history = ChatMessageHistory()
current_fen = {"fen": None}

@tool
def recognize_board_from_image(image_path: str) -> str:
    """从棋盘图片中识别棋局，返回 FEN 字符串。当用户上传了棋盘图片时调用此工具。"""
    try:
        fen = recognize_chess_board(image_path)
        current_fen["fen"] = fen
        return f"识别成功，FEN: {fen}。w=白方走，b=黑方走。"
    except Exception as e:
        return f"识别失败: {str(e)}"

# ✅ 直接用 create_agent，参数最简洁
agent = create_agent(
    model=chat_model,
    tools=[recognize_board_from_image],
    system_prompt="""你是一位专业的国际象棋助手。职责：
1. 帮助用户分析棋局
2. 推荐最佳走法并解释原因  
3. 提供友好、专业的建议
当用户上传图片时，调用 recognize_board_from_image 工具识别棋盘。"""
)

def chat(message, history):
    user_text = ""
    image_path = None

    if isinstance(message, dict):
        user_text = message.get("text", "") or ""
        files = message.get("files", [])
        if files:
            image_path = files[0]
    else:
        user_text = message

    user_input = f"{user_text}\n\n[图片路径: {image_path}，请识别棋盘]" if image_path else user_text

    messages = chat_history.messages + [HumanMessage(content=user_input)]
    result = agent.invoke({"messages": messages})
    response = result["messages"][-1].content

    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response)

    return response

demo = gr.ChatInterface(
    fn=chat,
    title="♟️ 国际象棋助手 (Agent 模式)",
    description="Agent 自动判断是否识别棋盘图片",
    examples=["你是谁", "什么是国际象棋开局原则？"],
    multimodal=True
)

demo.launch()