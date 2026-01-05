import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

basic_model = ChatTongyi(
    model_name="qwen-plus",
    temperature=0
)

advanced_model = ChatTongyi(
    model_name="qwen-plus",
    temperature=0,
    streaming=True
)

advanced_model = ChatOpenAI(
  model="gpt-5",
  temperature=0.1,
  base_url="https://api.fe8.cn/v1",
  api_key=os.getenv("OPENAI_API_KEY"),
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=[],
    middleware=[dynamic_model_selection],
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

# 测试 basic_model（消息数 <= 10）
print("=" * 60)
print("测试 1: basic_model（消息数 <= 10）")
print("=" * 60)
response1 = agent.invoke({"messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]})
print("使用的模型: basic_model (ChatTongyi)")
print("响应:", response1)
print()

# 测试 advanced_model（消息数 > 10）
# 创建一个包含超过10条消息的对话历史来触发 advanced_model
print("=" * 60)
print("测试 2: advanced_model（消息数 > 10）")
print("=" * 60)
print("构建包含11条消息的对话历史...")

# 构建一个包含超过10条消息的对话历史
messages_history = []
questions = [
    "什么是人工智能？",
    "Python 和 Java 有什么区别？",
    "如何学习编程？",
    "什么是机器学习？",
    "区块链技术有什么应用？"
]

for i, question in enumerate(questions, 1):
    messages_history.append({"role": "user", "content": f"问题 {i}: {question}"})
    messages_history.append({"role": "assistant", "content": f"回答 {i}: 这是一个关于{question.split('？')[0]}的问题，让我来为您解答。"})

# 添加第11条消息，这次会触发 advanced_model
messages_history.append({"role": "user", "content": "请详细解释一下：什么是深度学习？它和机器学习有什么区别？"})
print(f"总消息数: {len(messages_history)} (会触发 advanced_model)")
print("使用的模型: advanced_model (ChatOpenAI)")
print()

try:
    response2 = agent.invoke({"messages": messages_history})
    print("响应:", response2)
except Exception as e:
    print(f"错误: {e}")
    print("提示: 如果没有设置 OPENAI_API_KEY，advanced_model 会失败")
    print("可以在 .env 文件中添加: OPENAI_API_KEY=your-api-key")
print()