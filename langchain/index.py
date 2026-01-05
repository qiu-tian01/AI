import os
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain_community.chat_models import ChatTongyi

# 加载 .env 文件中的环境变量
load_dotenv()


# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

# Configure model - 使用 DashScope 的 qwen-plus 模型
# ChatTongyi 会自动从环境变量 DASHSCOPE_API_KEY 读取 API key
model = ChatTongyi(
    model_name="qwen-plus",  # DashScope qwen-plus 模型
    temperature=0 # 控制随机性
)

tools = [get_user_location, get_weather_for_location]

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
# 注意：DashScope API 对 tool_choice 参数有限制，需要确保传递正确的值
# 通过 monkey patch 修复 ChatTongyi 的 _generate 方法
original_generate = model._generate

def patched_generate(self, messages, stop=None, run_manager=None, **kwargs):
    # 确保 tool_choice 是 "auto" 或 "none"，或者移除它
    if 'tool_choice' in kwargs:
        tool_choice = kwargs['tool_choice']
        if tool_choice not in ['auto', 'none']:
            kwargs['tool_choice'] = 'auto'
    return original_generate(messages, stop=stop, run_manager=run_manager, **kwargs)

model._generate = patched_generate.__get__(model, type(model))

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    context_schema=Context,
    response_format=ToolStrategy[ResponseFormat](ResponseFormat), # ToolStrategy使用人工工具调用来生成结构化输出。这适用于任何支持工具调用的模型：
    checkpointer=checkpointer,
    system_prompt="You are a helpful assistant. Be concise and accurate."
    # 还有一个ProviderStrategy 使用模型提供商的原生结构化输出生成功能。这种方式更可靠，但仅适用于支持原生结构化输出的提供商（例如 OpenAI）：
    #  response_format=ProviderStrategy(ContactInfo)
)

# Run agent
# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
#     weather_conditions="It's always sunny in Florida!"
# )


# Note that we can continue the conversation using the same `thread_id`.
response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
# ResponseFormat(
#     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
#     weather_conditions=None
# )