"""
LangGraph Agent 示例程序 - 带工具调用的算术助手

程序功能：
    本程序演示了如何使用 LangGraph 构建一个具备工具调用能力的 AI Agent。
    Agent 可以理解用户的自然语言指令，自动调用相应的工具函数来执行算术运算。

主要组件：
    1. 模型配置：使用 DashScope 的 qwen-plus 模型（通过 ChatTongyi）
    2. 工具定义：定义了三个算术工具函数
       - add: 加法运算
       - multiply: 乘法运算
       - divide: 除法运算
    3. 状态图构建：使用 LangGraph 的 StateGraph 构建 Agent 工作流
       - llm_call 节点：LLM 分析用户输入，决定是否需要调用工具
       - tool_node 节点：执行工具调用并返回结果
    4. 条件路由：根据 LLM 是否调用工具来决定下一步流程

工作流程：
    START -> llm_call -> (判断是否需要工具) 
         -> tool_node (如果需要) -> llm_call -> (继续判断)
         -> END (如果不需要，返回最终答案)

使用示例：
    用户输入："Add 3 and 4."
    Agent 流程：
        1. LLM 分析输入，识别需要调用 add 工具
        2. 调用 add(3, 4) 工具，得到结果 7
        3. LLM 将结果整合成自然语言回复给用户

依赖：
    - 需要在 .env 文件中配置 DASHSCOPE_API_KEY
    - 需要安装：langchain, langchain-community, langgraph, python-dotenv
"""

# ==================== 步骤 1: 定义工具和模型 ====================

import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.chat_models import ChatTongyi

# 加载 .env 文件中的环境变量（包含 DASHSCOPE_API_KEY）
load_dotenv()

# 配置模型 - 使用 DashScope 的 qwen-plus 模型
# ChatTongyi 会自动从环境变量 DASHSCOPE_API_KEY 读取 API key
model = ChatTongyi(
    model_name="qwen-plus",  # DashScope qwen-plus 模型
    temperature=0  # 控制随机性，0 表示更确定性的输出
)


# ==================== 定义工具函数 ====================
# 使用 @tool 装饰器将普通函数转换为 LangChain 工具
# 这些工具可以被 LLM 自动调用

@tool
def multiply(a: int, b: int) -> int:
    """乘法运算：计算两个整数的乘积
    
    Args:
        a: 第一个整数
        b: 第二个整数
    
    Returns:
        两个数的乘积
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """加法运算：计算两个整数的和
    
    Args:
        a: 第一个整数
        b: 第二个整数
    
    Returns:
        两个数的和
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """除法运算：计算两个整数的商
    
    Args:
        a: 被除数
        b: 除数
    
    Returns:
        两个数的商（浮点数）
    """
    return a / b


# 将工具绑定到模型
# 这样模型就可以在需要时自动调用这些工具
tools = [add, multiply, divide]  # 工具列表
tools_by_name = {tool.name: tool for tool in tools}  # 通过名称快速查找工具的字典
model_with_tools = model.bind_tools(tools)  # 将工具绑定到模型，使模型具备工具调用能力

# ==================== 步骤 2: 定义状态结构 ====================

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    """Agent 的状态定义
    
    状态包含：
    - messages: 消息列表，使用 operator.add 实现消息的累积（新消息追加到列表）
    - llm_calls: LLM 调用次数计数器，用于统计和调试
    """
    messages: Annotated[list[AnyMessage], operator.add]  # 消息列表，支持消息累积
    llm_calls: int  # LLM 调用次数

# ==================== 步骤 3: 定义 LLM 调用节点 ====================

from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM 调用节点：分析用户输入并决定是否需要调用工具
    
    这个函数是 Agent 的核心节点之一，负责：
    1. 接收当前状态（包含用户消息和历史对话）
    2. 调用 LLM 分析输入
    3. LLM 会决定是直接回复还是调用工具
    
    Args:
        state: 当前状态字典，包含 messages 和 llm_calls
        
    Returns:
        包含 LLM 响应的新状态：
        - messages: LLM 的回复消息（可能包含工具调用请求）
        - llm_calls: 调用次数加 1
    """
    return {
        "messages": [
            # 调用绑定了工具的模型
            model_with_tools.invoke(
                [
                    # 系统提示词：告诉 LLM 它的角色和任务
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]  # 追加历史消息和用户输入
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1  # 增加调用计数
    }


# ==================== 步骤 4: 定义工具执行节点 ====================

from langchain.messages import ToolMessage


def tool_node(state: dict):
    """工具执行节点：执行 LLM 请求的工具调用
    
    当 LLM 决定调用工具时，会进入这个节点：
    1. 从状态中获取 LLM 的工具调用请求
    2. 根据工具名称找到对应的工具函数
    3. 执行工具并获取结果
    4. 将结果封装成 ToolMessage 返回给 LLM
    
    Args:
        state: 当前状态字典，包含 LLM 的工具调用请求
        
    Returns:
        包含工具执行结果的新状态：
        - messages: ToolMessage 列表，包含每个工具的执行结果
    """
    result = []
    # 遍历 LLM 在最后一条消息中请求的所有工具调用
    for tool_call in state["messages"][-1].tool_calls:
        # 根据工具名称从字典中获取对应的工具函数
        tool = tools_by_name[tool_call["name"]]
        # 使用工具调用中的参数执行工具
        observation = tool.invoke(tool_call["args"])
        # 将执行结果封装成 ToolMessage，关联到对应的 tool_call_id
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# ==================== 步骤 5: 定义条件路由逻辑 ====================

from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """条件路由函数：根据 LLM 的输出决定下一步流程
    
    这是 Agent 的决策核心：
    - 如果 LLM 的输出包含工具调用请求，则路由到 tool_node 执行工具
    - 如果 LLM 直接回复用户，则路由到 END 结束流程
    
    Args:
        state: 当前状态，包含消息列表
        
    Returns:
        "tool_node": 如果需要执行工具
        END: 如果可以直接回复用户，结束流程
    """
    messages = state["messages"]
    last_message = messages[-1]  # 获取最后一条消息（LLM 的回复）

    # 如果 LLM 的回复中包含工具调用请求，则执行工具
    if last_message.tool_calls:
        return "tool_node"

    # 否则，LLM 已经给出最终答案，结束流程
    return END

# ==================== 步骤 6: 构建 Agent 工作流 ====================

# 创建状态图构建器，指定状态类型
agent_builder = StateGraph(MessagesState)

# 添加节点：定义工作流中的各个处理步骤
agent_builder.add_node("llm_call", llm_call)      # LLM 调用节点
agent_builder.add_node("tool_node", tool_node)    # 工具执行节点

# 添加边：定义节点之间的连接关系
# 1. 从 START 开始，首先进入 llm_call 节点
agent_builder.add_edge(START, "llm_call")

# 2. 从 llm_call 节点根据条件路由：
#    - 如果需要工具：路由到 tool_node
#    - 如果不需要：路由到 END（结束）
agent_builder.add_conditional_edges(
    "llm_call",           # 源节点
    should_continue,      # 路由决策函数
    ["tool_node", END]    # 可能的目标节点
)

# 3. 工具执行完成后，再次回到 llm_call 节点（让 LLM 处理工具结果）
agent_builder.add_edge("tool_node", "llm_call")

# 编译 Agent：将构建的图结构编译成可执行的 Agent
agent = agent_builder.compile()


# ==================== 步骤 7: 可视化 Agent 结构（可选）====================

from IPython.display import Image, display
# 显示 Agent 的工作流图（仅在 Jupyter 环境中有效）
#display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# ==================== 步骤 8: 执行 Agent ====================

from langchain.messages import HumanMessage

# 创建用户消息
messages = [HumanMessage(content="Add 3 and 4.")]

# 调用 Agent 处理用户输入
# Agent 会自动执行：LLM 分析 -> 工具调用 -> LLM 整合结果 -> 返回答案
messages = agent.invoke({"messages": messages})

# 打印所有消息（包括用户输入、工具调用、LLM 回复等）
for m in messages["messages"]:
    m.pretty_print()