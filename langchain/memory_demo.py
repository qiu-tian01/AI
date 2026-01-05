"""
LangChain 1.0 短期记忆和长期记忆 Demo

本示例展示了如何在 LangChain 1.0 中使用：
1. 短期记忆（Short-term Memory）：对话缓冲区，保持当前会话的上下文
2. 长期记忆（Long-term Memory）：向量数据库存储，持久化保存历史对话信息

使用方法：
1. 确保已安装所需依赖：pip install langchain langchain-community langchain-dashscope chromadb
2. 设置环境变量 DASHSCOPE_API_KEY
3. 运行：python memory_demo.py
"""

import os
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.schema import Document
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# 加载环境变量
load_dotenv()

# ==================== 配置 ====================

# 初始化模型
llm = ChatTongyi(
    model_name="qwen-plus",
    temperature=0.7,
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 初始化嵌入模型（用于长期记忆的向量化）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# ==================== 长期记忆存储 ====================

# 使用 Chroma 作为向量数据库存储长期记忆
# 持久化目录：./chroma_db
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="conversation_memory"
)


def save_to_long_term_memory(user_message: str, ai_response: str, metadata: Dict[str, Any] = None):
    """
    将对话保存到长期记忆（向量数据库）
    
    Args:
        user_message: 用户消息
        ai_response: AI回复
        metadata: 可选的元数据（如时间戳、用户ID等）
    """
    # 将对话组合成文档
    conversation_text = f"用户: {user_message}\n助手: {ai_response}"
    
    # 创建文档对象
    doc = Document(
        page_content=conversation_text,
        metadata=metadata or {}
    )
    
    # 添加到向量数据库
    vectorstore.add_documents([doc])
    print(f"✅ 已保存到长期记忆: {conversation_text[:50]}...")


def search_long_term_memory(query: str, k: int = 3) -> List[str]:
    """
    从长期记忆中检索相关对话
    
    Args:
        query: 搜索查询
        k: 返回的结果数量
    
    Returns:
        相关对话列表
    """
    # 使用相似度搜索
    docs = vectorstore.similarity_search(query, k=k)
    
    # 提取对话内容
    results = [doc.page_content for doc in docs]
    
    if results:
        print(f"📚 从长期记忆中检索到 {len(results)} 条相关对话")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result[:100]}...")
    else:
        print("📚 长期记忆中没有找到相关对话")
    
    return results


# ==================== 工具定义 ====================

@tool
def search_memory(query: str) -> str:
    """
    搜索长期记忆中的相关对话
    
    Args:
        query: 搜索关键词或问题
    
    Returns:
        检索到的相关对话信息
    """
    results = search_long_term_memory(query, k=3)
    
    if results:
        return "\n\n".join([f"相关对话 {i+1}:\n{result}" for i, result in enumerate(results)])
    else:
        return "没有找到相关的历史对话记录。"


@tool
def get_user_info(name: str) -> str:
    """
    获取用户信息（模拟工具）
    
    Args:
        name: 用户名
    
    Returns:
        用户信息
    """
    return f"用户 {name} 的信息：这是一个示例用户。"


# ==================== Agent 创建 ====================

# 系统提示词
SYSTEM_PROMPT = """你是一个友好的AI助手，具有记忆功能。

你的能力：
1. 短期记忆：记住当前对话中的所有内容
2. 长期记忆：可以搜索和回忆之前保存的对话历史

使用指南：
- 当用户询问之前提到过的事情时，可以使用 search_memory 工具搜索长期记忆
- 对于当前对话中的信息，直接使用短期记忆（对话历史）
- 始终用中文回复用户
- 如果用户提到重要信息，可以主动保存到长期记忆中

记住：短期记忆用于当前会话，长期记忆用于跨会话的信息检索。"""

# 工具列表
tools = [search_memory, get_user_info]

# 创建短期记忆（MemorySaver）- 用于保持当前会话的上下文
short_term_memory = MemorySaver()

# 创建 Agent
agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    checkpointer=short_term_memory  # 短期记忆：保存当前会话状态
)


# ==================== 对话管理 ====================

def chat_with_memory(user_input: str, thread_id: str = "default", save_to_long_term: bool = False):
    """
    与 Agent 对话，同时管理短期和长期记忆
    
    Args:
        user_input: 用户输入
        thread_id: 对话线程ID（用于区分不同会话）
        save_to_long_term: 是否保存到长期记忆
    """
    print(f"\n{'='*60}")
    print(f"👤 用户: {user_input}")
    print(f"{'='*60}\n")
    
    # 配置对话上下文（使用 thread_id 区分不同会话）
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    # 调用 Agent（自动使用短期记忆）
    response = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    # 获取 AI 回复
    ai_message = response["messages"][-1].content
    print(f"🤖 助手: {ai_message}\n")
    
    # 如果需要，保存到长期记忆
    if save_to_long_term:
        save_to_long_term_memory(
            user_input,
            ai_message,
            metadata={
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    return ai_message


# ==================== 示例用法 ====================

def demo_short_term_memory():
    """演示短期记忆功能"""
    print("\n" + "="*60)
    print("📝 演示 1: 短期记忆（当前会话上下文）")
    print("="*60)
    
    # 第一次对话
    chat_with_memory("我的名字是张三，我喜欢编程。", thread_id="session_1")
    
    # 第二次对话（应该能记住名字）
    chat_with_memory("我刚才说我叫什么名字？", thread_id="session_1")
    
    # 第三次对话（应该能记住爱好）
    chat_with_memory("我的爱好是什么？", thread_id="session_1")
    
    print("\n💡 说明：短期记忆在同一 thread_id 的会话中保持上下文")


def demo_long_term_memory():
    """演示长期记忆功能"""
    print("\n" + "="*60)
    print("📚 演示 2: 长期记忆（跨会话信息检索）")
    print("="*60)
    
    # 第一次会话：保存信息到长期记忆
    print("\n--- 会话 A：保存信息 ---")
    chat_with_memory(
        "我想学习 Python 编程，有什么建议吗？",
        thread_id="session_A",
        save_to_long_term=True
    )
    
    chat_with_memory(
        "Python 很适合初学者，建议从基础语法开始学习。",
        thread_id="session_A",
        save_to_long_term=True
    )
    
    # 第二次会话：检索长期记忆
    print("\n--- 会话 B：检索信息 ---")
    chat_with_memory(
        "我之前问过关于学习编程的问题吗？",
        thread_id="session_B"
    )
    
    # 第三次会话：使用工具搜索长期记忆
    print("\n--- 会话 C：主动搜索长期记忆 ---")
    chat_with_memory(
        "搜索一下我之前关于 Python 的对话",
        thread_id="session_C"
    )
    
    print("\n💡 说明：长期记忆可以跨会话检索历史信息")


def demo_combined_memory():
    """演示短期记忆和长期记忆的结合使用"""
    print("\n" + "="*60)
    print("🔄 演示 3: 短期记忆 + 长期记忆结合使用")
    print("="*60)
    
    # 在当前会话中使用短期记忆
    chat_with_memory("我今天心情很好。", thread_id="combined_session")
    
    # 同时检索长期记忆
    chat_with_memory("我之前有没有提到过学习编程的事情？", thread_id="combined_session")
    
    # 继续使用短期记忆
    chat_with_memory("我刚才说我心情怎么样？", thread_id="combined_session")
    
    print("\n💡 说明：短期记忆处理当前会话，长期记忆处理历史信息")


def main():
    """主函数：运行所有演示"""
    print("\n" + "="*60)
    print("🚀 LangChain 1.0 记忆系统演示")
    print("="*60)
    print("\n本演示将展示：")
    print("1. 短期记忆：在同一会话中保持上下文")
    print("2. 长期记忆：跨会话检索历史信息")
    print("3. 两者结合：同时使用短期和长期记忆")
    
    try:
        # 运行演示
        demo_short_term_memory()
        demo_long_term_memory()
        demo_combined_memory()
        
        print("\n" + "="*60)
        print("✅ 演示完成！")
        print("="*60)
        print("\n📖 使用说明：")
        print("1. 短期记忆（MemorySaver）：")
        print("   - 使用 checkpointer 参数配置")
        print("   - 通过 thread_id 区分不同会话")
        print("   - 在同一会话中自动保持上下文")
        print("\n2. 长期记忆（向量数据库）：")
        print("   - 使用 Chroma 向量数据库存储")
        print("   - 通过 save_to_long_term_memory() 保存")
        print("   - 通过 search_long_term_memory() 检索")
        print("   - 可以跨会话访问历史信息")
        print("\n3. 结合使用：")
        print("   - Agent 自动使用短期记忆")
        print("   - 通过工具调用访问长期记忆")
        print("   - 两者互补，提供完整的记忆能力")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

