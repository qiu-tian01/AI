import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi

# 加载 .env 文件中的环境变量
load_dotenv()

# 配置流式模型 - 使用 DashScope 的 qwen-plus 模型
# ChatTongyi 会自动从环境变量 DASHSCOPE_API_KEY 读取 API key
streaming_model = ChatTongyi(
    model_name="qwen-plus",  # DashScope qwen-plus 模型
    temperature=0,  # 控制随机性
    streaming=True  # 启用流式输出
)

# 创建 agent（可以根据需要添加工具和系统提示）
agent = create_agent(
    model=streaming_model,
    tools=[],  # 可以添加工具，参考 index.py
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

# 流式 invoke 函数
def stream_invoke(messages, config=None):
    """
    流式调用 agent
    
    Args:
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
        config: 可选的配置字典，例如 {"configurable": {"thread_id": "1"}}
    
    Yields:
        流式响应的每个 chunk
    """
    if config is None:
        config = {"configurable": {"thread_id": "default"}}
    
    # 使用 stream() 方法进行流式调用
    for chunk in agent.stream(
        {"messages": messages},
        config=config
    ):
        yield chunk


# 辅助函数：从 chunk 中提取文本内容
def extract_content_from_chunk(chunk):
    """
    从流式响应的 chunk 中提取文本内容
    
    Args:
        chunk: 流式响应的 chunk
    
    Returns:
        提取的文本内容（如果有）
    """
    if isinstance(chunk, dict):
        # 检查不同的可能字段
        for key in ['messages', 'agent', 'content']:
            if key in chunk:
                value = chunk[key]
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            if 'content' in item and item['content']:
                                return item['content']
                        elif hasattr(item, 'content'):
                            content = item.content
                            if content:
                                return content
                elif isinstance(value, dict):
                    if 'content' in value and value['content']:
                        return value['content']
                elif isinstance(value, str) and value:
                    return value
        # 如果直接包含 content 字段
        if 'content' in chunk and chunk['content']:
            return chunk['content']
    elif hasattr(chunk, 'content'):
        content = chunk.content
        if content:
            return content
    
    return None


# 示例使用
if __name__ == "__main__":
    print("=" * 60)
    print("流式调用示例 - 使用 qwen-plus 模型")
    print("=" * 60)
    
    # 准备消息
    messages = [{"role": "user", "content": "请介绍一下人工智能的发展历史，控制在200字以内"}]
    
    # 配置（可选）
    config = {"configurable": {"thread_id": "stream-1"}}
    
    print("\n开始流式输出：\n")
    print("-" * 60)
    
    # 流式调用并打印结果
    full_response = ""
    chunk_count = 0
    
    try:
        for chunk in stream_invoke(messages, config):
            chunk_count += 1
            content = extract_content_from_chunk(chunk)
            
            if content:
                print(content, end='', flush=True)
                full_response += content
            else:
                # 如果没有提取到内容，打印 chunk 的结构（用于调试）
                if chunk_count <= 3:  # 只打印前几个 chunk 的结构
                    print(f"\n[调试信息] Chunk {chunk_count} 结构: {chunk}")
        
        print("\n" + "-" * 60)
        print(f"\n流式输出完成！共处理 {chunk_count} 个 chunk")
        print(f"完整响应长度: {len(full_response)} 字符")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示:")
        print("1. 确保已设置 DASHSCOPE_API_KEY 环境变量")
        print("2. 检查网络连接")
        print("3. 确认 qwen-plus 模型可用")

