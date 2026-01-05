import os
import json
import time
import requests
from typing import Optional
from dotenv import load_dotenv
import dashscope
from dashscope import ImageSynthesis

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()

# 配置 DashScope API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ==================== 模型初始化 ====================

# 创建提示词优化模型（独立实例，避免与主 Agent 模型冲突）
optimizer_model = ChatTongyi(
    model_name="qwen-plus",
    temperature=0.7
)

# ==================== 工具函数定义 ====================

@tool
def optimize_prompt(user_description: str, style: Optional[str] = None) -> str:
    """
    优化用户的文字描述，增强提示词的艺术性和细节描述。
    
    Args:
        user_description: 用户输入的原始文字描述
        style: 图片风格（可选），如 'realistic', 'anime', '3d', 'oil-painting' 等
    
    Returns:
        优化后的提示词（英文）
    """
    
    style_prompt = f" in {style} style" if style else ""
    
    optimization_instruction = f"""你是一个专业的AI绘画提示词优化专家。请将用户的中文描述优化为详细的英文提示词，用于AI图片生成。

要求：
1. 保留用户描述的核心内容
2. 添加艺术性的细节描述（如光线、色彩、构图、氛围等）
3. 使用专业的艺术术语
4. 确保提示词清晰、具体、富有表现力
5. 如果用户输入是中文，请翻译成英文
6. 如果指定了风格（{style if style else '无'}），请在提示词中体现该风格

用户描述：{user_description}
风格：{style if style else '无特定风格'}

请只返回优化后的英文提示词，不要添加其他解释："""
    
    try:
        response = optimizer_model.invoke([
            {"role": "user", "content": optimization_instruction}
        ])
        
        optimized_prompt = response.content.strip()
        print(f"优化后的提示词: {optimized_prompt}")
        return optimized_prompt
    except Exception as e:
        # 如果优化失败，返回原始描述（尝试翻译）
        print(f"提示词优化失败: {e}")
        return user_description


def download_image(url: str, save_path: str) -> bool:
    """
    下载图片并保存到指定路径
    
    Args:
        url: 图片URL
        save_path: 保存路径
    
    Returns:
        是否下载成功
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        # 下载失败，返回False，错误信息会在上层处理
        return False

def normalize_style(style: str) -> str:
    """
    规范化风格参数，确保符合API要求
    
    Args:
        style: 用户输入的风格
    
    Returns:
        规范化后的风格（带尖括号）
    """
    if not style:
        return '<auto>'
    
    # 移除可能的尖括号
    style = style.strip().strip('<>')
    
    # 风格映射表（用户友好名称 -> API格式）
    style_mapping = {
        'auto': '<auto>',
        'anime': '<anime>',
        '3d cartoon': '<3d cartoon>',
        '3d-cartoon': '<3d cartoon>',
        'oil painting': '<oil painting>',
        'oil-painting': '<oil painting>',
        'watercolor': '<watercolor>',
        'water-color': '<watercolor>',
        'sketch': '<sketch>',
        'chinese painting': '<chinese painting>',
        'chinese-painting': '<chinese painting>',
        'flat illustration': '<flat illustration>',
        'flat-illustration': '<flat illustration>',
        'flat-pattern': '<flat illustration>',
        'photography': '<photography>',
        'portrait': '<portrait>',
    }
    
    # 转换为小写进行匹配
    style_lower = style.lower()
    
    # 查找映射
    if style_lower in style_mapping:
        return style_mapping[style_lower]
    
    # 如果已经包含尖括号，直接返回
    if style.startswith('<') and style.endswith('>'):
        return style
    
    # 否则尝试添加尖括号
    return f'<{style}>'

def poll_task_status(task_id: str, max_wait_time: int = 300) -> dict:
    """
    轮询任务状态直到完成
    
    Args:
        task_id: 任务ID
        max_wait_time: 最大等待时间（秒）
    
    Returns:
        任务结果
    """
    start_time = time.time()
    poll_count = 0
    while time.time() - start_time < max_wait_time:
        try:
            poll_count += 1
            elapsed_time = int(time.time() - start_time)
            
            # 尝试使用 fetch 方法（如果存在）
            if hasattr(ImageSynthesis, 'fetch'):
                result = ImageSynthesis.fetch(task_id=task_id)
            else:
                # 如果没有 fetch 方法，尝试使用 call 方法查询
                # 注意：这可能需要根据实际 API 调整
                result = ImageSynthesis.call(
                    model=ImageSynthesis.Models.wanx_v1,
                    task_id=task_id
                )
            
            if result.status_code == 200:
                output = result.output
                if output:
                    task_status = output.get('task_status', '')
                    # 检查是否有 results 字段（表示任务完成）
                    if 'results' in output and output['results']:
                        return {"status": "success", "output": output}
                    elif task_status == 'SUCCEEDED':
                        return {"status": "success", "output": output}
                    elif task_status == 'FAILED':
                        return {"status": "failed", "message": output.get('message', '任务失败')}
                    # 任务还在处理中，继续等待
                    if poll_count % 10 == 0:  # 每10次轮询显示一次进度
                        print(f"任务处理中... (已等待 {elapsed_time} 秒)")
                    time.sleep(3)
                else:
                    time.sleep(3)
            else:
                time.sleep(3)
        except AttributeError:
            # 如果 fetch 方法不存在，可能需要直接返回，让调用者处理
            return {"status": "no_polling", "task_id": task_id, "message": "API不支持轮询"}
        except Exception as e:
            # 查询失败，继续重试
            time.sleep(3)
    
    return {"status": "timeout", "message": f"任务超时（已等待 {max_wait_time} 秒）", "task_id": task_id}

@tool
def generate_image(
    prompt: str,
    style: Optional[str] = None,
    size: Optional[str] = None,
    n: int = 1
) -> str:
    """
    使用 DashScope Wanx API 生成图片。
    
    Args:
        prompt: 图片生成的提示词（英文，建议使用优化后的提示词）
        style: 图片风格，可选值：
               - '<auto>' (自动)
               - '<anime>' (动漫)
               - '<3d cartoon>' (3D卡通)
               - '<oil painting>' (油画)
               - '<watercolor>' (水彩)
               - '<sketch>' (素描)
               - '<chinese painting>' (中国画)
               - '<flat illustration>' (扁平插画)
               - '<photography>' (摄影)
               - '<portrait>' (肖像)
               也支持不带尖括号的格式，会自动转换
        size: 图片尺寸，可选值：'1024*1024', '720*1280', '1280*720', '768*1344', 
              '1344*768', '768*768', '384*640', '640*384'
        n: 生成图片数量，默认为1，最多4张
    
    Returns:
        JSON字符串，包含图片URL列表、任务ID和本地保存路径
    """
    try:
        # 设置默认参数并规范化风格
        if style is None:
            style = '<auto>'
        else:
            style = normalize_style(style)
        
        if size is None:
            size = '1024*1024'
        
        # 调用 DashScope Wanx API
        result = ImageSynthesis.call(
            model=ImageSynthesis.Models.wanx_v1,
            prompt=prompt,
            n=n,
            size=size,
            style=style
        )
        
        if result.status_code == 200:
            print(f"图片生成result: {result}")
            output = result.output
            if not output:
                return json.dumps({
                    "success": False,
                    "error": "API返回空结果"
                }, ensure_ascii=False)
            
            task_id = output.get('task_id', '')
            task_status = output.get('task_status', '')
            
            # 检查任务是否失败
            if task_status == 'FAILED':
                error_msg = f"图片生成任务失败"
                if output.get('code'):
                    error_msg += f"\n错误代码: {output.get('code')}"
                if output.get('message'):
                    error_msg += f"\n错误详情: {output.get('message')}"
                return json.dumps({
                    "success": False,
                    "error": error_msg,
                    "task_id": task_id,
                    "task_status": task_status
                }, ensure_ascii=False)
            
            # 检查是否直接返回了结果
            image_urls = []
            if 'results' in output and output['results']:
                # 直接返回了结果
                for item in output['results']:
                    if isinstance(item, dict) and 'url' in item:
                        image_urls.append(item['url'])
            
            # 如果没有图片URL但有task_id，尝试轮询任务状态
            if not image_urls and task_id:
                poll_result = poll_task_status(task_id)
                if poll_result['status'] == 'success':
                    output = poll_result['output']
                    if 'results' in output and output['results']:
                        for item in output['results']:
                            if isinstance(item, dict) and 'url' in item:
                                image_urls.append(item['url'])
                elif poll_result['status'] == 'no_polling':
                    # API不支持轮询，返回task_id让用户稍后查询
                    return json.dumps({
                        "success": False,
                        "error": f"任务已提交（任务ID: {task_id}），但API不支持自动轮询，请稍后查询任务状态",
                        "task_id": task_id
                    }, ensure_ascii=False)
                elif poll_result['status'] == 'failed':
                    return json.dumps({
                        "success": False,
                        "error": f"图片生成失败: {poll_result.get('message', '未知错误')}"
                    }, ensure_ascii=False)
                elif poll_result['status'] == 'timeout':
                    return json.dumps({
                        "success": False,
                        "error": f"图片生成超时（任务ID: {task_id}），请稍后查询任务状态",
                        "task_id": task_id
                    }, ensure_ascii=False)
            
            # 如果仍然没有图片URL，返回错误
            if not image_urls:
                return json.dumps({
                    "success": False,
                    "error": f"图片生成完成但没有返回图片URL。任务ID: {task_id}。输出内容: {json.dumps(output, ensure_ascii=False)}",
                    "task_id": task_id,
                    "output": output
                }, ensure_ascii=False)
            
            # 下载图片到本地
            saved_paths = []
            if image_urls:
                # 确保 imgs 文件夹存在
                imgs_dir = os.path.join(os.path.dirname(__file__), 'imgs')
                os.makedirs(imgs_dir, exist_ok=True)
                
                # 下载每张图片
                for idx, url in enumerate(image_urls):
                    timestamp = int(time.time())
                    filename = f"image_{timestamp}_{idx + 1}.png"
                    save_path = os.path.join(imgs_dir, filename)
                    
                    if download_image(url, save_path):
                        saved_paths.append(save_path)
            
            response_data = {
                "success": True,
                "image_urls": image_urls,
                "saved_paths": saved_paths,
                "task_id": task_id,
                "prompt": prompt,
                "style": style,
                "size": size
            }
            return json.dumps(response_data, ensure_ascii=False)
        else:
            error_msg = f"图片生成失败: {result.message}"
            if result.code:
                error_msg += f" (错误代码: {result.code})"
            
            # 检查输出中的详细错误信息
            if result.output:
                output = result.output
                if output.get('code'):
                    error_msg += f"\n错误代码: {output.get('code')}"
                if output.get('message'):
                    error_msg += f"\n错误详情: {output.get('message')}"
                if output.get('task_status') == 'FAILED':
                    error_msg += f"\n任务状态: {output.get('task_status')}"
            
            return json.dumps({
                "success": False,
                "error": error_msg
            }, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"图片生成异常: {str(e)}"
        }, ensure_ascii=False)


def _query_task_status_impl(task_id: str) -> str:
    """
    查询图片生成任务的状态（内部实现函数）。
    
    Args:
        task_id: 任务ID（从之前的生成请求中获得）
    
    Returns:
        JSON字符串，包含任务状态和图片URL（如果已完成）
    """
    try:
        # 尝试使用 fetch 方法（如果存在）
        if hasattr(ImageSynthesis, 'fetch'):
            result = ImageSynthesis.fetch(task_id=task_id)
        else:
            # 如果没有 fetch 方法，尝试使用 call 方法查询
            result = ImageSynthesis.call(
                model=ImageSynthesis.Models.wanx_v1,
                task_id=task_id
            )
        
        if result.status_code == 200:
            output = result.output
            if output:
                task_status = output.get('task_status', '')
                image_urls = []
                
                # 提取图片URL
                if 'results' in output and output['results']:
                    for item in output['results']:
                        if isinstance(item, dict) and 'url' in item:
                            image_urls.append(item['url'])
                
                # 下载图片到本地
                saved_paths = []
                if image_urls:
                    imgs_dir = os.path.join(os.path.dirname(__file__), 'imgs')
                    os.makedirs(imgs_dir, exist_ok=True)
                    
                    for idx, url in enumerate(image_urls):
                        timestamp = int(time.time())
                        filename = f"image_{timestamp}_{idx + 1}.png"
                        save_path = os.path.join(imgs_dir, filename)
                        
                        if download_image(url, save_path):
                            saved_paths.append(save_path)
                
                if task_status == 'SUCCEEDED' or image_urls:
                    return json.dumps({
                        "success": True,
                        "task_status": task_status or "SUCCEEDED",
                        "image_urls": image_urls,
                        "saved_paths": saved_paths,
                        "task_id": task_id
                    }, ensure_ascii=False)
                elif task_status == 'FAILED':
                    return json.dumps({
                        "success": False,
                        "task_status": "FAILED",
                        "error": output.get('message', '任务失败'),
                        "task_id": task_id
                    }, ensure_ascii=False)
                else:
                    return json.dumps({
                        "success": False,
                        "task_status": task_status or "PROCESSING",
                        "message": "任务仍在处理中，请稍后再试",
                        "task_id": task_id
                    }, ensure_ascii=False)
            else:
                return json.dumps({
                    "success": False,
                    "error": "无法获取任务状态",
                    "task_id": task_id
                }, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error": f"查询任务状态失败: {result.message}",
                "task_id": task_id
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"查询任务状态异常: {str(e)}",
            "task_id": task_id
        }, ensure_ascii=False)

@tool
def query_task_status(task_id: str) -> str:
    """
    查询图片生成任务的状态。
    
    Args:
        task_id: 任务ID（从之前的生成请求中获得）
    
    Returns:
        JSON字符串，包含任务状态和图片URL（如果已完成）
    """
    return _query_task_status_impl(task_id)

@tool
def validate_image(image_url: str, original_prompt: str) -> str:
    """
    验证生成的图片是否符合用户需求。
    
    Args:
        image_url: 生成的图片URL
        original_prompt: 原始的用户提示词
    
    Returns:
        验证结果和建议（JSON格式）
    """
    # 这里可以实现图片验证逻辑
    # 例如：使用视觉模型分析图片内容，检查是否匹配提示词
    # 目前返回简单的验证结果
    
    validation_result = {
        "image_url": image_url,
        "original_prompt": original_prompt,
        "validation_status": "pending",
        "suggestions": [
            "图片已生成，请查看是否符合您的需求",
            "如需调整，可以提供更详细的描述或指定不同的风格"
        ]
    }
    
    return json.dumps(validation_result, ensure_ascii=False)


# ==================== 模型配置 ====================

# 配置 Qwen 模型
qwen_model = ChatTongyi(
    model_name="qwen-plus",
    temperature=0.7,  # 适中的温度，保持创造性和准确性
    streaming=False  # 可以根据需要设置为 True
)

# 修复 ChatTongyi 的 tool_choice 参数问题（参考 index.py）
original_generate = qwen_model._generate

def patched_generate(self, messages, stop=None, run_manager=None, **kwargs):
    """修复 tool_choice 参数，确保兼容 DashScope API"""
    if 'tool_choice' in kwargs:
        tool_choice = kwargs['tool_choice']
        if tool_choice not in ['auto', 'none']:
            kwargs['tool_choice'] = 'auto'
    return original_generate(messages, stop=stop, run_manager=run_manager, **kwargs)

qwen_model._generate = patched_generate.__get__(qwen_model, type(qwen_model))

# ==================== Agent Flow 配置 ====================

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的AI图片生成助手。你的任务是帮助用户根据文字描述生成高质量的图片。

工作流程：
1. 当用户提供文字描述时，首先使用 optimize_prompt 工具优化提示词，增强艺术性和细节描述
2. 然后使用 generate_image 工具生成图片
3. （可选）使用 validate_image 工具验证生成的图片是否符合需求
4. 最后将结果清晰地呈现给用户

重要提示：
- 如果用户没有指定风格，你可以根据描述内容智能推荐合适的风格
- 生成的图片URL可以直接在浏览器中打开查看
- 如果生成失败，请分析原因并提供改进建议
- 始终用中文回复用户

可用工具：
- optimize_prompt: 优化用户的文字描述为专业的AI绘画提示词
- generate_image: 使用 DashScope Wanx API 生成图片
- validate_image: 验证生成的图片是否符合需求
- query_task_status: 查询图片生成任务的状态（如果生成超时，可以使用此工具查询任务状态）"""

# 工具列表
tools = [optimize_prompt, generate_image, validate_image, query_task_status]

# 设置记忆
checkpointer = InMemorySaver()

# 创建 Agent
agent = create_agent(
    model=qwen_model,
    system_prompt=SYSTEM_PROMPT,
    tools=tools,
    checkpointer=checkpointer
)

# ==================== 主程序 ====================

def generate_image_from_text(user_input: str, style: Optional[str] = None, thread_id: str = "default") -> dict:
    """
    根据用户输入的文字描述生成图片。
    
    Args:
        user_input: 用户的文字描述
        style: 图片风格（可选）
        thread_id: 对话线程ID，用于保持上下文
    
    Returns:
        Agent 的响应结果
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # 构建用户消息
    user_message = user_input
    if style:
        user_message += f"（风格：{style}）"
    
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config
        )
        return response
    except Exception as e:
        return {
            "error": f"生成图片时发生错误: {str(e)}",
            "suggestion": "请检查 DASHSCOPE_API_KEY 是否正确配置"
        }


def stream_generate_image(user_input: str, style: Optional[str] = None, thread_id: str = "default"):
    """
    流式生成图片（支持流式输出）。
    
    Args:
        user_input: 用户的文字描述
        style: 图片风格（可选）
        thread_id: 对话线程ID
    
    Yields:
        流式响应的每个 chunk
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    user_message = user_input
    if style:
        user_message += f"（风格：{style}）"
    
    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config
        ):
            yield chunk
    except Exception as e:
        yield {"error": f"生成图片时发生错误: {str(e)}"}


# ==================== 辅助函数 ====================

def get_message_type(msg):
    """获取消息类型，兼容字典和 LangChain 消息对象"""
    if isinstance(msg, dict):
        return msg.get("role") or msg.get("type")
    return getattr(msg, 'type', None)

def get_message_content(msg):
    """获取消息内容，兼容字典和 LangChain 消息对象"""
    if isinstance(msg, dict):
        return msg.get('content', '')
    return getattr(msg, 'content', '')

def is_assistant_message(msg):
    """判断是否为助手消息"""
    msg_type = get_message_type(msg)
    return msg_type == "ai" or msg_type == "assistant"

def safe_get(obj, key, default=None):
    """安全地获取对象或字典的值"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def safe_has_key(obj, key):
    """安全地检查对象或字典是否包含键"""
    if isinstance(obj, dict):
        return key in obj
    return hasattr(obj, key)

def extract_image_info_from_messages(messages):
    """从消息中提取图片生成信息"""
    import re
    image_info_list = []
    for msg in messages:
        content = get_message_content(msg)
        if not content:
            continue
        
        # 方法1: 检查是否是工具调用结果消息（ToolMessage）
        msg_type = get_message_type(msg)
        if msg_type == "tool" or (isinstance(msg, dict) and msg.get("type") == "tool"):
            # 工具调用结果通常直接包含 JSON
            try:
                if content.strip().startswith('{'):
                    result_json = json.loads(content)
                    if result_json.get('success') and result_json.get('saved_paths'):
                        image_info_list.append(result_json)
                        continue
            except:
                pass
        
        # 方法2: 检查消息内容中是否包含 JSON 格式的图片信息
        # 查找包含 "saved_paths" 的 JSON 对象
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"saved_paths"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        for match in matches:
            try:
                result_json = json.loads(match)
                if result_json.get('success') and result_json.get('saved_paths'):
                    image_info_list.append(result_json)
            except:
                pass
        
        # 方法3: 尝试解析整个内容为 JSON（如果可能）
        try:
            if content.strip().startswith('{'):
                result_json = json.loads(content)
                if result_json.get('success') and result_json.get('saved_paths'):
                    image_info_list.append(result_json)
        except:
            pass
    
    return image_info_list

# ==================== 示例和测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("基于 LangChain 1.0 和 Qwen 的文字生成图片 Agent Flow")
    print("=" * 60)
    print()
    
    # 检查 API Key
    if not dashscope.api_key:
        print("错误: 未找到 DASHSCOPE_API_KEY 环境变量")
        print("请在 .env 文件中设置: DASHSCOPE_API_KEY=your-api-key")
        exit(1)
    
    # 支持的风格列表（API实际支持的风格）
    supported_styles = [
        '<auto>', '<anime>', '<3d cartoon>', '<oil painting>', '<watercolor>',
        '<sketch>', '<chinese painting>', '<flat illustration>', '<photography>', '<portrait>'
    ]
    
    # 用户友好的风格名称映射
    style_aliases = {
        'auto': '<auto>',
        'anime': '<anime>',
        '3d cartoon': '<3d cartoon>',
        '3d-cartoon': '<3d cartoon>',
        'oil painting': '<oil painting>',
        'oil-painting': '<oil painting>',
        'watercolor': '<watercolor>',
        'water-color': '<watercolor>',
        'sketch': '<sketch>',
        'chinese painting': '<chinese painting>',
        'chinese-painting': '<chinese painting>',
        'flat illustration': '<flat illustration>',
        'flat-illustration': '<flat illustration>',
        'flat-pattern': '<flat illustration>',
        'photography': '<photography>',
        'portrait': '<portrait>',
    }
    
    # 交互式用户输入
    thread_id_counter = 1
    
    while True:
        print("\n" + "=" * 60)
        print("请选择操作模式:")
        print("1. 普通模式生成图片")
        print("2. 指定风格生成图片")
        print("3. 流式输出模式")
        print("4. 查询任务状态（输入任务ID）")
        print("0. 退出")
        print("=" * 60)
        
        choice = input("\n请输入选项 (0-3): ").strip()
        
        if choice == "0":
            print("\n感谢使用！再见！")
            break
        elif choice not in ["1", "2", "3", "4"]:
            print("\n无效选项，请重新选择！")
            continue
        
        # 获取用户输入
        user_input = input("\n请输入图片描述（中文或英文）: ").strip()
        if not user_input:
            print("\n输入不能为空，请重新输入！")
            continue
        
        style = None
        if choice == "2":
            print("\n支持的风格:")
            print("  " + ", ".join(supported_styles))
            print("\n提示: 也可以输入简化名称（如 'anime', 'oil-painting' 等），会自动转换")
            style_input = input("\n请输入风格（直接回车使用默认 <auto>）: ").strip()
            if style_input:
                style = normalize_style(style_input)
                print(f"使用风格: {style}")
        
        try:
            thread_id = f"user-{thread_id_counter}"
            thread_id_counter += 1
            
            if choice == "1":
                print("\n" + "-" * 60)
                print("正在生成图片...")
                print("-" * 60)
                
                response = generate_image_from_text(user_input, thread_id=thread_id)
                
                # 提取响应内容
                if safe_has_key(response, "messages"):
                    messages = safe_get(response, "messages", [])
                    # 提取图片信息
                    image_info_list = extract_image_info_from_messages(messages)
                    for msg in messages:
                        if is_assistant_message(msg):
                            content = get_message_content(msg)
                            print(f"\n助手回复: {content}")
                    # 显示图片信息
                    if image_info_list:
                        print("\n" + "=" * 60)
                        print("图片生成成功！")
                        print("=" * 60)
                        for img_info in image_info_list:
                            if img_info.get('saved_paths'):
                                print(f"\n已保存的图片:")
                                for path in img_info['saved_paths']:
                                    print(f"  - {path}")
                            if img_info.get('image_urls'):
                                print(f"\n图片URL:")
                                for url in img_info['image_urls']:
                                    print(f"  - {url}")
                        print("=" * 60)
                elif safe_has_key(response, "error"):
                    error_msg = safe_get(response, 'error', '未知错误')
                    print(f"\n错误: {error_msg}")
                    # 检查是否包含任务ID（超时情况）
                    import re
                    task_id_match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', error_msg)
                    if task_id_match:
                        task_id = task_id_match.group()
                        print(f"\n任务ID: {task_id}")
                        print("提示: 可以使用选项 4 查询任务状态，或稍后重试")
                    suggestion = safe_get(response, 'suggestion')
                    if suggestion:
                        print(f"建议: {suggestion}")
            
            elif choice == "2":
                print("\n" + "-" * 60)
                print(f"正在生成图片（风格: {style or '<auto>'}）...")
                print("-" * 60)
                
                response = generate_image_from_text(user_input, style=style, thread_id=thread_id)
                
                if safe_has_key(response, "messages"):
                    messages = safe_get(response, "messages", [])
                    # 提取图片信息
                    image_info_list = extract_image_info_from_messages(messages)
                    for msg in messages:
                        if is_assistant_message(msg):
                            content = get_message_content(msg)
                            print(f"\n助手回复: {content}")
                    # 显示图片信息
                    if image_info_list:
                        print("\n" + "=" * 60)
                        print("图片生成成功！")
                        print("=" * 60)
                        for img_info in image_info_list:
                            if img_info.get('saved_paths'):
                                print(f"\n已保存的图片:")
                                for path in img_info['saved_paths']:
                                    print(f"  - {path}")
                            if img_info.get('image_urls'):
                                print(f"\n图片URL:")
                                for url in img_info['image_urls']:
                                    print(f"  - {url}")
                        print("=" * 60)
                elif safe_has_key(response, "error"):
                    error_msg = safe_get(response, 'error', '未知错误')
                    print(f"\n错误: {error_msg}")
                    # 检查是否包含任务ID（超时情况）
                    import re
                    task_id_match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', error_msg)
                    if task_id_match:
                        task_id = task_id_match.group()
                        print(f"\n任务ID: {task_id}")
                        print("提示: 可以使用选项 4 查询任务状态，或稍后重试")
                    suggestion = safe_get(response, 'suggestion')
                    if suggestion:
                        print(f"建议: {suggestion}")
            
            elif choice == "3":
                print("\n" + "-" * 60)
                print("流式输出:")
                print("-" * 60)
                
                full_response = ""
                for chunk in stream_generate_image(user_input, style=style, thread_id=thread_id):
                    # 安全地访问 chunk，无论是字典还是对象
                    if safe_has_key(chunk, "error"):
                        error_msg = safe_get(chunk, 'error', '未知错误')
                        print(f"\n错误: {error_msg}")
                        break
                    # 尝试提取流式内容
                    # 根据实际的 chunk 结构来提取
                    if safe_has_key(chunk, "messages"):
                        messages = safe_get(chunk, "messages", [])
                        for msg in messages:
                            if is_assistant_message(msg):
                                content = get_message_content(msg)
                                if content:
                                    print(content, end="", flush=True)
                                    full_response += content
                
                print("\n" + "-" * 60)
                print("流式输出完成")
            
            elif choice == "4":
                print("\n" + "-" * 60)
                print("查询任务状态")
                print("-" * 60)
                
                task_id = input("\n请输入任务ID: ").strip()
                if not task_id:
                    print("\n任务ID不能为空！")
                    continue
                
                # 使用内部实现函数查询（不是工具版本）
                query_result = _query_task_status_impl(task_id)
                try:
                    result_data = json.loads(query_result)
                    if result_data.get('success'):
                        print("\n" + "=" * 60)
                        print("任务查询成功！")
                        print("=" * 60)
                        print(f"任务状态: {result_data.get('task_status', '未知')}")
                        if result_data.get('saved_paths'):
                            print(f"\n已保存的图片:")
                            for path in result_data['saved_paths']:
                                print(f"  - {path}")
                        if result_data.get('image_urls'):
                            print(f"\n图片URL:")
                            for url in result_data['image_urls']:
                                print(f"  - {url}")
                        print("=" * 60)
                    else:
                        print(f"\n查询结果: {result_data.get('message') or result_data.get('error', '未知错误')}")
                        if result_data.get('task_status'):
                            print(f"任务状态: {result_data['task_status']}")
                        if result_data.get('task_id'):
                            print(f"任务ID: {result_data['task_id']}")
                except json.JSONDecodeError:
                    print(f"\n查询结果: {query_result}")
        
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
    
    print("\n" + "=" * 60)
    print("使用说明:")
    print("=" * 60)
    print("1. 确保已设置 DASHSCOPE_API_KEY 环境变量")
    print("2. 调用 generate_image_from_text(user_input) 生成图片")
    print("3. 可选参数: style (风格), thread_id (对话ID)")
    print("4. 支持的风格: anime, oil-painting, water-color, chinese-painting 等")
    print("=" * 60)

