"""
使用 LangChain 1.0 LCEL (LangChain Expression Language) 管道操作符的示例
展示如何使用 | 操作符来组合组件
参考 text2img.py 的逻辑，添加完整功能
"""
import os
import json
import time
import requests
from typing import Optional
from dotenv import load_dotenv
import dashscope
from dashscope import ImageSynthesis

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 加载环境变量
load_dotenv()

# 配置 DashScope API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ==================== 辅助函数 ====================

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
        print(f"下载图片失败 {url}: {e}")
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
                        print(f"[轮询] 任务处理中... (已等待 {elapsed_time} 秒，轮询次数: {poll_count})")
                    time.sleep(3)
                else:
                    time.sleep(3)
            else:
                time.sleep(3)
        except AttributeError:
            return {"status": "no_polling", "task_id": task_id, "message": "API不支持轮询"}
        except Exception as e:
            time.sleep(3)
    
    return {"status": "timeout", "message": f"任务超时（已等待 {max_wait_time} 秒）", "task_id": task_id}

def query_task_status(task_id: str) -> dict:
    """
    查询图片生成任务的状态
    
    Args:
        task_id: 任务ID
    
    Returns:
        任务状态字典
    """
    print("\n" + "=" * 60)
    print("[查询] 查询任务状态")
    print("=" * 60)
    print(f"任务ID: {task_id}")
    print("=" * 60)
    
    try:
        # 尝试使用 fetch 方法（如果存在）
        if hasattr(ImageSynthesis, 'fetch'):
            print("[日志] 使用 fetch 方法查询")
            result = ImageSynthesis.fetch(task_id=task_id)
        else:
            print("[日志] 使用 call 方法查询")
            result = ImageSynthesis.call(
                model=ImageSynthesis.Models.wanx_v1,
                task_id=task_id
            )
        
        print(f"[日志] 查询结果状态码: {result.status_code}")
        
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
                    print(f"[日志] 开始下载 {len(image_urls)} 张图片")
                    imgs_dir = os.path.join(os.path.dirname(__file__), 'imgs')
                    os.makedirs(imgs_dir, exist_ok=True)
                    
                    for idx, url in enumerate(image_urls):
                        print(f"[日志] 下载图片 {idx + 1}/{len(image_urls)}: {url[:50]}...")
                        timestamp = int(time.time())
                        filename = f"image_{timestamp}_{idx + 1}.png"
                        save_path = os.path.join(imgs_dir, filename)
                        
                        if download_image(url, save_path):
                            saved_paths.append(save_path)
                            print(f"[成功] 图片已保存: {save_path}")
                
                if task_status == 'SUCCEEDED' or image_urls:
                    return {
                        "success": True,
                        "task_status": task_status or "SUCCEEDED",
                        "image_urls": image_urls,
                        "saved_paths": saved_paths,
                        "task_id": task_id
                    }
                elif task_status == 'FAILED':
                    return {
                        "success": False,
                        "task_status": "FAILED",
                        "error": output.get('message', '任务失败'),
                        "task_id": task_id
                    }
                else:
                    return {
                        "success": False,
                        "task_status": task_status or "PROCESSING",
                        "message": "任务仍在处理中，请稍后再试",
                        "task_id": task_id
                    }
            else:
                return {
                    "success": False,
                    "error": "无法获取任务状态",
                    "task_id": task_id
                }
        else:
            return {
                "success": False,
                "error": f"查询任务状态失败: {result.message}",
                "task_id": task_id
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"查询任务状态异常: {str(e)}",
            "task_id": task_id
        }

# ==================== LCEL 管道写法示例 ====================

# 1. 创建模型
qwen_model = ChatTongyi(
    model_name="qwen-plus",
    temperature=0.7
)

# 2. 创建提示词模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的AI绘画提示词优化专家。请将用户的中文描述优化为详细的英文提示词，用于AI图片生成。

要求：
1. 保留用户描述的核心内容
2. 添加艺术性的细节描述（如光线、色彩、构图、氛围等）
3. 使用专业的艺术术语
4. 确保提示词清晰、具体、富有表现力
5. 如果用户输入是中文，请翻译成英文
6. 请只返回优化后的英文提示词，不要添加其他解释"""),
    ("user", "{user_input}")
])

# 3. 创建输出解析器
output_parser = StrOutputParser()

# 4. 使用管道操作符 | 组合链
# 这是 LCEL 的核心语法：prompt | model | parser
def log_optimized_prompt(optimized_prompt: str) -> str:
    """记录优化后的提示词"""
    print("\n" + "=" * 60)
    print("[步骤1] 提示词优化完成")
    print("=" * 60)
    print(f"优化后的提示词: {optimized_prompt}")
    print("=" * 60)
    return optimized_prompt

optimize_chain = (
    prompt_template 
    | qwen_model 
    | output_parser
    | RunnableLambda(log_optimized_prompt)
)

# 5. 图片生成函数（作为 Runnable）
def generate_image_from_prompt(prompt: str, style: Optional[str] = None, size: Optional[str] = None) -> dict:
    """
    使用 DashScope Wanx API 生成图片
    
    Args:
        prompt: 图片生成的提示词（英文）
        style: 图片风格（可选）
        size: 图片尺寸（可选）
    
    Returns:
        包含图片URL和保存路径的字典
    """
    try:
        # 设置默认参数并规范化风格
        if style is None:
            style = '<auto>'
        else:
            original_style = style
            style = normalize_style(style)
            if original_style != style:
                print(f"[日志] 风格参数已规范化: '{original_style}' -> '{style}'")
        
        if size is None:
            size = '1024*1024'
        
        # 打印API调用参数
        print("\n" + "=" * 60)
        print("[步骤2] 调用 DashScope API 生成图片")
        print("=" * 60)
        print(f"提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"风格: {style}")
        print(f"尺寸: {size}")
        print(f"数量: 1")
        print("=" * 60)
        
        # 调用 DashScope Wanx API
        result = ImageSynthesis.call(
            model=ImageSynthesis.Models.wanx_v1,
            prompt=prompt,
            n=1,
            size=size,
            style=style
        )
        
        print(f"[日志] API调用完成，状态码: {result.status_code}")
        
        if result.status_code == 200:
            output = result.output
            if not output:
                print("[错误] API返回空结果")
                return {
                    "success": False,
                    "error": "API返回空结果"
                }
            
            task_id = output.get('task_id', '')
            task_status = output.get('task_status', '')
            
            print(f"[日志] 任务ID: {task_id}")
            print(f"[日志] 任务状态: {task_status}")
            
            # 检查任务是否失败
            if task_status == 'FAILED':
                print("[错误] 任务失败")
                error_msg = f"图片生成任务失败"
                if output.get('code'):
                    error_msg += f"\n错误代码: {output.get('code')}"
                if output.get('message'):
                    error_msg += f"\n错误详情: {output.get('message')}"
                return {
                    "success": False,
                    "error": error_msg,
                    "task_id": task_id,
                    "task_status": task_status
                }
            
            # 检查是否直接返回了结果
            image_urls = []
            if 'results' in output and output['results']:
                print(f"[日志] 直接返回了结果，结果数量: {len(output['results'])}")
                # 直接返回了结果
                for idx, item in enumerate(output['results']):
                    if isinstance(item, dict) and 'url' in item:
                        image_urls.append(item['url'])
                        print(f"[日志] 图片 {idx + 1} URL: {item['url']}")
            else:
                print("[日志] 未直接返回结果，需要轮询任务状态")
            
            # 如果没有图片URL但有task_id，尝试轮询任务状态
            if not image_urls and task_id:
                print("\n" + "=" * 60)
                print("[步骤3] 轮询任务状态")
                print("=" * 60)
                print(f"任务ID: {task_id}")
                print("等待生成完成...")
                print("=" * 60)
                poll_result = poll_task_status(task_id)
                print(f"[日志] 轮询结果: {poll_result['status']}")
                if poll_result['status'] == 'success':
                    print("[成功] 任务完成，开始提取图片URL")
                    output = poll_result['output']
                    if 'results' in output and output['results']:
                        print(f"[日志] 提取到 {len(output['results'])} 个结果")
                        for idx, item in enumerate(output['results']):
                            if isinstance(item, dict) and 'url' in item:
                                image_urls.append(item['url'])
                                print(f"[日志] 图片 {idx + 1} URL: {item['url']}")
                elif poll_result['status'] == 'failed':
                    return {
                        "success": False,
                        "error": f"图片生成失败: {poll_result.get('message', '未知错误')}",
                        "task_id": task_id
                    }
                elif poll_result['status'] == 'timeout':
                    return {
                        "success": False,
                        "error": f"图片生成超时（任务ID: {task_id}），请稍后查询任务状态",
                        "task_id": task_id
                    }
            
            # 如果仍然没有图片URL，返回错误
            if not image_urls:
                print("[错误] 未获取到图片URL")
                return {
                    "success": False,
                    "error": f"图片生成完成但没有返回图片URL。任务ID: {task_id}",
                    "task_id": task_id,
                    "output": output
                }
            
            print(f"\n[成功] 共获取到 {len(image_urls)} 张图片")
            
            # 下载图片到本地
            saved_paths = []
            if image_urls:
                print("\n" + "=" * 60)
                print("[步骤4] 下载图片到本地")
                print("=" * 60)
                # 确保 imgs 文件夹存在
                imgs_dir = os.path.join(os.path.dirname(__file__), 'imgs')
                os.makedirs(imgs_dir, exist_ok=True)
                print(f"[日志] 保存目录: {imgs_dir}")
                
                # 下载每张图片
                for idx, url in enumerate(image_urls):
                    print(f"\n[日志] 正在下载图片 {idx + 1}/{len(image_urls)}...")
                    timestamp = int(time.time())
                    filename = f"image_{timestamp}_{idx + 1}.png"
                    save_path = os.path.join(imgs_dir, filename)
                    
                    if download_image(url, save_path):
                        saved_paths.append(save_path)
                        print(f"[成功] 图片已保存: {save_path}")
                    else:
                        print(f"[失败] 图片下载失败: {url}")
                
                print("=" * 60)
            
            print("\n" + "=" * 60)
            print("[完成] 图片生成流程完成")
            print("=" * 60)
            print(f"成功保存 {len(saved_paths)}/{len(image_urls)} 张图片")
            print("=" * 60)
            
            return {
                "success": True,
                "image_urls": image_urls,
                "saved_paths": saved_paths,
                "task_id": task_id,
                "prompt": prompt,
                "style": style,
                "size": size
            }
        else:
            print(f"[错误] API调用失败，状态码: {result.status_code}")
            error_msg = f"图片生成失败: {result.message}"
            if result.code:
                error_msg += f" (错误代码: {result.code})"
            
            # 检查输出中的详细错误信息
            if result.output:
                output = result.output
                if output.get('code'):
                    print(f"[错误] 错误代码: {output.get('code')}")
                    error_msg += f"\n错误代码: {output.get('code')}"
                if output.get('message'):
                    print(f"[错误] 错误详情: {output.get('message')}")
                    error_msg += f"\n错误详情: {output.get('message')}"
            
            return {
                "success": False,
                "error": error_msg
            }
            
    except Exception as e:
        print(f"[异常] 图片生成过程发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"图片生成异常: {str(e)}"
        }

# 6. 创建带参数的图片生成函数包装器
def create_image_generator(style: Optional[str] = None, size: Optional[str] = None):
    """创建带参数的图片生成器"""
    def _generate(prompt: str) -> dict:
        return generate_image_from_prompt(prompt, style=style, size=size)
    return RunnableLambda(_generate)

# 7. 组合完整的链：优化提示词 -> 生成图片
# 使用管道操作符连接多个步骤
text_to_image_chain = (
    optimize_chain  # 优化提示词
    | create_image_generator()  # 生成图片
)

# ==================== 使用示例 ====================

def generate_image_lcel(user_input: str, style: Optional[str] = None, size: Optional[str] = None) -> dict:
    """
    使用 LCEL 管道写法生成图片
    
    Args:
        user_input: 用户的文字描述
        style: 图片风格（可选）
        size: 图片尺寸（可选）
    
    Returns:
        包含图片URL和保存路径的字典
    """
    print("\n" + "=" * 60)
    print("[开始] 图片生成流程")
    print("=" * 60)
    print(f"用户输入: {user_input}")
    if style:
        print(f"指定风格: {style}")
    if size:
        print(f"指定尺寸: {size}")
    print("=" * 60)
    
    # 创建带参数的链
    chain = (
        optimize_chain
        | create_image_generator(style=style, size=size)
    )
    
    # 直接调用链，输入会自动传递
    result = chain.invoke({"user_input": user_input})
    
    print("\n" + "=" * 60)
    print("[结果] 最终返回结果")
    print("=" * 60)
    print(f"成功: {result.get('success', False)}")
    if result.get('success'):
        if result.get('saved_paths'):
            print(f"保存路径数量: {len(result['saved_paths'])}")
        if result.get('image_urls'):
            print(f"图片URL数量: {len(result['image_urls'])}")
    else:
        print(f"错误信息: {result.get('error', '未知错误')}")
    print("=" * 60)
    
    return result


# ==================== 更复杂的链式组合示例 ====================

# 示例：添加格式化步骤
def format_result(result: dict) -> str:
    """格式化结果"""
    if result.get("success"):
        saved_paths = result.get("saved_paths", [])
        image_urls = result.get("image_urls", [])
        if saved_paths:
            return f"图片生成成功！\n已保存的图片:\n" + "\n".join(f"  - {path}" for path in saved_paths)
        elif image_urls:
            return f"图片生成成功！\n图片URL: {image_urls[0]}"
        return "图片生成成功，但未获取到URL"
    else:
        error = result.get('error', '未知错误')
        task_id = result.get('task_id')
        if task_id:
            return f"图片生成失败: {error}\n任务ID: {task_id}\n提示: 可以使用选项 4 查询任务状态"
        return f"图片生成失败: {error}"

formatter = RunnableLambda(format_result)

# 完整的链：优化提示词 -> 生成图片 -> 格式化结果
complete_chain = (
    optimize_chain
    | create_image_generator()
    | formatter
)

# ==================== 流式输出示例 ====================

def stream_generate_lcel(user_input: str, style: Optional[str] = None):
    """
    使用 LCEL 流式生成（流式输出优化过程）
    """
    print("\n" + "=" * 60)
    print("[流式] 开始流式生成")
    print("=" * 60)
    print("[步骤1] 流式优化提示词:")
    print("-" * 60)
    
    # 流式调用优化链
    optimized_prompt = ""
    for chunk in optimize_chain.stream({"user_input": user_input}):
        print(chunk, end="", flush=True)
        optimized_prompt += chunk
    
    print("\n" + "-" * 60)
    print(f"[完成] 提示词优化完成，长度: {len(optimized_prompt)} 字符")
    print("=" * 60)
    
    # 然后生成图片（非流式）
    result = generate_image_from_prompt(optimized_prompt, style=style)
    return result


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("LangChain 1.0 LCEL 管道操作符示例")
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
    
    # 交互式用户输入
    while True:
        print("\n" + "=" * 60)
        print("请选择操作模式:")
        print("1. 基本管道链 (优化提示词 -> 生成图片)")
        print("2. 完整链 (优化 -> 生成 -> 格式化结果)")
        print("3. 流式输出 (实时显示优化过程)")
        print("4. 指定风格生成图片")
        print("5. 查询任务状态（输入任务ID）")
        print("0. 退出")
        print("=" * 60)
        
        choice = input("\n请输入选项 (0-5): ").strip()
        
        if choice == "0":
            print("\n感谢使用！再见！")
            break
        elif choice not in ["1", "2", "3", "4", "5"]:
            print("\n无效选项，请重新选择！")
            continue
        
        # 获取用户输入
        user_input = input("\n请输入图片描述（中文或英文）: ").strip()
        if not user_input:
            print("\n输入不能为空，请重新输入！")
            continue
        
        style = None
        if choice == "4":
            print("\n支持的风格:")
            print("  " + ", ".join(supported_styles))
            print("\n提示: 也可以输入简化名称（如 'anime', 'oil-painting' 等），会自动转换")
            style_input = input("\n请输入风格（直接回车使用默认 <auto>）: ").strip()
            if style_input:
                style = normalize_style(style_input)
                print(f"使用风格: {style}")
        
        try:
            if choice == "1":
                print("\n" + "-" * 60)
                print("正在生成图片...")
                print("-" * 60)
                result = generate_image_lcel(user_input)
                print(f"\n结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
                if result.get("success"):
                    if result.get('saved_paths'):
                        print(f"\n已保存的图片:")
                        for path in result['saved_paths']:
                            print(f"  - {path}")
                    if result.get('image_urls'):
                        print(f"\n图片URL:")
                        for url in result['image_urls']:
                            print(f"  - {url}")
            
            elif choice == "2":
                print("\n" + "-" * 60)
                print("正在生成图片...")
                print("-" * 60)
                formatted_result = complete_chain.invoke({"user_input": user_input})
                print(f"\n格式化结果:\n{formatted_result}")
            
            elif choice == "3":
                print("\n" + "-" * 60)
                print("流式输出优化过程:")
                print("-" * 60)
                result = stream_generate_lcel(user_input)
                print(f"\n\n图片生成结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
                if result.get("success"):
                    if result.get('saved_paths'):
                        print(f"\n已保存的图片:")
                        for path in result['saved_paths']:
                            print(f"  - {path}")
                    if result.get('image_urls'):
                        print(f"\n图片URL:")
                        for url in result['image_urls']:
                            print(f"  - {url}")
            
            elif choice == "4":
                print("\n" + "-" * 60)
                print(f"正在生成图片（风格: {style or '<auto>'}）...")
                print("-" * 60)
                result = generate_image_lcel(user_input, style=style)
                print(f"\n结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
                if result.get("success"):
                    if result.get('saved_paths'):
                        print(f"\n已保存的图片:")
                        for path in result['saved_paths']:
                            print(f"  - {path}")
                    if result.get('image_urls'):
                        print(f"\n图片URL:")
                        for url in result['image_urls']:
                            print(f"  - {url}")
                else:
                    # 检查是否包含任务ID
                    task_id = result.get('task_id')
                    if task_id:
                        print(f"\n任务ID: {task_id}")
                        print("提示: 可以使用选项 5 查询任务状态")
            
            elif choice == "5":
                print("\n" + "-" * 60)
                print("查询任务状态")
                print("-" * 60)
                
                task_id = input("\n请输入任务ID: ").strip()
                if not task_id:
                    print("\n任务ID不能为空！")
                    continue
                
                result = query_task_status(task_id)
                if result.get('success'):
                    print("\n" + "=" * 60)
                    print("任务查询成功！")
                    print("=" * 60)
                    print(f"任务状态: {result.get('task_status', '未知')}")
                    if result.get('saved_paths'):
                        print(f"\n已保存的图片:")
                        for path in result['saved_paths']:
                            print(f"  - {path}")
                    if result.get('image_urls'):
                        print(f"\n图片URL:")
                        for url in result['image_urls']:
                            print(f"  - {url}")
                    print("=" * 60)
                else:
                    print(f"\n查询结果: {result.get('message') or result.get('error', '未知错误')}")
                    if result.get('task_status'):
                        print(f"任务状态: {result.get('task_status')}")
                    if result.get('task_id'):
                        print(f"任务ID: {result.get('task_id')}")
        
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
    
    print("\n" + "=" * 60)
    print("LCEL 管道操作符说明:")
    print("=" * 60)
    print("""
LangChain 1.0 支持 LCEL (LangChain Expression Language)，可以使用管道操作符 | 来组合组件：

1. 基本语法：
   chain = prompt | model | parser

2. 复杂组合：
   chain = step1 | step2 | step3 | ...

3. 函数组合：
   chain = prompt | model | RunnableLambda(custom_function)

4. 流式输出：
   for chunk in chain.stream(input):
       process(chunk)

5. 与 Agent 的区别：
   - LCEL: 声明式，适合简单的线性流程
   - Agent: 智能决策，可以自主选择工具和步骤

当前 text2img.py 使用 create_agent，这是更高级的 API，适合复杂的多步骤决策场景。
如果流程是线性的，可以使用 LCEL 的管道写法，代码会更简洁。
    """)
    print("=" * 60)
