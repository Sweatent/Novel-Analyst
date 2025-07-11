import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import requests
import json
import logging
import time
import re
from typing import List, Dict, Optional, Any
import os
from datetime import datetime
import signal
import sys
import pickle
from pathlib import Path
import traceback
from functools import wraps
import argparse
import subprocess
import platform

# ==============================================================================
# 1. 配置与日志
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('novel_analysis_refactored.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """从JSON文件加载配置"""
    if not Path(file_path).exists():
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"加载配置文件 {file_path} 失败: {e}")
        return {}

def save_config_to_file(file_path: str, config_data: Dict[str, Any]):
    """将配置保存到JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        logger.info(f"配置已保存到 {file_path}")
    except IOError as e:
        logger.error(f"保存配置文件 {file_path} 失败: {e}")

class Config:
    """统一管理所有配置"""
    def __init__(self, settings: Dict[str, Any]):
        # 从字典加载配置，提供默认值
        self.api_key: str = settings.get("api_key")
        self.base_url: str = settings.get("base_url", "https://api.openai.com/v1")
        self.model: str = settings.get("model", "gpt-3.5-turbo")
        self.max_tokens: int = settings.get("max_tokens", 3000)
        self.api_timeout: float = settings.get("timeout", 180.0)
        self.rate_limit_delay: float = settings.get("delay", 2.0)
        self.max_retries: int = 5
        self.retry_delay: float = 3.0
        self.backoff: float = 1.5
        
        self.epub_path: str = settings.get("epub_path")
        self.output_path: str = settings.get("output", "analysis_results_refactored.json")
        self.report_path: str = settings.get("report", "analysis_report_refactored.md")
        self.progress_file: str = "analysis_progress_refactored.pkl"
        
        self.chunk_size: int = 150 # 每150章切割一次
        self.long_chapter_threshold: int = 6000 # 长章节分段阈值
        
        self.prompt: str = settings.get("prompt", self._get_default_prompt())
        self.use_json_schema: bool = settings.get("use_json_schema", False)

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'Config':
        """从argparse.Namespace对象创建Config实例"""
        return Config(vars(args))

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为可序列化的字典"""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "timeout": self.api_timeout,
            "delay": self.rate_limit_delay,
            "epub_path": self.epub_path,
            "output": self.output_path,
            "report": self.report_path,
            "prompt": self.prompt,
            "use_json_schema": self.use_json_schema,
        }

    def _get_default_prompt(self) -> str:
        return """
你现在需要分析这篇内容并完成下面的任务，你需要选取人生感悟或者是恋爱方法经验等，保留上下文。如果没有可以再金句那里设置一个空列表
请注意：
1.请勿包含故事情节：样例：
1>虽然他不知道别的小说作者是怎么样的，但他平时就喜欢去更多的地方，见更多的风景，这样才能写出来更加精彩的故事。
2>不知不觉中，苏白粥的心情变得十分安逸，之前的不愉快已经烟消云散。
2.你的目的是保存金句让人感觉有价值的，不是保留人物的无用对话等
请严格遵守以下JSON格式要求（请勿在JSON内部添加任何注释）：

<JSON>
{
    "chapter_title": "在此处填写章节标题",
    "golden_sentences": [
        {
            "sentence": "在此处填写黄金句子的纯文本内容",
            "speaker": "在此处填写句子来源（角色名/旁白等，你也可以根据上下文判断）",
            "reason": "在此处填写选择该句的恋爱理由"
        }
    ],
    "chapter_summary": "在此处填写本章的恋爱主线总结"
}
</JSON>
章节内容如下：
"""

# ==============================================================================
# 2. 工具函数与装饰器
# ==============================================================================

class RateLimitError(Exception): pass
class APITimeoutError(Exception): pass
class SensitiveWordsError(Exception): pass

def confirm_action(prompt: str) -> bool:
    """向用户请求确认危险操作"""
    if not sys.stdin.isatty():
        logger.warning(f"非交互环境，自动跳过操作: {prompt}")
        return False
    
    try:
        response = input(f"⚠️  {prompt} (yes/no): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        logger.info("操作已被用户取消。")
        return False
    except (EOFError, KeyboardInterrupt):
        logger.info("\n操作已被用户中断。")
        return False

def retry_on_failure(max_retries: int, delay: float, backoff: float):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            self_obj = args[0] if args else None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    logger.info("检测到中断信号，终止重试流程")
                    raise
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败: {e}")
                        raise
                    
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                    logger.info(f"等待 {current_delay:.1f} 秒后重试...")
                    
                    interrupt_handler = getattr(self_obj, 'interrupt_handler', None)
                    if interrupt_handler:
                        interrupt_handler.safe_sleep(current_delay)
                    else:
                        time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator

# ==============================================================================
# 3. 核心模块类
# ==============================================================================

class InterruptHandler:
    """处理Ctrl+C等中断信号"""
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        logger.info("收到中断信号，正在安全退出...")
        self.interrupted = True
    
    def should_continue(self) -> bool:
        return not self.interrupted

    def safe_sleep(self, seconds: float):
        """可中断的sleep"""
        for _ in range(int(seconds)):
            if self.interrupted: raise KeyboardInterrupt
            time.sleep(1)
        if seconds % 1 > 0:
            if self.interrupted: raise KeyboardInterrupt
            time.sleep(seconds % 1)

class FileManager:
    """负责所有文件读写、加载、保存和清理操作"""
    def __init__(self, config: Config):
        self.config = config

    def clean_all(self):
        """清理所有生成的文件，并请求用户确认"""
        logger.info("开始清理分析文件...")
        
        # 清理进度文件
        progress_file = Path(self.config.progress_file)
        if progress_file.exists():
            if confirm_action(f"是否删除进度文件 '{progress_file}'?"):
                progress_file.unlink()
                logger.info(f"进度文件已删除: {progress_file}")

        # 清理结果文件（包括分块）
        output_path = Path(self.config.output_path)
        if output_path.exists():
             if confirm_action(f"是否删除主结果文件 '{output_path}'?"):
                output_path.unlink()
                logger.info(f"主结果文件已删除: {output_path}")
        
        chunk_files = list(output_path.parent.glob(f"{output_path.stem}_part_*{output_path.suffix}"))
        if chunk_files:
            if confirm_action(f"是否删除 {len(chunk_files)} 个结果分块文件?"):
                for f in chunk_files:
                    f.unlink()
                logger.info(f"{len(chunk_files)} 个结果分块文件已删除。")

        # 清理报告文件
        report_path = Path(self.config.report_path)
        if report_path.exists():
            if confirm_action(f"是否删除报告文件 '{report_path}'?"):
                report_path.unlink()
                logger.info(f"报告文件已删除: {report_path}")
        
        logger.info("文件清理完成。")

    def load_progress(self) -> Dict[str, Any]:
        """加载进度，并能自动迁移旧版本进度文件"""
        new_progress_file = Path(self.config.progress_file)
        old_progress_file = Path("analysis_progress.pkl")

        # 优先使用新文件
        if new_progress_file.exists():
            try:
                with open(new_progress_file, 'rb') as f:
                    progress = pickle.load(f)
                logger.info(f"已加载进度: {len(progress.get('completed_chapters', []))}/{progress.get('total_chapters', 0)} 章节完成")
                return progress
            except Exception as e:
                logger.error(f"加载进度文件 '{new_progress_file}' 失败: {e}")
        
        # 如果新文件不存在，则尝试加载并迁移旧文件
        elif old_progress_file.exists():
            logger.info(f"未找到新进度文件，但检测到旧进度文件 '{old_progress_file}'。将加载并迁移。")
            try:
                with open(old_progress_file, 'rb') as f:
                    progress = pickle.load(f)
                logger.info(f"成功加载旧进度。任何新进度将被保存到 '{new_progress_file}'。")
                return progress
            except Exception as e:
                logger.error(f"加载旧进度文件 '{old_progress_file}' 失败: {e}")

        return self._get_default_progress()

    def save_progress(self, progress: Dict[str, Any]):
        """保存进度"""
        try:
            progress['last_save_time'] = datetime.now().isoformat()
            with open(self.config.progress_file, 'wb') as f:
                pickle.dump(progress, f)
            logger.debug("进度已保存")
        except Exception as e:
            logger.error(f"保存进度失败: {e}")

    def _get_default_progress(self) -> Dict[str, Any]:
        return {
            'completed_chapters': [], 'failed_chapters': [], 'current_chapter_index': 0,
            'total_chapters': 0, 'start_time': None, 'last_save_time': None
        }

    def load_existing_results(self) -> Optional[Dict]:
        """加载结果文件，并能自动迁移旧版本结果"""
        # 优先尝试加载新版本文件
        results = self._load_results_from_path(self.config.output_path)
        if results:
            return results

        # 如果新文件不存在，尝试加载旧版本文件
        old_output_path = "analysis_results.json"
        logger.info(f"未找到新结果文件，尝试查找旧结果文件 '{old_output_path}'...")
        results = self._load_results_from_path(old_output_path)
        if results:
            logger.info(f"成功加载旧结果。任何新结果将被保存到 '{self.config.output_path}' 相关文件中。")
            return results
            
        return None

    def _load_results_from_path(self, path_str: str) -> Optional[Dict]:
        """从指定路径加载单个或分块的结果文件"""
        output_path = Path(path_str)
        base_path, file_stem, file_suffix = output_path, output_path.stem, output_path.suffix
        
        # 筛选出符合命名规则的文件并排序
        valid_chunk_files = []
        for p in base_path.parent.glob(f"{file_stem}_part_*{file_suffix}"):
            match = re.search(r'_part_(\d+)', p.name)
            if match:
                valid_chunk_files.append((p, int(match.group(1))))
            else:
                logger.warning(f"发现不符合命名规则的文件，已跳过: {p.name}")
        
        chunk_files = [p for p, num in sorted(valid_chunk_files, key=lambda item: item[1])]

        if not chunk_files:
            if base_path.exists():
                logger.info(f"找到单文件结果: {base_path}")
                try:
                    with open(base_path, 'r', encoding='utf-8') as f: return json.load(f)
                except Exception as e: logger.error(f"加载文件 '{base_path}' 失败: {e}")
            return None

        logger.info(f"找到 {len(chunk_files)} 个 '{file_stem}' 系列的分块文件，正在合并...")
        merged_results, all_chapters = None, []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                if merged_results is None:
                    merged_results = chunk_data
                all_chapters.extend(chunk_data.get('chapters', []))
            except json.JSONDecodeError:
                logger.warning(f"文件 {chunk_file} 格式损坏，已跳过。")
            except Exception as e:
                logger.error(f"加载分块文件 {chunk_file} 时发生未知错误: {e}")
        
        if merged_results:
            merged_results['chapters'] = all_chapters
            logger.info(f"成功合并 {len(all_chapters)} 个章节的结果")
            return merged_results
        return None

    def save_results(self, results: Dict):
        """将分析结果按指定大小分块保存"""
        try:
            output_path = Path(self.config.output_path)
            base_path, file_stem, file_suffix = output_path, output_path.stem, output_path.suffix
            output_dir = base_path.parent
            chunk_size = self.config.chunk_size

            if not output_dir.exists(): output_dir.mkdir(parents=True)

            all_chapters = results.get('chapters', [])
            
            # 核心修复：在保存前，始终按章节索引排序，确保输出文件顺序正确
            all_chapters.sort(key=lambda c: c.get('chapter_index', 0))
            
            if not all_chapters:
                logger.info("没有章节结果需要保存")
                return

            num_chunks = (len(all_chapters) + chunk_size - 1) // chunk_size
            for i in range(num_chunks):
                chunk_path = output_dir / f"{file_stem}_part_{i+1}{file_suffix}"
                chunk_chapters = all_chapters[i*chunk_size : (i+1)*chunk_size]
                
                chunk_results = results.copy()
                chunk_results['chapters'] = chunk_chapters
                chunk_results['novel_info'] = results['novel_info'].copy()
                chunk_results['novel_info']['chunk_number'] = i + 1
                chunk_results['novel_info']['total_chunks'] = num_chunks

                with open(chunk_path, 'w', encoding='utf-8') as f:
                    json.dump(chunk_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分析结果已分块保存到 {num_chunks} 个文件中。")

            if base_path.exists():
                if confirm_action(f"是否删除旧的、未分块的结果文件 '{base_path}'?"):
                    base_path.unlink()
                    logger.info(f"已删除旧的单体结果文件: {base_path}")

        except Exception as e:
            logger.error(f"保存分块结果失败: {e}\n{traceback.format_exc()}")
            raise

    def generate_report(self, results: Dict):
        """生成Markdown格式的分析报告"""
        report_path = self.config.report_path
        logger.info(f"正在生成分析报告: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            info = results['novel_info']
            f.write(f"# 小说金句分析报告: {Path(info['file_path']).name}\n\n")
            f.write("## 摘要\n")
            f.write(f"- **总章节数**: {info['total_chapters']}\n")
            f.write(f"- **已完成**: {info.get('completed_chapters', 0)}\n")
            f.write(f"- **总字数**: {info['total_words']:,}\n")
            f.write(f"- **总金句数**: {info['total_golden_sentences']}\n")
            f.write(f"- **分析用时**: {info.get('analysis_duration', 'N/A')}\n\n")
            
            f.write("---\n\n")
            
            for chapter in sorted(results['chapters'], key=lambda c: c['chapter_index']):
                f.write(f"### 章节 {chapter['chapter_index']}: {chapter['chapter_title']}\n")
                f.write(f"字数: {chapter['word_count']} | 金句数: {len(chapter.get('golden_sentences', []))}\n\n")
                if chapter.get('chapter_summary'):
                    f.write(f"**总结**: {chapter['chapter_summary']}\n\n")
                if chapter.get('golden_sentences'):
                    f.write("**金句列表**:\n")
                    for i, s in enumerate(chapter['golden_sentences'], 1):
                        f.write(f"{i}. **{s.get('speaker', '未知')}**: {s.get('sentence', '')}\n")
                        f.write(f"   - *理由*: {s.get('reason', 'AI分析')}\n")
                    f.write("\n")
                f.write("---\n\n")
        
        logger.info(f"报告已生成: {report_path}")

class APIHandler:
    """处理对AI模型的API调用"""
    def __init__(self, config: Config, interrupt_handler: InterruptHandler):
        self.config = config
        self.interrupt_handler = interrupt_handler
        self.headers = {
            'Authorization': f'Bearer {config.api_key}',
            'Content-Type': 'application/json'
        }

    def analyze_content(self, content: str) -> Optional[Dict]:
        """调用API分析内容，根据配置选择是否使用JSON Schema"""
        
        system_prompt = (
            "你是一个专门用于文本分析的API端点。你的唯一任务是根据用户提供的内容，"
            "返回一个严格符合JSON格式的字符串。不要添加任何解释、注释或Markdown标记。"
            "你的输出必须能够直接被json.loads()解析。"
        )

        api_params = {
            "temperature": 0.1,
            "max_tokens": self.config.max_tokens,
        }

        if self.config.use_json_schema:
            logger.info("正在调用API并启用JSON Schema模式...")
            # 在Schema模式下，使用一个更简洁的、不包含格式描述的提示词
            schema_prompt = """
你现在需要分析这篇内容并完成下面的任务，你需要选取人生感悟或者是恋爱方法经验等，保留上下文。如果没有可以再金句那里设置一个空列表
请注意：
1.请勿包含故事情节：样例：
1>虽然他不知道别的小说作者是怎么样的，但他平时就喜欢去更多的地方，见更多的风景，这样才能写出来更加精彩的故事。
2>不知不觉中，苏白粥的心情变得十分安逸，之前的不愉快已经烟消云散。
2.你的目的是保存金句让人感觉有价值的，不是保留人物的无用对话等
章节内容如下：
"""
            full_prompt = schema_prompt + content
            api_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "novel_analysis_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "chapter_title": {"type": "string"},
                            "golden_sentences": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "sentence": {"type": "string"},
                                        "speaker": {"type": "string"},
                                        "reason": {"type": "string"}
                                    },
                                    "required": ["sentence", "speaker", "reason"]
                                }
                            },
                            "chapter_summary": {"type": "string"}
                        },
                        "required": ["chapter_title", "golden_sentences", "chapter_summary"]
                    },
                    "strict": True,
                }
            }
        else:
            logger.info("正在调用API并启用标准JSON模式...")
            full_prompt = self.config.prompt + content
            api_params["response_format"] = {"type": "json_object"}

        try:
            response = self._call_api(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                **api_params
            )
            
            if not response.get('choices') or not response['choices'][0].get('message', {}).get('content'):
                raise ValueError("API返回空响应")

            result_text = response['choices'][0]['message']['content']
            
            if response['choices'][0].get('finish_reason') == "length":
                logger.warning("API响应因达到max_tokens而被截断。建议增加 --max-tokens 的值。")

            parsed_json = self._parse_api_response(result_text)
            if not isinstance(parsed_json, dict):
                 raise TypeError(f"解析后的JSON不是一个字典，而是 {type(parsed_json)}")
            return parsed_json

        except SensitiveWordsError as e:
            logger.error(f"章节内容包含敏感词，已被API拒绝: {e}")
            raise
        except Exception as e:
            logger.error(f"API调用或解析失败: {e}\n{traceback.format_exc()}")
            raise

    @retry_on_failure(max_retries=5, delay=3.0, backoff=1.5)
    def _call_api(self, messages: List[Dict], **kwargs) -> Dict:
        """使用requests库进行API调用"""
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        data = {"model": self.config.model, "messages": messages, **kwargs}

        response = requests.post(url, headers=self.headers, json=data, timeout=self.config.api_timeout)

        if response.status_code == 200: return response.json()
        if response.status_code == 429: raise RateLimitError(f"API rate limit: {response.text}")
        if response.status_code in [408, 504]: raise APITimeoutError(f"API timeout: {response.text}")
        
        # 专门处理敏感词错误
        if response.status_code == 400 and 'sensitive_words_detected' in response.text:
            raise SensitiveWordsError(f"API检测到敏感词: {response.text}")
            
        raise Exception(f"API error {response.status_code}: {response.text}")

    def _parse_api_response(self, data: str) -> Optional[Dict]:
        """解析API响应，由于使用JSON模式，此处变得更简单"""
        if isinstance(data, dict):
            return data # 如果已经是字典，直接返回
        try:
            # JSON模式下，响应本身就应该是合法的JSON字符串
            return json.loads(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"即使在JSON模式下，解析仍然失败: {e}")
            logger.error(f"收到的无效JSON数据: {data}")
            return None

    def test_connection(self) -> bool:
        """测试API连接和认证"""
        logger.info("正在测试API连接...")
        try:
            self._call_api(
                messages=[{"role": "user", "content": "Say 'test'"}],
                max_tokens=5,
                temperature=0
            )
            logger.info("✅ API连接成功！")
            return True
        except Exception as e:
            logger.error(f"❌ API连接失败: {e}")
            return False

class EPUBParser:
    """负责解析EPUB文件和提取章节"""
    def extract_chapters(self, epub_path: str) -> List[Dict]:
        logger.info(f"开始提取EPUB文件: {epub_path}")
        try:
            book = epub.read_epub(epub_path)
            chapters = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = re.sub(r'\s+', ' ', soup.get_text()).strip()
                if len(text) > 100:
                    title = self._extract_title(item.get_name(), text)
                    chapters.append({'title': title, 'content': text, 'word_count': len(text)})
            logger.info(f"成功提取 {len(chapters)} 个章节")
            return chapters
        except Exception as e:
            logger.error(f"提取EPUB文件失败: {e}\n{traceback.format_exc()}")
            raise

    def _extract_title(self, filename: str, content: str) -> str:
        # 优先从内容前几行提取
        for line in content.split('\n')[:5]:
            line = line.strip()
            if line and (('第' in line and '章' in line) or len(line) < 50):
                return line
        return Path(filename).stem

# ==============================================================================
# 4. 主编排器
# ==============================================================================

class AnalysisOrchestrator:
    """主编排器，协调所有模块完成分析任务"""
    def __init__(self, config: Config):
        self.config = config
        self.interrupt_handler = InterruptHandler()
        self.file_manager = FileManager(config)
        self.api_handler = APIHandler(config, self.interrupt_handler)
        self.epub_parser = EPUBParser()
        
        self.progress = self.file_manager.load_progress()

    def run_analysis(self, resume: bool = True, test_mode: bool = False, re_analyze_chapters: Optional[List[int]] = None, retry_failed: bool = False, find_missing: bool = False):
        """执行完整的小说分析流程"""
        if find_missing:
            self.run_find_missing_chapters()
            return

        if re_analyze_chapters:
            self._prepare_for_reanalysis(re_analyze_chapters)
        
        if retry_failed:
            self.run_retry_failed_chapters()
            return

        start_time = datetime.now()
        logger.info(f"开始分析小说: {self.config.epub_path}")

        try:
            chapters = self.epub_parser.extract_chapters(self.config.epub_path)
            if not chapters:
                logger.error("未能提取任何章节，程序终止。")
                return

            if test_mode:
                logger.info("测试模式启动，仅分析前3章。")
                chapters = chapters[:3]

            results = self._initialize_results(chapters, start_time, resume)
            
            # 核心修复：如果指定了重新分析，必须从头扫描，而不是从上次的索引继续
            if re_analyze_chapters:
                start_index = 0
                logger.info("重新分析模式已激活，将从头扫描所有章节以找到待处理项。")
            else:
                start_index = self.progress.get('current_chapter_index', 0)
                logger.info(f"将从第 {start_index} 章开始分析。")

            for i in range(start_index, len(chapters)):
                chapter = chapters[i]
                if self.interrupt_handler.interrupted:
                    logger.info("检测到中断，正在保存进度...")
                    break
                
                if self.is_chapter_completed(i):
                    continue

                progress_percent = (i + 1) / len(chapters) * 100
                logger.info(f"分析中... {progress_percent:.1f}% ({i+1}/{len(chapters)})")
                try:
                    analysis_result = self.analyze_single_chapter(chapter, i)
                    if analysis_result:
                        self._update_chapter_in_results(results, analysis_result)
                        self.mark_chapter_completed(i)
                        logger.info(self._get_concise_result_log(analysis_result))
                    else:
                        self.mark_chapter_failed(i, "分析返回空结果")
                except Exception as e:
                    logger.error(f"处理章节 {i} 时发生严重错误: {e}") # 移除traceback以简化日志
                    self.mark_chapter_failed(i, str(e))
                
                if (i + 1) % 5 == 0:
                    self.file_manager.save_results(results)
                
                self.interrupt_handler.safe_sleep(self.config.rate_limit_delay)

            self._finalize_analysis(results, start_time)
            self.file_manager.save_results(results)
            self.file_manager.generate_report(results)

        except KeyboardInterrupt:
            logger.warning("\n程序被用户中断。进度已保存。")
        except Exception as e:
            logger.error(f"分析过程中发生致命错误: {e}\n{traceback.format_exc()}")
        finally:
            if 'results' in locals() and results.get('chapters'):
                self.file_manager.save_results(results)
                logger.info("最终结果已保存。")

    def analyze_single_chapter(self, chapter: Dict, index: int) -> Optional[Dict]:
        """分析单个章节，并处理特定API错误"""
        content = chapter['content']
        if len(content) > self.config.long_chapter_threshold:
            logger.info(f"章节 {index} 内容过长，将进行截断处理以避免API错误。")
            content = content[:self.config.long_chapter_threshold]

        try:
            api_result = self.api_handler.analyze_content(content)
            if not api_result:
                return None

            # 补充元数据
            api_result['original_title'] = chapter['title']
            api_result['word_count'] = chapter['word_count']
            api_result['chapter_index'] = index
            api_result['analysis_timestamp'] = datetime.now().isoformat()
            return api_result
        except SensitiveWordsError as e:
            # 捕获敏感词错误，标记失败并继续
            logger.error(f"章节 {index} ({chapter['title']}) 因包含敏感词而无法分析。")
            self.mark_chapter_failed(index, f"内容包含敏感词，被API拒绝: {e}")
            return None

    def _initialize_results(self, chapters: List[Dict], start_time: datetime, resume: bool) -> Dict:
        """初始化或加载分析结果"""
        if resume:
            results = self.file_manager.load_existing_results()
            if results:
                logger.info(f"从上次中断处恢复分析，已完成 {len(results['chapters'])} 章")
                results['novel_info']['resume_count'] = results['novel_info'].get('resume_count', 0) + 1
                results['novel_info']['last_resume_time'] = start_time.isoformat()
                return results
        
        logger.info("未找到或不使用恢复数据，开始新的分析。")
        self.progress = self.file_manager._get_default_progress() # 重置进度
        return {
            'novel_info': {
                'file_path': self.config.epub_path,
                'total_chapters': len(chapters),
                'total_words': sum(c['word_count'] for c in chapters),
                'analysis_start_time': start_time.isoformat(),
                'resume_count': 0
            },
            'chapters': []
        }

    def _finalize_analysis(self, results: Dict, start_time: datetime):
        """完成分析并更新最终统计信息"""
        end_time = datetime.now()
        completed_chapters = len(results['chapters'])
        total_sentences = sum(len(ch.get('golden_sentences', [])) for ch in results['chapters'])
        
        results['novel_info']['analysis_end_time'] = end_time.isoformat()
        results['novel_info']['analysis_duration'] = str(end_time - start_time)
        results['novel_info']['completed_chapters'] = completed_chapters
        results['novel_info']['total_golden_sentences'] = total_sentences
        
        logger.info(f"分析完成! 总用时: {end_time - start_time}")
        logger.info(f"完成章节: {completed_chapters}/{results['novel_info']['total_chapters']}")
        logger.info(f"共找到 {total_sentences} 个金句")

    def _prepare_for_reanalysis(self, chapters_to_reanalyze: List[int]):
        """为重新分析准备，从进度中移除指定章节"""
        logger.info(f"准备重新分析章节: {chapters_to_reanalyze}")
        
        original_completed = set(self.progress['completed_chapters'])
        chapters_to_remove = set(chapters_to_reanalyze)
        
        self.progress['completed_chapters'] = list(original_completed - chapters_to_remove)
        
        # 可选：也从失败列表中移除，以便重试
        self.progress['failed_chapters'] = [
            f for f in self.progress.get('failed_chapters', []) if f['index'] not in chapters_to_remove
        ]
        
        self.file_manager.save_progress(self.progress)
        logger.info("进度已更新，现在将重新分析指定章节。")

    def _get_concise_result_log(self, result: Dict) -> str:
        """为日志生成一个简明扼要的结果摘要"""
        num_sentences = len(result.get('golden_sentences', []))
        summary = result.get('chapter_summary', '无摘要')
        truncated_summary = (summary[:40] + '...') if len(summary) > 40 else summary
        return f"-> 找到 {num_sentences} 个金句。摘要: '{truncated_summary}'"

    def run_retry_failed_chapters(self):
        """仅重试所有失败的章节"""
        logger.info("开始重试所有失败的章节...")
        results = self.file_manager.load_existing_results()
        if not results:
            logger.error("未找到结果文件，无法进行重试。请先正常运行一次。")
            return

        chapters = self.epub_parser.extract_chapters(self.config.epub_path)
        
        all_failed_chapters = self.progress.get('failed_chapters', [])
        completed_chapters_set = set(self.progress.get('completed_chapters', []))

        # 核心修复：只重试那些在失败列表里，但不在成功列表里的章节
        chapters_to_retry = [
            f for f in all_failed_chapters if f['index'] not in completed_chapters_set
        ]

        if not chapters_to_retry:
            logger.info("没有真正需要重试的失败章节。（失败列表可能包含已在后续运行中成功的章节）")
            # 可选：清理过时的失败列表
            if len(all_failed_chapters) > 0:
                self.progress['failed_chapters'] = []
                self.file_manager.save_progress(self.progress)
                logger.info("已清理过时的失败章节列表。")
            return

        logger.info(f"将尝试重试 {len(chapters_to_retry)} 个真正失败的章节。")
        
        for failed_info_index, failed_info in enumerate(chapters_to_retry):
            if self.interrupt_handler.interrupted:
                logger.info("检测到中断，停止重试流程。")
                break
            
            index = failed_info['index']
            chapter = chapters[index]
            logger.info(f"重试章节 {index}: {chapter['title']}")
            
            try:
                analysis_result = self.analyze_single_chapter(chapter, index)
                if analysis_result:
                    self._update_chapter_in_results(results, analysis_result)
                    self.mark_chapter_completed(index)
                    # 从失败列表中移除
                    self.progress['failed_chapters'] = [f for f in self.progress['failed_chapters'] if f['index'] != index]
                    logger.info(f"章节 {index} 重试成功。")
                else:
                    logger.error(f"章节 {index} 重试后仍然失败。")
            except Exception as e:
                logger.error(f"重试章节 {index} 时发生严重错误: {e}")

            # 在重试模式下也定期保存
            if (failed_info_index + 1) % 5 == 0:
                self.file_manager.save_results(results)
                logger.info(f"已保存重试进度 (处理到第 {failed_info_index + 1}/{len(chapters_to_retry)} 个失败章节)")

            self.interrupt_handler.safe_sleep(self.config.rate_limit_delay)

        self.file_manager.save_results(results)
        self.file_manager.save_progress(self.progress)
        logger.info("失败章节重试完成。")

    def run_find_missing_chapters(self):
        """
        校验模式：扫描结果文件，找出所有缺失的章节并进行分析。
        这是最可靠的续跑模式。
        """
        logger.info("--- Running in Find Missing Chapters Mode ---")
        
        # 1. 获取所有章节的元信息
        chapters = self.epub_parser.extract_chapters(self.config.epub_path)
        if not chapters:
            logger.error("未能从EPUB中提取任何章节，无法继续。")
            return

        # 2. 加载所有已存在的结果
        results = self.file_manager.load_existing_results()
        if not results:
            logger.info("未找到任何现有结果，将分析所有章节。")
            results = self._initialize_results(chapters, datetime.now(), resume=False)
        else:
            logger.info(f"已加载 {len(results.get('chapters', []))} 个现有结果。")

        # 3. 识别缺失的章节
        completed_indices = {ch['chapter_index'] for ch in results.get('chapters', [])}
        all_indices = set(range(len(chapters)))
        missing_indices = sorted(list(all_indices - completed_indices))

        if not missing_indices:
            logger.info("🎉 所有章节均已完成分析并存在于结果文件中。无需操作。")
            return

        logger.warning(f"校验发现 {len(missing_indices)} 个缺失的章节。即将开始分析...")
        self.interrupt_handler.safe_sleep(2)

        # 4. 分析缺失的章节
        for i, index in enumerate(missing_indices):
            if self.interrupt_handler.interrupted:
                logger.info("检测到中断，停止分析。")
                break

            chapter_info = chapters[index]
            logger.info(f"正在分析缺失章节 {index} (总进度 {i+1}/{len(missing_indices)}): '{chapter_info['title']}'")

            try:
                analysis_result = self.analyze_single_chapter(chapter_info, index)
                if analysis_result:
                    self._update_chapter_in_results(results, analysis_result)
                    self.mark_chapter_completed(index)
                    logger.info(self._get_concise_result_log(analysis_result))
                else:
                    self.mark_chapter_failed(index, "在校验模式下分析返回空结果")
            except Exception as e:
                logger.error(f"处理缺失章节 {index} 时发生严重错误: {e}")
                self.mark_chapter_failed(index, f"校验模式下发生异常: {str(e)}")

            if (i + 1) % 5 == 0:
                self.file_manager.save_results(results)
                logger.info(f"已保存校验进度 (已处理 {i + 1}/{len(missing_indices)} 个缺失章节)。")

            self.interrupt_handler.safe_sleep(self.config.rate_limit_delay)

        # 5. 最终保存
        logger.info("缺失章节处理完毕。正在执行最终保存...")
        start_time_str = results.get('novel_info', {}).get('analysis_start_time')
        start_time = datetime.fromisoformat(start_time_str) if start_time_str else datetime.now()
        self._finalize_analysis(results, start_time)
        self.file_manager.save_results(results)
        self.file_manager.generate_report(results)
        logger.info("所有缺失章节均已处理完毕，结果和报告已更新。")

    def _update_chapter_in_results(self, results: Dict, new_chapter: Dict):
        """在结果列表中更新或添加一个章节，确保唯一性。"""
        chapter_index = new_chapter['chapter_index']
        
        # 查找现有章节并替换
        for i, chapter in enumerate(results['chapters']):
            if chapter.get('chapter_index') == chapter_index:
                logger.debug(f"找到并替换已存在的章节 {chapter_index} 的结果。")
                results['chapters'][i] = new_chapter
                return
        
        # 如果未找到，则追加
        logger.debug(f"未找到章节 {chapter_index} 的现有结果，将追加新结果。")
        results['chapters'].append(new_chapter)

    def is_chapter_completed(self, index: int) -> bool:
        """检查章节是否已完成（同时检查进度和现有结果）"""
        if index in self.progress['completed_chapters']:
            return True
        # 可以在这里加入对 results 的检查，但进度文件应为准
        return False

    def mark_chapter_completed(self, index: int):
        """标记章节完成，并确保将其从失败列表中移除。"""
        if index not in self.progress['completed_chapters']:
            self.progress['completed_chapters'].append(index)
        
        # 无论如何，都尝试从失败列表中移除
        original_failed_count = len(self.progress.get('failed_chapters', []))
        self.progress['failed_chapters'] = [
            f for f in self.progress.get('failed_chapters', []) if f['index'] != index
        ]
        if len(self.progress.get('failed_chapters', [])) < original_failed_count:
            logger.info(f"章节 {index} 已从失败列表中移除。")

        self.progress['current_chapter_index'] = max(self.progress.get('current_chapter_index', 0), index + 1)
        self.file_manager.save_progress(self.progress)

    def mark_chapter_failed(self, index: int, error: str):
        self.progress['failed_chapters'].append({'index': index, 'error': error})
        self.file_manager.save_progress(self.progress)

    def run_deduplicate_results(self):
        """加载、去重并重新保存所有结果，确保每个章节索引唯一。"""
        logger.info("--- Running in Deduplicate Results Mode ---")
        logger.info("Loading all existing result files...")
        
        results = self.file_manager.load_existing_results()
        
        if not results or not results.get('chapters'):
            logger.warning("No results found to deduplicate. Exiting.")
            return
            
        all_chapters = results.get('chapters', [])
        logger.info(f"Found {len(all_chapters)} total chapter entries. Starting deduplication...")

        unique_chapters = {}
        for chapter in all_chapters:
            index = chapter.get('chapter_index')
            if index is not None:
                # 后出现的会覆盖先出现的，保留最新的
                unique_chapters[index] = chapter
            else:
                logger.warning(f"发现缺少 'chapter_index' 的章节数据，已跳过: {str(chapter)[:100]}")

        deduplicated_list = list(unique_chapters.values())
        
        if len(deduplicated_list) == len(all_chapters):
            logger.info("✅ No duplicates found. Your data is clean.")
            return

        logger.warning(f"去重完成: 从 {len(all_chapters)} 条记录中保留了 {len(deduplicated_list)} 条唯一记录。")
        
        results['chapters'] = deduplicated_list
        
        try:
            self.file_manager.save_results(results)
            logger.info("✅ Deduplicated results have been successfully re-saved.")
        except Exception as e:
            logger.error(f"An error occurred while re-saving deduplicated results: {e}")

    def run_sort_results(self):
        """加载、排序并重新保存所有现有的结果文件，以修复顺序。"""
        logger.info("--- Running in Sort Results Mode ---")
        logger.info("Loading all existing result files...")
        
        results = self.file_manager.load_existing_results()
        
        if not results or not results.get('chapters'):
            logger.warning("No results found to sort. Exiting.")
            return
        
        logger.info(f"Found {len(results['chapters'])} chapters. Sorting and re-saving...")
        
        # 核心逻辑：直接调用save_results，它内部已经包含了排序功能
        try:
            self.file_manager.save_results(results)
            logger.info("✅ All result files have been successfully sorted and re-saved.")
        except Exception as e:
            logger.error(f"An error occurred while sorting and re-saving results: {e}")
            logger.error(traceback.format_exc())

# ==============================================================================
# 5. 命令行接口
# ==============================================================================

class InteractiveMode:
    """封装所有交互式菜单逻辑"""
    def __init__(self):
        # 定义默认配置模板，确保所有键都存在
        default_config = {
            "api_key": "",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-3.5-turbo",
            "epub_path": "",
            "output": "analysis_results_refactored.json",
            "report": "analysis_report_refactored.md",
            "max_tokens": 3000,
            "timeout": 180.0,
            "delay": 2.0,
            "prompt": self._get_default_prompt(),
            "use_json_schema": False
        }
        loaded_config = load_config_from_file(CONFIG_FILE)
        # 合并配置：以默认配置为基础，用加载的配置覆盖
        self.config_data = default_config
        self.config_data.update(loaded_config)

    def _edit_prompt_in_editor(self):
        """用外部编辑器编辑提示词"""
        prompt_file = Path("prompt_editor_temp.txt")
        try:
            # 1. 写入当前提示词到临时文件
            current_prompt = self.config_data.get('prompt', '')
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(current_prompt)

            logger.info("即将打开记事本编辑提示词。请在记事本中修改、保存，然后关闭窗口。")
            time.sleep(2) # 让用户有时间阅读信息

            # 2. 调用记事本并等待它关闭
            if platform.system() == "Windows":
                subprocess.run(["notepad.exe", str(prompt_file)], check=True)
            else:
                logger.warning("此功能在非Windows系统上将尝试打开默认编辑器。")
                editor = os.getenv('EDITOR')
                if editor:
                    subprocess.run([editor, str(prompt_file)], check=True)
                else:
                    logger.error("无法找到默认文本编辑器。请设置您的 EDITOR 环境变量。")
                    return

            # 3. 读取修改后的内容
            with open(prompt_file, 'r', encoding='utf-8') as f:
                new_prompt = f.read()

            # 4. 更新配置
            if new_prompt.strip() != current_prompt.strip():
                self.config_data['prompt'] = new_prompt
                logger.info("✅ 提示词已更新。")
            else:
                logger.info("提示词未发生变化。")

        except FileNotFoundError:
             logger.error(f"❌ 错误：找不到默认的文本编辑器。在Windows上，请确保'notepad.exe'在系统路径中。")
        except Exception as e:
            logger.error(f"❌ 编辑提示词时发生错误: {e}")
        finally:
            # 5. 清理临时文件
            if prompt_file.exists():
                prompt_file.unlink()

    def run(self):
        """启动交互模式的主循环"""
        while True:
            self._show_main_menu()
            choice = input("请输入您的选择: ").strip()
            if choice == '1':
                self.start_analysis_from_menu()
            elif choice == '2':
                self._show_settings_menu()
            elif choice == '3':
                self._show_help()
            elif choice == '4':
                logger.info("感谢使用，再见！")
                break
            else:
                logger.warning("无效输入，请重新选择。")

    def _get_default_prompt(self) -> str:
        # This is now a method of InteractiveMode to be used by the default_config
        return """
你现在需要分析这篇内容并完成下面的任务，你需要选取人生感悟或者是恋爱方法经验等，保留上下文。如果没有可以再金句那里设置一个空列表
请注意：
1.请勿包含故事情节：样例：
1>虽然他不知道别的小说作者是怎么样的，但他平时就喜欢去更多的地方，见更多的风景，这样才能写出来更加精彩的故事。
2>不知不觉中，苏白粥的心情变得十分安逸，之前的不愉快已经烟消云散。
2.你的目的是保存金句让人感觉有价值的，不是保留人物的无用对话等
请严格遵守以下JSON格式要求（请勿在JSON内部添加任何注释）：

<JSON>
{
    "chapter_title": "在此处填写章节标题",
    "golden_sentences": [
        {
            "sentence": "在此处填写黄金句子的纯文本内容",
            "speaker": "在此处填写句子来源（角色名/旁白等，你也可以根据上下文判断）",
            "reason": "在此处填写选择该句的恋爱理由"
        }
    ],
    "chapter_summary": "在此处填写本章的恋爱主线总结"
}
</JSON>
章节内容如下：
"""

    def _show_main_menu(self):
        print("\n" + "="*50)
        print("  EPUB小说金句分析器 - 交互模式")
        print("="*50)
        print("1. 开始分析")
        print("2. 设置")
        print("3. 帮助")
        print("4. 退出")
        print("-"*50)

    def _show_settings_menu(self):
        """显示并处理设置菜单"""
        while True:
            print("\n--- 设置菜单 ---")
            # 为了稳定的顺序，对键进行排序
            all_keys = sorted(self.config_data.keys())
            editable_keys = [k for k in all_keys if k != 'prompt']
            
            for i, key in enumerate(editable_keys, 1):
                value = self.config_data[key]
                display_value = '*****' if 'key' in key and value else value
                if isinstance(value, bool):
                    display_value = "✅ 已启用" if value else "❌ 已禁用"
                print(f"{i}. {key}: {display_value}")
            
            base_index = len(editable_keys)
            print(f"{base_index + 1}. 编辑提示词 (将打开记事本)")
            print(f"{base_index + 2}. 测试API连接")
            print(f"{base_index + 3}. 保存并返回主菜单")
            print(f"{base_index + 4}. 不保存并返回主菜单")
            
            choice = input("请选择要修改的配置项或操作: ").strip()
            
            try:
                choice_int = int(choice)
                if 1 <= choice_int <= len(editable_keys):
                    key_to_edit = editable_keys[choice_int - 1]
                    new_value_str = input(f"请输入新的 '{key_to_edit}' 值: ").strip()
                    
                    # 尝试将输入转换为原始值的类型
                    original_value = self.config_data[key_to_edit]
                    try:
                        if isinstance(original_value, bool):
                            if new_value_str.lower() in ['true', 'yes', 'y', '1', 'on']:
                                self.config_data[key_to_edit] = True
                            elif new_value_str.lower() in ['false', 'no', 'n', '0', 'off']:
                                self.config_data[key_to_edit] = False
                            else:
                                logger.warning("无效的布尔值输入。请输入 'yes' 或 'no'。")
                        elif isinstance(original_value, (int, float)):
                                self.config_data[key_to_edit] = type(original_value)(new_value_str)
                        else:
                                self.config_data[key_to_edit] = new_value_str
                    except ValueError:
                        logger.error(f"输入的值 '{new_value_str}' 类型不正确。已忽略更改。")
                elif choice_int == base_index + 1:
                    self._edit_prompt_in_editor()
                elif choice_int == base_index + 2:
                    # 测试API连接
                    temp_config = Config(self.config_data)
                    api_handler = APIHandler(temp_config, InterruptHandler())
                    api_handler.test_connection()
                    input("按回车键继续...")
                elif choice_int == base_index + 3:
                    save_config_to_file(CONFIG_FILE, self.config_data)
                    break
                elif choice_int == base_index + 4:
                    self.__init__() # 重新初始化以放弃更改
                    break
                else:
                    logger.warning("无效选择。")
            except ValueError:
                logger.warning("请输入数字。")

    def _show_help(self):
        print("\n--- 帮助信息 ---")
        print("本工具用于分析EPUB格式的小说，提取有价值的'金句'。")
        print("\n主要功能:")
        print("- 开始分析: 根据当前设置，启动分析流程。")
        print("- 设置: 配置API密钥、模型、文件路径等。")
        print("  - API密钥和模型是必需的。")
        print("  - 文件路径等可以根据需要配置。")
        print("- 命令行模式: 您也可以通过命令行参数直接运行分析，具体参数请使用 --help 查看。")
        input("\n按回车键返回主菜单...")

    def start_analysis_from_menu(self):
        """从菜单启动分析"""
        if not self.config_data.get("api_key") or not self.config_data.get("epub_path"):
            logger.error("错误: API密钥和EPUB文件路径是必需的。请先在'设置'中配置。")
            return
        
        config = Config(self.config_data)
        orchestrator = AnalysisOrchestrator(config)
        
        # 在交互模式下，默认启用续跑
        orchestrator.run_analysis(resume=True)

def main():
    parser = argparse.ArgumentParser(
        description='(Refactored) EPUB小说金句分析器',
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 调整参数，使其变为可选，以便交互模式可以启动
    parser.add_argument('epub_path', nargs='?', default=None, help='EPUB文件路径 (在命令行模式下为必需)')
    parser.add_argument('--api-key', help='OpenAI API密钥')
    parser.add_argument('--base-url', help='API基础URL')
    parser.add_argument('--output', help='输出文件路径')
    parser.add_argument('--report', help='报告文件路径')
    parser.add_argument('--no-resume', action='store_true', help='不从上次中断处恢复')
    parser.add_argument('--clean', action='store_true', help='清理所有分析文件（进度、结果、报告）')
    parser.add_argument('--max-tokens', type=int, help='API最大token数')
    parser.add_argument('--timeout', type=float, help='API超时时间（秒）')
    parser.add_argument('--delay', type=float, help='API调用间隔（秒）')
    parser.add_argument('--model', help='使用的AI模型')
    parser.add_argument('--test-mode', action='store_true', help='测试模式（仅分析前3章）')
    parser.add_argument('--re-analyze', nargs='+', help='强制重新分析一个或多个章节索引 (例如: 5 10-15 20)')
    parser.add_argument('--retry-failed', action='store_true', help='仅重试所有之前失败的章节')
    parser.add_argument('--find-missing', action='store_true', help='Verify and analyze all missing chapters, the most robust resume mode.')
    parser.add_argument('--sort-results', action='store_true', help='Load, sort, and re-save all existing result files to fix chapter order.')
    parser.add_argument('--deduplicate-results', action='store_true', help='Load, deduplicate, and re-save all results to ensure chapter uniqueness.')
    
    args = parser.parse_args()

    # 判断运行模式
    # 如果提供了epub_path或任何其他特定操作，则进入命令行模式
    if args.epub_path or any([args.clean, args.sort_results, args.deduplicate_results]):
        # 在命令行模式下，epub_path和api_key是必需的
        if not args.epub_path:
            parser.error("在命令行模式下，'epub_path' 是必需的。")
        
        # 命令行参数优先于配置文件
        config_from_file = load_config_from_file(CONFIG_FILE)
        arg_dict = {k: v for k, v in vars(args).items() if v is not None}
        config_from_file.update(arg_dict)

        if not config_from_file.get('api_key'):
             parser.error("在命令行模式下，'--api-key' 是必需的（或在config.json中提供）。")

        config = Config(config_from_file)
        orchestrator = AnalysisOrchestrator(config)

        if args.clean:
            orchestrator.file_manager.clean_all()
            return

        if args.sort_results:
            orchestrator.run_sort_results()
            return

        if args.deduplicate_results:
            orchestrator.run_deduplicate_results()
            return

        re_analyze_list = []
        if args.re_analyze:
            for item in args.re_analyze:
                if '-' in item:
                    try:
                        start, end = map(int, item.split('-'))
                        re_analyze_list.extend(range(start, end + 1))
                    except ValueError: logger.error(f"无效的章节范围格式: {item}")
                else:
                    try: re_analyze_list.append(int(item))
                    except ValueError: logger.error(f"无效的章节索引: {item}")
        
        orchestrator.run_analysis(
            resume=not args.no_resume,
            test_mode=args.test_mode,
            re_analyze_chapters=re_analyze_list if re_analyze_list else None,
            retry_failed=args.retry_failed,
            find_missing=args.find_missing
        )
    else:
        # 进入交互模式
        interactive_mode = InteractiveMode()
        interactive_mode.run()

if __name__ == "__main__":
    main()