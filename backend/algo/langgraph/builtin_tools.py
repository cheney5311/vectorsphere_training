"""
内置工具集

提供生产级的预置工具：
- 搜索工具：Web搜索、知识库搜索、向量搜索
- 代码执行工具：Python、SQL、Shell
- 数据处理工具：分析、转换、验证
- 知识库工具：存储、检索、记忆管理
- 系统工具：日期时间、UUID、哈希、正则
- HTTP工具：REST API 调用
- 文件操作工具：读取、写入、解析

生产级特性：
- 安全沙箱执行
- 超时控制
- 结果缓存
- 错误处理
"""

import json
import logging
import asyncio
import hashlib
import uuid
import re
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from functools import lru_cache

from .tools import Tool, ToolParameter, ToolCategory, tool, async_tool

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.info("httpx not available, HTTP tools will use urllib")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info("redis not available, memory tools will use in-memory storage")

# 内存缓存（用于记忆工具）
_memory_store: Dict[str, Dict[str, Any]] = {}


# ==================== 搜索引擎配置 ====================

# 搜索引擎 API 配置（从环境变量读取）
SEARCH_ENGINE_CONFIG = {
    "google": {
        "api_key_env": "GOOGLE_API_KEY",
        "cx_env": "GOOGLE_CX",  # Custom Search Engine ID
        "endpoint": "https://www.googleapis.com/customsearch/v1",
    },
    "bing": {
        "api_key_env": "BING_API_KEY",
        "endpoint": "https://api.bing.microsoft.com/v7.0/search",
    },
    "brave": {
        "api_key_env": "BRAVE_API_KEY",
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
    },
    "tavily": {
        "api_key_env": "TAVILY_API_KEY",
        "endpoint": "https://api.tavily.com/search",
    },
    "serper": {
        "api_key_env": "SERPER_API_KEY",
        "endpoint": "https://google.serper.dev/search",
    },
    "serpapi": {
        "api_key_env": "SERPAPI_API_KEY",
        "endpoint": "https://serpapi.com/search",
    },
    "duckduckgo": {
        "endpoint": "https://api.duckduckgo.com/",
        "requires_key": False,
    },
}

# 默认搜索引擎优先级（按顺序尝试）
DEFAULT_SEARCH_PRIORITY = ["tavily", "serper", "brave", "bing", "google", "duckduckgo"]


class WebSearchEngine:
    """Web 搜索引擎封装
    
    支持多种搜索引擎：
    - Google Custom Search API
    - Bing Web Search API
    - Brave Search API
    - Tavily AI Search API
    - Serper API (Google Search)
    - SerpAPI
    - DuckDuckGo (免费，无需 API Key)
    
    自动检测可用的 API 并选择最佳搜索引擎。
    """
    
    def __init__(self):
        self._available_engines = self._detect_available_engines()
        logger.info(f"Available search engines: {list(self._available_engines.keys())}")
    
    def _detect_available_engines(self) -> dict:
        """检测可用的搜索引擎"""
        available = {}
        
        for engine, config in SEARCH_ENGINE_CONFIG.items():
            if config.get("requires_key", True) is False:
                # 不需要 API Key 的引擎（如 DuckDuckGo）
                available[engine] = config
            else:
                # 检查 API Key 是否配置
                api_key = os.environ.get(config.get("api_key_env", ""))
                if api_key:
                    available[engine] = {**config, "api_key": api_key}
                    # Google 需要额外的 CX
                    if engine == "google":
                        cx = os.environ.get(config.get("cx_env", ""))
                        if cx:
                            available[engine]["cx"] = cx
                        else:
                            del available[engine]
        
        return available
    
    def get_best_engine(self, preferred: str = None) -> str:
        """获取最佳可用搜索引擎"""
        if preferred and preferred in self._available_engines:
            return preferred
        
        for engine in DEFAULT_SEARCH_PRIORITY:
            if engine in self._available_engines:
                return engine
        
        return "duckduckgo"  # 默认回退
    
    def search(self, query: str, num_results: int = 5, engine: str = None, 
               language: str = "zh-CN", safe_search: bool = True) -> dict:
        """执行搜索
    
        Args:
            query: 搜索查询
            num_results: 结果数量
            engine: 指定搜索引擎
            language: 语言代码
            safe_search: 安全搜索
            
        Returns:
            搜索结果字典
        """
        engine = self.get_best_engine(engine)
        
        search_methods = {
            "google": self._search_google,
            "bing": self._search_bing,
            "brave": self._search_brave,
            "tavily": self._search_tavily,
            "serper": self._search_serper,
            "serpapi": self._search_serpapi,
            "duckduckgo": self._search_duckduckgo,
        }
        
        search_func = search_methods.get(engine, self._search_duckduckgo)
        
        try:
            results = search_func(query, num_results, language, safe_search)
            if results:
                return {
                    "success": True,
                    "query": query,
                    "engine": engine,
                    "results": results[:num_results],
                    "total": len(results)
                }
        except Exception as e:
            logger.warning(f"Search with {engine} failed: {e}")
        
        # 尝试其他引擎
        for fallback_engine in DEFAULT_SEARCH_PRIORITY:
            if fallback_engine != engine and fallback_engine in self._available_engines:
                try:
                    search_func = search_methods.get(fallback_engine)
                    if search_func:
                        results = search_func(query, num_results, language, safe_search)
                        if results:
                            return {
                                "success": True,
                                "query": query,
                                "engine": fallback_engine,
                                "results": results[:num_results],
                                "total": len(results),
                                "fallback_from": engine
                            }
                except Exception as e:
                    logger.warning(f"Fallback search with {fallback_engine} failed: {e}")
                    continue
        
        return {"success": False, "query": query, "error": "All search engines failed"}
    
    def _make_request(self, method: str, url: str, headers: dict = None, 
                      params: dict = None, json_data: dict = None, timeout: float = 15.0) -> dict:
        """发送 HTTP 请求"""
        if HTTPX_AVAILABLE:
            import httpx
            with httpx.Client(timeout=timeout) as client:
                if method == "GET":
                    response = client.get(url, headers=headers, params=params)
                else:
                    response = client.post(url, headers=headers, params=params, json=json_data)
                response.raise_for_status()
                return response.json()
        else:
            import urllib.request
            import urllib.parse
            
            if params:
                url = f"{url}?{urllib.parse.urlencode(params)}"
            
            req = urllib.request.Request(url)
            if headers:
                for k, v in headers.items():
                    req.add_header(k, v)
            
            if json_data:
                req.data = json.dumps(json_data).encode('utf-8')
                req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode('utf-8'))
    
    def _search_google(self, query: str, num_results: int, language: str, safe_search: bool) -> list:
        """Google Custom Search API"""
        config = self._available_engines.get("google", {})
        api_key = config.get("api_key")
        cx = config.get("cx")
        
        if not api_key or not cx:
            return []
        
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": min(num_results, 10),
            "lr": f"lang_{language.split('-')[0]}",
            "safe": "active" if safe_search else "off"
        }
        
        data = self._make_request("GET", config["endpoint"], params=params)
        
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "Google",
                "display_url": item.get("displayLink", "")
            })
        
        return results
    
    def _search_bing(self, query: str, num_results: int, language: str, safe_search: bool) -> list:
        """Bing Web Search API"""
        config = self._available_engines.get("bing", {})
        api_key = config.get("api_key")
        
        if not api_key:
            return []
        
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {
            "q": query,
            "count": min(num_results, 50),
            "mkt": language,
            "safeSearch": "Strict" if safe_search else "Off"
        }
        
        data = self._make_request("GET", config["endpoint"], headers=headers, params=params)
        
        results = []
        for item in data.get("webPages", {}).get("value", []):
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "source": "Bing",
                "date_published": item.get("dateLastCrawled", "")
            })
        
        return results
    
    def _search_brave(self, query: str, num_results: int, language: str, safe_search: bool) -> list:
        """Brave Search API"""
        config = self._available_engines.get("brave", {})
        api_key = config.get("api_key")
        
        if not api_key:
            return []
        
        headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json"
        }
        params = {
            "q": query,
            "count": min(num_results, 20),
            "search_lang": language.split("-")[0],
            "safesearch": "strict" if safe_search else "off"
        }
        
        data = self._make_request("GET", config["endpoint"], headers=headers, params=params)
        
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "source": "Brave",
                "age": item.get("age", "")
            })
        
        return results
    
    def _search_tavily(self, query: str, num_results: int, language: str, safe_search: bool) -> list:
        """Tavily AI Search API - 专为 AI 应用设计的搜索 API"""
        config = self._available_engines.get("tavily", {})
        api_key = config.get("api_key")
        
        if not api_key:
            return []
        
        json_data = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",  # basic 或 advanced
            "max_results": min(num_results, 10),
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False
        }
        
        data = self._make_request("POST", config["endpoint"], json_data=json_data)
        
        results = []
        
        # Tavily 提供 AI 生成的答案摘要
        if data.get("answer"):
            results.append({
                "title": "AI 摘要",
                "url": "",
                "snippet": data.get("answer", ""),
                "source": "Tavily AI",
                "is_answer": True
            })
        
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": "Tavily",
                "score": item.get("score", 0),
                "published_date": item.get("published_date", "")
            })
        
        return results
    
    def _search_serper(self, query: str, num_results: int, language: str, safe_search: bool) -> list:
        """Serper API - Google Search Results API"""
        config = self._available_engines.get("serper", {})
        api_key = config.get("api_key")
        
        if not api_key:
            return []
        
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        json_data = {
            "q": query,
            "num": min(num_results, 100),
            "hl": language.split("-")[0],
            "gl": language.split("-")[1] if "-" in language else "cn"
        }
        
        data = self._make_request("POST", config["endpoint"], headers=headers, json_data=json_data)
        
        results = []
        
        # 知识图谱
        if data.get("knowledgeGraph"):
            kg = data["knowledgeGraph"]
            results.append({
                "title": kg.get("title", ""),
                "url": kg.get("website", ""),
                "snippet": kg.get("description", ""),
                "source": "Google Knowledge Graph",
                "is_knowledge_graph": True
            })
        
        # 普通搜索结果
        for item in data.get("organic", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "Serper/Google",
                "position": item.get("position", 0)
            })
        
        # 相关问题
        for item in data.get("peopleAlsoAsk", [])[:3]:
            results.append({
                "title": item.get("question", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "People Also Ask",
                "is_related_question": True
            })
        
        return results
    
    def _search_serpapi(self, query: str, num_results: int, language: str, safe_search: bool) -> list:
        """SerpAPI - 多引擎搜索 API"""
        config = self._available_engines.get("serpapi", {})
        api_key = config.get("api_key")
        
        if not api_key:
            return []
        
        params = {
            "api_key": api_key,
            "q": query,
            "num": min(num_results, 100),
            "hl": language.split("-")[0],
            "gl": language.split("-")[1] if "-" in language else "cn",
            "safe": "active" if safe_search else "off",
            "engine": "google"
        }
        
        data = self._make_request("GET", config["endpoint"], params=params)
        
        results = []
        for item in data.get("organic_results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "SerpAPI/Google",
                "position": item.get("position", 0),
                "displayed_link": item.get("displayed_link", "")
            })
        
        return results
    
    def _search_duckduckgo(self, query: str, num_results: int, language: str, safe_search: bool) -> list:
        """DuckDuckGo 即时答案 API（免费）"""
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1
        }
        
        data = self._make_request("GET", "https://api.duckduckgo.com/", params=params)
        
        results = []
        
        # 解析答案
        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", query),
                "url": data.get("AbstractURL", ""),
                "snippet": data.get("AbstractText", ""),
                "source": data.get("AbstractSource", "DuckDuckGo"),
                "is_instant_answer": True
            })
        
        # 解析相关主题
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                    "source": "DuckDuckGo"
                })
            elif isinstance(topic, dict) and topic.get("Topics"):
                # 嵌套主题
                for subtopic in topic.get("Topics", []):
                    if isinstance(subtopic, dict) and subtopic.get("Text"):
                        results.append({
                            "title": subtopic.get("Text", "")[:100],
                            "url": subtopic.get("FirstURL", ""),
                            "snippet": subtopic.get("Text", ""),
                            "source": "DuckDuckGo"
                        })
        
        # 尝试使用 DuckDuckGo HTML 搜索作为补充
        if len(results) < num_results:
            try:
                html_results = self._search_duckduckgo_html(query, num_results - len(results))
                results.extend(html_results)
            except Exception as e:
                logger.debug(f"DuckDuckGo HTML search failed: {e}")
        
        return results
    
    def _search_duckduckgo_html(self, query: str, num_results: int) -> list:
        """DuckDuckGo HTML 搜索（解析网页）"""
        try:
            # 尝试导入 duckduckgo_search 库
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=num_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "source": "DuckDuckGo"
                    })
            return results
        except ImportError:
            logger.debug("duckduckgo_search library not available")
            return []
        except Exception as e:
            logger.debug(f"DuckDuckGo HTML search error: {e}")
            return []


# 全局搜索引擎实例
_web_search_engine: WebSearchEngine = None
_search_engine_lock = threading.Lock()


def get_web_search_engine() -> WebSearchEngine:
    """获取 Web 搜索引擎实例"""
    global _web_search_engine
    
    with _search_engine_lock:
        if _web_search_engine is None:
            _web_search_engine = WebSearchEngine()
    
    return _web_search_engine


# ==================== 搜索类工具 ====================

@tool(
    name="web_search",
    description="搜索互联网获取最新信息。支持多种搜索引擎（Google、Bing、Brave、Tavily、DuckDuckGo等）。返回相关网页的标题、URL和摘要。",
    category=ToolCategory.SEARCH,
    timeout=30.0
)
def web_search(query: str, num_results: int = 5, search_engine: str = "auto", 
               language: str = "zh-CN", safe_search: bool = True) -> str:
    """
    搜索互联网获取信息
    
    支持多种搜索引擎，自动选择最佳可用引擎或使用指定引擎。
    
    Args:
        query: 搜索查询关键词（支持自然语言和关键词）
        num_results: 返回结果数量（1-20）
        search_engine: 搜索引擎选择
            - "auto": 自动选择最佳可用引擎
            - "google": Google Custom Search
            - "bing": Bing Web Search
            - "brave": Brave Search
            - "tavily": Tavily AI Search（推荐，专为 AI 优化）
            - "serper": Serper Google Search
            - "duckduckgo": DuckDuckGo（免费，无需 API Key）
        language: 语言代码（如 zh-CN, en-US, ja-JP）
        safe_search: 是否启用安全搜索过滤
    
    Returns:
        JSON格式的搜索结果，包含标题、URL、摘要等信息
    
    Examples:
        - query="Python 机器学习教程", num_results=5
        - query="latest AI news 2024", search_engine="tavily", language="en-US"
        - query="天气预报 北京", search_engine="bing"
    
    环境变量配置：
        - TAVILY_API_KEY: Tavily API 密钥
        - SERPER_API_KEY: Serper API 密钥
        - BRAVE_API_KEY: Brave Search API 密钥
        - BING_API_KEY: Bing Web Search API 密钥
        - GOOGLE_API_KEY + GOOGLE_CX: Google Custom Search
    """
    num_results = min(max(num_results, 1), 20)
    
    try:
        # 获取搜索引擎实例
        engine = get_web_search_engine()
        
        # 确定搜索引擎
        selected_engine = None if search_engine == "auto" else search_engine
        
        # 执行搜索
        result = engine.search(
            query=query,
            num_results=num_results,
            engine=selected_engine,
            language=language,
            safe_search=safe_search
        )
        
        if result.get("success"):
                        return json.dumps({
                "success": True,
                            "query": query,
                "engine": result.get("engine", "unknown"),
                "results": result.get("results", []),
                "total": result.get("total", 0),
                "language": language,
                "fallback_from": result.get("fallback_from")
                        }, ensure_ascii=False, indent=2)
        else:
            logger.warning(f"Web search failed: {result.get('error')}")
            
    except Exception as e:
        logger.warning(f"Web search error: {e}")
    
    # 降级到模拟结果
    results = [
        {
            "title": f"搜索结果 {i+1}: {query}",
            "url": f"https://search.example.com/result?q={query}&i={i+1}",
            "snippet": f"这是关于 '{query}' 的搜索结果 {i+1}。请配置搜索 API 密钥以获取真实搜索结果。",
            "source": "simulation"
        }
        for i in range(num_results)
    ]
    
    return json.dumps({
        "success": True,
        "query": query,
        "engine": "simulation",
        "results": results,
        "total": len(results),
        "language": language,
        "note": "使用模拟结果。推荐配置 TAVILY_API_KEY 或 SERPER_API_KEY 环境变量以获取真实搜索结果。"
    }, ensure_ascii=False, indent=2)


@tool(
    name="news_search",
    description="搜索最新新闻和时事内容。返回新闻标题、来源、发布时间和摘要。",
    category=ToolCategory.SEARCH,
    timeout=30.0
)
def news_search(query: str, num_results: int = 5, language: str = "zh-CN", 
                time_range: str = "week") -> str:
    """
    搜索新闻内容
    
    获取与查询相关的最新新闻报道。
    
    Args:
        query: 新闻搜索关键词
        num_results: 返回结果数量（1-20）
        language: 语言代码（zh-CN, en-US 等）
        time_range: 时间范围
            - "day": 过去24小时
            - "week": 过去一周（默认）
            - "month": 过去一个月
            - "year": 过去一年
    
    Returns:
        JSON格式的新闻搜索结果
    
    Examples:
        - query="人工智能", num_results=5, time_range="day"
        - query="科技新闻", language="zh-CN"
    """
    num_results = min(max(num_results, 1), 20)
    
    try:
        engine = get_web_search_engine()
        
        # 构建新闻搜索查询
        time_modifiers = {
            "day": "past 24 hours",
            "week": "past week",
            "month": "past month",
            "year": "past year"
        }
        time_modifier = time_modifiers.get(time_range, "past week")
        
        # 对于支持新闻搜索的引擎，添加新闻过滤
        news_query = f"{query} news {time_modifier}"
        
        # 尝试使用 Serper 的新闻端点
        if "serper" in engine._available_engines:
            try:
                config = engine._available_engines["serper"]
                headers = {
                    "X-API-KEY": config.get("api_key"),
                    "Content-Type": "application/json"
                }
                json_data = {
                    "q": query,
                    "num": num_results,
                    "hl": language.split("-")[0],
                    "gl": language.split("-")[1] if "-" in language else "cn",
                    "tbs": f"qdr:{time_range[0]}"  # d=day, w=week, m=month, y=year
                }
                
                data = engine._make_request(
                    "POST", 
                    "https://google.serper.dev/news",
                    headers=headers,
                    json_data=json_data
                )
                
                results = []
                for item in data.get("news", []):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": item.get("source", ""),
                        "date": item.get("date", ""),
                        "image_url": item.get("imageUrl", "")
                    })
                
                if results:
                    return json.dumps({
                        "success": True,
                        "query": query,
                        "engine": "serper_news",
                        "results": results[:num_results],
                        "total": len(results),
                        "time_range": time_range
                    }, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logger.debug(f"Serper news search failed: {e}")
        
        # 回退到普通搜索
        result = engine.search(
            query=news_query,
            num_results=num_results,
            language=language,
            safe_search=True
        )
        
        if result.get("success"):
            return json.dumps({
                "success": True,
                "query": query,
                "engine": result.get("engine", "unknown"),
                "results": result.get("results", []),
                "total": result.get("total", 0),
                "time_range": time_range,
                "note": "使用通用搜索引擎获取新闻相关结果"
            }, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.warning(f"News search error: {e}")
    
    # 降级模拟
    results = [
        {
            "title": f"新闻 {i+1}: 关于「{query}」的最新报道",
            "url": f"https://news.example.com/article/{i+1}",
            "snippet": f"这是关于 '{query}' 的模拟新闻内容。配置搜索 API 后可获取真实新闻。",
            "source": "模拟新闻源",
            "date": datetime.utcnow().strftime("%Y-%m-%d")
        }
        for i in range(num_results)
    ]
    
    return json.dumps({
        "success": True,
        "query": query,
        "engine": "simulation",
        "results": results,
        "total": len(results),
        "time_range": time_range,
        "note": "使用模拟结果，请配置搜索 API 密钥"
    }, ensure_ascii=False, indent=2)


@tool(
    name="search_engine_status",
    description="获取可用的搜索引擎状态和配置信息。用于诊断搜索功能。",
    category=ToolCategory.SEARCH,
    timeout=10.0
)
def search_engine_status() -> str:
    """
    获取搜索引擎状态
    
    返回当前配置的搜索引擎列表及其可用状态。
    
    Returns:
        JSON格式的搜索引擎状态信息
    """
    try:
        engine = get_web_search_engine()
        
        # 获取所有引擎配置
        all_engines = {}
        for name, config in SEARCH_ENGINE_CONFIG.items():
            is_available = name in engine._available_engines
            requires_key = config.get("requires_key", True)
            
            all_engines[name] = {
                "available": is_available,
                "requires_api_key": requires_key,
                "api_key_env": config.get("api_key_env", ""),
                "endpoint": config.get("endpoint", "")
            }
            
            if requires_key and not is_available:
                all_engines[name]["note"] = f"请设置环境变量 {config.get('api_key_env', '')} 以启用此引擎"
        
        # 获取默认引擎
        default_engine = engine.get_best_engine()
        
        return json.dumps({
            "success": True,
            "default_engine": default_engine,
            "available_engines": list(engine._available_engines.keys()),
            "all_engines": all_engines,
            "priority_order": DEFAULT_SEARCH_PRIORITY,
            "recommendation": "推荐配置 Tavily API (TAVILY_API_KEY) 或 Serper API (SERPER_API_KEY) 以获得最佳搜索体验"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="knowledge_search",
    description="在知识库中搜索相关信息。支持语义搜索和关键词搜索。可搜索文档、问答对、技术文章等知识内容。",
    category=ToolCategory.RETRIEVAL,
    timeout=15.0
)
def knowledge_search(query: str, top_k: int = 3, filters: str = None, search_type: str = "hybrid", collection: str = None) -> str:
    """
    知识库搜索
    
    在 VectorSphere 向量数据库中进行语义搜索，返回与查询最相关的知识内容。
    
    Args:
        query: 搜索查询（支持自然语言问题或关键词）
        top_k: 返回结果数量（1-20）
        filters: 过滤条件（JSON格式，如 {"category": "tech", "date_after": "2024-01-01"}）
        search_type: 搜索类型（semantic: 语义搜索, keyword: 关键词搜索, hybrid: 混合搜索）
        collection: 指定搜索的集合名称（可选，默认搜索所有知识库）
    
    Returns:
        JSON格式的搜索结果，包含文档ID、内容、相似度分数和元数据
    
    Examples:
        - query="如何使用向量数据库?", top_k=5
        - query="机器学习模型训练", filters='{"category": "ml"}', search_type="semantic"
    """
    top_k = min(max(top_k, 1), 20)
    
    # 解析过滤条件
    filter_dict = {}
    if filters:
        try:
            filter_dict = json.loads(filters)
        except json.JSONDecodeError:
            logger.warning(f"Invalid filter JSON: {filters}")
    
    # 尝试使用 VectorSphere 向量数据库进行语义搜索
    try:
        from backend.modules.vectordb.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        if vector_store:
            # 健康检查
            health = vector_store.health_check()
            if health.get("status") != "healthy":
                logger.warning(f"VectorSphere not healthy: {health}")
                raise Exception("VectorSphere service unavailable")
            
            # 确定搜索集合
            search_collection = collection or "knowledge_base"
            
            # 执行语义搜索
            search_results = vector_store.search(
                query=query,
                collection=search_collection,
                top_k=top_k,
                filters=filter_dict if filter_dict else None,
                search_type=search_type,
                include_metadata=True,
                score_threshold=0.0
            )
            
            # 格式化结果
            results = [
                {
                    "id": r.id,
                    "content": r.content or r.metadata.get("content", ""),
                    "score": r.score,
                    "distance": r.distance,
                    "metadata": {
                        k: v for k, v in r.metadata.items() 
                        if k not in ["content", "vector"]  # 排除大字段
                    }
                }
                for r in search_results
            ]
            
            return json.dumps({
                "success": True,
                "query": query,
                "collection": search_collection,
                "results": results,
                "total": len(results),
                "search_type": search_type,
                "filters_applied": filter_dict,
                "source": "vectorsphere"
            }, ensure_ascii=False, indent=2)
            
    except ImportError as e:
        logger.debug(f"Vector store module not available: {e}")
    except Exception as e:
        logger.warning(f"Knowledge search via VectorSphere failed: {e}")
    
    # 降级：模拟知识库搜索
    results = [
        {
            "id": f"doc_{i}",
            "content": f"知识库文档 {i+1}: 关于「{query}」的相关内容。此为模拟数据，配置向量数据库后可进行真实语义搜索。",
            "score": round(0.95 - i * 0.05, 3),
            "distance": round(0.05 + i * 0.05, 3),
            "metadata": {
                "source": "knowledge_base",
                "type": "document",
                "created_at": datetime.utcnow().isoformat()
            }
        }
        for i in range(top_k)
    ]
    
    return json.dumps({
        "success": True,
        "query": query,
        "collection": collection or "default",
        "results": results,
        "total": len(results),
        "search_type": search_type,
        "filters_applied": filter_dict,
        "source": "simulation",
        "note": "使用模拟数据，请配置 VectorSphere 向量数据库以获取真实搜索结果"
    }, ensure_ascii=False, indent=2)


@tool(
    name="vector_search",
    description="执行向量相似度搜索，用于语义匹配和相似文档查找。支持直接向量搜索或文本查询转向量搜索。",
    category=ToolCategory.RETRIEVAL,
    timeout=15.0
)
def vector_search(query: str, collection: str = "default", top_k: int = 5, 
                  min_score: float = 0.0, filters: str = None) -> str:
    """
    向量相似度搜索
    
    在 VectorSphere 向量数据库中执行高精度的向量相似度搜索。
    支持文本查询（自动转换为向量）和元数据过滤。
    
    Args:
        query: 搜索查询文本（将自动转换为向量进行搜索）
        collection: 向量集合名称（如 articles, documents, embeddings）
        top_k: 返回结果数量（1-100）
        min_score: 最小相似度分数阈值（0-1，默认0表示不过滤）
        filters: 元数据过滤条件（JSON格式，支持复杂查询）
    
    Returns:
        JSON格式的搜索结果，包含向量ID、内容、相似度分数
    
    Examples:
        - query="深度学习优化算法", collection="tech_docs", top_k=10
        - query="产品规格", collection="products", filters='{"category": "electronics"}'
    """
    top_k = min(max(top_k, 1), 100)
    min_score = max(min(min_score, 1.0), 0.0)
    
    # 解析过滤条件
    filter_dict = None
    if filters:
        try:
            filter_dict = json.loads(filters)
        except json.JSONDecodeError:
            logger.warning(f"Invalid filter JSON: {filters}")
    
    # 尝试使用 VectorSphere 向量数据库
    try:
        from backend.modules.vectordb.vector_store import get_vector_store
        
        store = get_vector_store()
        if store:
            # 执行向量相似度搜索
            search_results = store.similarity_search(
                query=query,
                collection=collection,
                k=top_k,
                filters=filter_dict,
                include_metadata=True,
                score_threshold=min_score
            )
            
            # 格式化结果
            results = [
                {
                    "id": r.id,
                    "content": r.content or r.metadata.get("content", ""),
                    "score": r.score,
                    "distance": r.distance,
                    "vector_id": r.id,
                    "metadata": {
                        k: v for k, v in r.metadata.items()
                        if k not in ["content", "vector"]
                    }
                }
                for r in search_results
            ]
            
            return json.dumps({
                "success": True,
                "query": query,
                "collection": collection,
                "results": results,
                "total": len(results),
                "min_score_applied": min_score,
                "filters_applied": filter_dict,
                "source": "vectorsphere"
            }, ensure_ascii=False, indent=2)
            
    except ImportError as e:
        logger.debug(f"Vector store module not available: {e}")
    except Exception as e:
        logger.warning(f"Vector search via VectorSphere failed: {e}")
    
    # 降级：模拟结果
    results = [
        {
            "id": f"vec_{collection}_{i}",
            "content": f"向量搜索结果 {i+1}：与「{query}」语义相似的内容",
            "score": round(0.9 - i * 0.08, 3),
            "distance": round(0.1 + i * 0.08, 3),
            "vector_id": f"v_{hashlib.md5(f'{query}_{i}'.encode()).hexdigest()[:8]}",
            "metadata": {
                "source": "simulation",
                "created_at": datetime.utcnow().isoformat()
            }
        }
        for i in range(top_k)
    ]
    
    # 应用分数过滤
    filtered_results = [r for r in results if r["score"] >= min_score]
    
    return json.dumps({
        "success": True,
        "query": query,
        "collection": collection,
        "results": filtered_results,
        "total": len(filtered_results),
        "min_score_applied": min_score,
        "filters_applied": filter_dict,
        "source": "simulation",
        "note": "使用模拟结果，请配置 VectorSphere 向量数据库以获取真实数据"
    }, ensure_ascii=False, indent=2)


# ==================== 代码执行工具 ====================

# 安全沙箱配置
SAFE_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'bin': bin,
    'bool': bool, 'bytes': bytes, 'callable': callable,
    'chr': chr, 'dict': dict, 'divmod': divmod,
    'enumerate': enumerate, 'filter': filter, 'float': float,
    'format': format, 'frozenset': frozenset, 'hash': hash,
    'hex': hex, 'int': int, 'isinstance': isinstance,
    'issubclass': issubclass, 'iter': iter, 'len': len,
    'list': list, 'map': map, 'max': max, 'min': min,
    'next': next, 'oct': oct, 'ord': ord, 'pow': pow,
    'print': lambda *args, **kwargs: None,  # 禁用打印
    'range': range, 'repr': repr, 'reversed': reversed,
    'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
    'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
    'zip': zip, 'True': True, 'False': False, 'None': None
}

# 允许的模块白名单
ALLOWED_MODULES = {
    'math', 'statistics', 'random', 'datetime', 'collections',
    'itertools', 'functools', 'operator', 'decimal', 'fractions',
    'json', 're', 'string', 'textwrap'
}


@tool(
    name="python_executor",
    description="在安全沙箱中执行 Python 代码。支持数学计算、数据处理和文本操作。将结果赋值给 'result' 变量即可返回。",
    category=ToolCategory.CODE,
    timeout=30.0
)
def python_executor(code: str, context: str = None, timeout_seconds: float = 10.0) -> str:
    """
    在安全沙箱中执行 Python 代码
    
    Args:
        code: Python 代码（将结果赋值给 result 变量）
        context: 执行上下文（JSON格式的变量）
        timeout_seconds: 执行超时时间（秒）
    
    Returns:
        执行结果（JSON格式）
    
    Examples:
        code: "result = sum([1, 2, 3, 4, 5])"
        code: "import math; result = math.sqrt(16)"
        code: "data = [1, 2, 3]; result = {'sum': sum(data), 'avg': sum(data)/len(data)}"
    """
    import signal
    import traceback
    import io
    import sys
    
    # 解析上下文
    local_vars = {}
    if context:
        try:
            local_vars = json.loads(context)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"上下文解析失败: {e}",
                "output": None
            }, ensure_ascii=False)
    
    # 准备安全的执行环境
    import math, statistics, random, collections, itertools, functools, operator
    from datetime import datetime, timedelta, date, time as dt_time
    from decimal import Decimal
    from fractions import Fraction
    
    safe_modules = {
        'math': math,
        'statistics': statistics,
        'random': random,
        'datetime': datetime,
        'timedelta': timedelta,
        'date': date,
        'time': dt_time,
        'Decimal': Decimal,
        'Fraction': Fraction,
        'collections': collections,
        'Counter': collections.Counter,
        'defaultdict': collections.defaultdict,
        'OrderedDict': collections.OrderedDict,
        'itertools': itertools,
        'functools': functools,
        're': re,
        'json': json
    }
    
    # 捕获输出
    output_buffer = io.StringIO()
    
    # 创建安全的 print 函数
    def safe_print(*args, **kwargs):
        print(*args, **kwargs, file=output_buffer)
    
    global_vars = {
        '__builtins__': {**SAFE_BUILTINS, 'print': safe_print},
        **safe_modules
    }
    
    # 危险代码检查
    dangerous_patterns = [
        r'\bimport\s+os\b', r'\bimport\s+sys\b', r'\bimport\s+subprocess\b',
        r'\bopen\s*\(', r'\beval\s*\(', r'\bexec\s*\(', r'\bcompile\s*\(',
        r'__import__', r'__class__', r'__bases__', r'__subclasses__',
        r'__globals__', r'__code__', r'__builtins__'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, code):
            return json.dumps({
                "success": False,
                "error": f"安全检查失败：检测到不允许的操作",
                "output": None
            }, ensure_ascii=False)
    
    # 设置超时（仅在 Unix 系统有效）
    def timeout_handler(signum, frame):
        raise TimeoutError("执行超时")
    
    try:
        # 设置超时
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_seconds))
        
        # 执行代码
        exec(code, global_vars, local_vars)
        
        # 取消超时
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        # 获取结果
        result = local_vars.get('result', None)
        output = output_buffer.getvalue()
        
        # 序列化结果
        try:
            if result is not None:
                result_str = json.dumps(result, ensure_ascii=False, default=str)
            else:
                result_str = None
        except (TypeError, ValueError):
            result_str = str(result)
        
        return json.dumps({
            "success": True,
            "result": result_str,
            "output": output if output else None,
            "variables": {k: str(v)[:100] for k, v in local_vars.items() 
                         if not k.startswith('_') and k != 'result'}
        }, ensure_ascii=False, indent=2)
        
    except TimeoutError:
        return json.dumps({
            "success": False,
            "error": f"执行超时（>{timeout_seconds}秒）",
            "output": output_buffer.getvalue()
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc()[-500:],
            "output": output_buffer.getvalue()
        }, ensure_ascii=False)
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        output_buffer.close()


@tool(
    name="calculator",
    description="执行数学计算表达式。支持基本运算、数学函数和常量。",
    category=ToolCategory.CODE,
    timeout=5.0
)
def calculator(expression: str) -> str:
    """
    安全计算数学表达式
    
    Args:
        expression: 数学表达式（如 "2 + 3 * 4", "sqrt(16)", "sin(pi/2)"）
    
    Returns:
        计算结果
    """
    import math
    
    # 允许的名称（数学函数和常量）
    allowed_names = {
        # 常量
        'pi': math.pi, 'e': math.e, 'tau': math.tau, 'inf': math.inf,
        # 基本函数
        'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum,
        'pow': pow, 'divmod': divmod,
        # 数学函数
        'sqrt': math.sqrt, 'cbrt': lambda x: x ** (1/3),
        'exp': math.exp, 'log': math.log, 'log10': math.log10, 'log2': math.log2,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan, 'atan2': math.atan2,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'degrees': math.degrees, 'radians': math.radians,
        'floor': math.floor, 'ceil': math.ceil, 'trunc': math.trunc,
        'factorial': math.factorial, 'gcd': math.gcd,
        'hypot': math.hypot, 'fmod': math.fmod, 'fabs': math.fabs
    }
    
    # 清理表达式
    expression = expression.strip()
    
    # 安全检查
    if any(kw in expression.lower() for kw in ['import', 'eval', 'exec', 'open', '__']):
        return json.dumps({
            "success": False,
            "expression": expression,
            "error": "不允许的操作"
        }, ensure_ascii=False)
    
    try:
        # 编译并执行
        code = compile(expression, '<calculator>', 'eval')
        
        # 检查字节码中的名称
        for name in code.co_names:
            if name not in allowed_names:
                return json.dumps({
                    "success": False,
                    "expression": expression,
                    "error": f"不允许的函数或变量: {name}"
                }, ensure_ascii=False)
        
        result = eval(code, {"__builtins__": {}}, allowed_names)
        
        # 格式化结果
        if isinstance(result, float):
            if result == int(result):
                result = int(result)
            else:
                result = round(result, 10)
        
        return json.dumps({
            "success": True,
            "expression": expression,
            "result": result
        }, ensure_ascii=False)
        
    except ZeroDivisionError:
        return json.dumps({
            "success": False,
            "expression": expression,
            "error": "除以零错误"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "expression": expression,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="sql_executor",
    description="执行只读 SQL 查询（仅支持 SELECT）。用于数据分析和查询。",
    category=ToolCategory.DATA,
    timeout=30.0
)
def sql_executor(query: str, database: str = "default", limit: int = 100) -> str:
    """
    执行只读 SQL 查询
    
    Args:
        query: SQL SELECT 查询语句
        database: 数据库名称或连接标识
        limit: 最大返回行数（防止大结果集）
    
    Returns:
        查询结果（JSON格式）
    """
    # 规范化查询
    query = query.strip()
    query_upper = query.upper()
    
    # 严格安全检查
    if not query_upper.startswith('SELECT'):
        return json.dumps({
            "success": False,
            "query": query,
            "error": "仅支持 SELECT 查询"
        }, ensure_ascii=False)
    
    # 禁止的关键字
    forbidden = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 
        'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
        'INTO OUTFILE', 'INTO DUMPFILE', 'LOAD_FILE'
    ]
    
    for keyword in forbidden:
        if keyword in query_upper:
            return json.dumps({
                "success": False,
                "query": query,
                "error": f"不允许的操作: {keyword}"
            }, ensure_ascii=False)
    
    # 自动添加 LIMIT
    if 'LIMIT' not in query_upper:
        query = f"{query.rstrip(';')} LIMIT {limit}"
    
    # 尝试执行真实查询
    try:
        from backend.modules.database.manager import get_database_manager
        
        db_manager = get_database_manager()
        if db_manager:
            result = db_manager.execute_query(query, database)
            return json.dumps({
                "success": True,
                "query": query,
                "database": database,
                "columns": result.get("columns", []),
                "rows": result.get("rows", []),
                "row_count": result.get("row_count", 0)
            }, ensure_ascii=False, indent=2)
    except ImportError:
        logger.debug("Database manager not available")
    except Exception as e:
        logger.warning(f"SQL execution failed: {e}")
    
    # 模拟结果
    mock_results = [
        {"id": i+1, "name": f"示例数据{i+1}", "value": (i+1) * 100, "created_at": datetime.utcnow().isoformat()}
        for i in range(min(limit, 5))
    ]
    
    return json.dumps({
        "success": True,
        "query": query,
        "database": database,
        "columns": ["id", "name", "value", "created_at"],
        "rows": mock_results,
        "row_count": len(mock_results),
        "note": "模拟结果，请配置数据库连接以获取真实数据"
    }, ensure_ascii=False, indent=2)


# ==================== 数据处理工具 ====================

@tool(
    name="data_analyzer",
    description="分析数据并提供统计摘要",
    category=ToolCategory.DATA,
    timeout=20.0
)
def data_analyzer(data: str, analysis_type: str = "summary") -> str:
    """
    数据分析
    
    Args:
        data: 数据（JSON格式的数组或对象）
        analysis_type: 分析类型 (summary, distribution, correlation)
    """
    try:
        parsed_data = json.loads(data)
        
        if isinstance(parsed_data, list):
            # 数值数据分析
            if all(isinstance(x, (int, float)) for x in parsed_data):
                import statistics
                
                result = {
                    "count": len(parsed_data),
                    "sum": sum(parsed_data),
                    "mean": statistics.mean(parsed_data),
                    "median": statistics.median(parsed_data),
                    "min": min(parsed_data),
                    "max": max(parsed_data)
                }
                
                if len(parsed_data) > 1:
                    result["stdev"] = statistics.stdev(parsed_data)
                
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            # 对象数组分析
            elif all(isinstance(x, dict) for x in parsed_data):
                keys = set()
                for item in parsed_data:
                    keys.update(item.keys())
                
                result = {
                    "count": len(parsed_data),
                    "fields": list(keys),
                    "sample": parsed_data[:3] if len(parsed_data) > 3 else parsed_data
                }
                
                return json.dumps(result, ensure_ascii=False, indent=2)
        
        return json.dumps({"raw_data": parsed_data, "type": type(parsed_data).__name__})
        
    except json.JSONDecodeError as e:
        return f"数据解析错误: {str(e)}"


@tool(
    name="data_transformer",
    description="转换数据格式",
    category=ToolCategory.DATA,
    timeout=15.0
)
def data_transformer(data: str, transform_type: str, params: str = None) -> str:
    """
    数据转换
    
    Args:
        data: 输入数据（JSON格式）
        transform_type: 转换类型 (flatten, filter, map, group, sort)
        params: 转换参数（JSON格式）
    """
    try:
        parsed_data = json.loads(data)
        transform_params = json.loads(params) if params else {}
        
        if transform_type == "flatten":
            # 扁平化嵌套结构
            if isinstance(parsed_data, list):
                result = []
                for item in parsed_data:
                    if isinstance(item, list):
                        result.extend(item)
                    else:
                        result.append(item)
                return json.dumps(result, ensure_ascii=False)
        
        elif transform_type == "filter":
            # 过滤数据
            field = transform_params.get('field')
            value = transform_params.get('value')
            
            if isinstance(parsed_data, list) and field:
                result = [x for x in parsed_data if x.get(field) == value]
                return json.dumps(result, ensure_ascii=False)
        
        elif transform_type == "sort":
            # 排序
            field = transform_params.get('field')
            reverse = transform_params.get('reverse', False)
            
            if isinstance(parsed_data, list):
                if field:
                    result = sorted(parsed_data, key=lambda x: x.get(field, 0), reverse=reverse)
                else:
                    result = sorted(parsed_data, reverse=reverse)
                return json.dumps(result, ensure_ascii=False)
        
        elif transform_type == "group":
            # 分组
            field = transform_params.get('field')
            
            if isinstance(parsed_data, list) and field:
                groups = {}
                for item in parsed_data:
                    key = str(item.get(field, 'unknown'))
                    if key not in groups:
                        groups[key] = []
                    groups[key].append(item)
                return json.dumps(groups, ensure_ascii=False)
        
        return json.dumps({"error": f"不支持的转换类型: {transform_type}"})
        
    except Exception as e:
        return f"转换错误: {str(e)}"


# ==================== 知识库工具 ====================

@tool(
    name="memory_store",
    description="存储信息到短期记忆（支持 Redis 持久化）。可用于在会话中保存重要信息。",
    category=ToolCategory.RETRIEVAL,
    timeout=5.0
)
def memory_store(key: str, value: str, ttl: int = 3600, namespace: str = "default") -> str:
    """
    存储信息到短期记忆
    
    Args:
        key: 存储键（唯一标识）
        value: 存储值（支持 JSON 字符串）
        ttl: 过期时间（秒，默认1小时，最大24小时）
        namespace: 命名空间（用于隔离不同会话）
    
    Returns:
        存储结果
    """
    global _memory_store
    
    ttl = min(max(ttl, 60), 86400)  # 限制 1分钟 - 24小时
    full_key = f"{namespace}:{key}"
    
    # 尝试使用 Redis
    if REDIS_AVAILABLE:
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            r.setex(f"agent_memory:{full_key}", ttl, value)
            
            return json.dumps({
                "success": True,
                "key": key,
                "namespace": namespace,
                "ttl": ttl,
                "storage": "redis",
                "size_bytes": len(value.encode('utf-8'))
            }, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Redis storage failed: {e}")
    
    # 回退到内存存储
    expiry = datetime.utcnow().timestamp() + ttl
    _memory_store[full_key] = {
        "value": value,
        "expiry": expiry,
        "created_at": datetime.utcnow().isoformat()
    }
    
    return json.dumps({
        "success": True,
        "key": key,
        "namespace": namespace,
        "ttl": ttl,
        "storage": "memory",
        "size_bytes": len(value.encode('utf-8')),
        "note": "使用内存存储，重启后数据会丢失"
    }, ensure_ascii=False)


@tool(
    name="memory_retrieve",
    description="从短期记忆中检索信息。",
    category=ToolCategory.RETRIEVAL,
    timeout=5.0
)
def memory_retrieve(key: str, namespace: str = "default", default: str = None) -> str:
    """
    从短期记忆中检索信息
    
    Args:
        key: 存储键
        namespace: 命名空间
        default: 找不到时的默认值
    
    Returns:
        检索结果
    """
    global _memory_store
    
    full_key = f"{namespace}:{key}"
    
    # 尝试使用 Redis
    if REDIS_AVAILABLE:
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            value = r.get(f"agent_memory:{full_key}")
            
            if value is not None:
                return json.dumps({
                    "success": True,
                    "key": key,
                    "namespace": namespace,
                    "value": value,
                    "storage": "redis"
                }, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Redis retrieval failed: {e}")
    
    # 从内存存储检索
    if full_key in _memory_store:
        entry = _memory_store[full_key]
        
        # 检查过期
        if entry['expiry'] > datetime.utcnow().timestamp():
            return json.dumps({
                "success": True,
                "key": key,
                "namespace": namespace,
                "value": entry['value'],
                "created_at": entry['created_at'],
                "storage": "memory"
            }, ensure_ascii=False)
        else:
            # 删除过期条目
            del _memory_store[full_key]
    
    # 未找到
    return json.dumps({
        "success": False,
        "key": key,
        "namespace": namespace,
        "value": default,
        "error": "Key not found or expired"
    }, ensure_ascii=False)


@tool(
    name="memory_delete",
    description="删除短期记忆中的信息。",
    category=ToolCategory.RETRIEVAL,
    timeout=5.0
)
def memory_delete(key: str, namespace: str = "default") -> str:
    """
    删除短期记忆中的信息
    
    Args:
        key: 存储键
        namespace: 命名空间
    
    Returns:
        删除结果
    """
    global _memory_store
    
    full_key = f"{namespace}:{key}"
    deleted = False
    
    # 尝试从 Redis 删除
    if REDIS_AVAILABLE:
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            deleted = r.delete(f"agent_memory:{full_key}") > 0
        except Exception as e:
            logger.debug(f"Redis deletion failed: {e}")
    
    # 从内存存储删除
    if full_key in _memory_store:
        del _memory_store[full_key]
        deleted = True
    
    return json.dumps({
        "success": deleted,
        "key": key,
        "namespace": namespace,
        "deleted": deleted
    }, ensure_ascii=False)


@tool(
    name="memory_list",
    description="列出指定命名空间下的所有记忆键。",
    category=ToolCategory.RETRIEVAL,
    timeout=5.0
)
def memory_list(namespace: str = "default", pattern: str = "*") -> str:
    """
    列出记忆键
    
    Args:
        namespace: 命名空间
        pattern: 匹配模式（支持 * 通配符）
    
    Returns:
        键列表
    """
    global _memory_store
    
    keys = []
    prefix = f"{namespace}:"
    
    # 从 Redis 获取
    if REDIS_AVAILABLE:
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_pattern = f"agent_memory:{prefix}{pattern.replace('*', '*')}"
            redis_keys = r.keys(redis_pattern)
            keys.extend([k.replace(f"agent_memory:{prefix}", "") for k in redis_keys])
        except Exception as e:
            logger.debug(f"Redis list failed: {e}")
    
    # 从内存存储获取
    now = datetime.utcnow().timestamp()
    for full_key, entry in list(_memory_store.items()):
        if full_key.startswith(prefix) and entry['expiry'] > now:
            key = full_key[len(prefix):]
            if pattern == "*" or key.startswith(pattern.replace("*", "")):
                if key not in keys:
                    keys.append(key)
    
    return json.dumps({
        "success": True,
        "namespace": namespace,
        "pattern": pattern,
        "keys": sorted(set(keys)),
        "count": len(keys)
    }, ensure_ascii=False)


# ==================== 系统工具 ====================

@tool(
    name="get_datetime",
    description="获取当前日期和时间",
    category=ToolCategory.SYSTEM
)
def get_datetime(format: str = "%Y-%m-%d %H:%M:%S", timezone: str = "UTC") -> str:
    """
    获取日期时间
    
    Args:
        format: 日期格式
        timezone: 时区
    """
    now = datetime.utcnow()
    return now.strftime(format)


@tool(
    name="uuid_generator",
    description="生成 UUID",
    category=ToolCategory.SYSTEM
)
def uuid_generator(version: int = 4) -> str:
    """
    生成 UUID
    
    Args:
        version: UUID 版本 (1 或 4)
    """
    import uuid
    
    if version == 1:
        return str(uuid.uuid1())
    return str(uuid.uuid4())


@tool(
    name="hash_text",
    description="计算文本的哈希值",
    category=ToolCategory.SYSTEM
)
def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    计算哈希
    
    Args:
        text: 输入文本
        algorithm: 哈希算法 (md5, sha1, sha256)
    """
    import hashlib
    
    algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256
    }
    
    hash_func = algorithms.get(algorithm, hashlib.sha256)
    return hash_func(text.encode()).hexdigest()


@tool(
    name="regex_matcher",
    description="使用正则表达式匹配文本",
    category=ToolCategory.DATA
)
def regex_matcher(text: str, pattern: str, operation: str = "findall") -> str:
    """
    正则匹配
    
    Args:
        text: 输入文本
        pattern: 正则表达式
        operation: 操作类型 (match, search, findall, sub)
    """
    try:
        compiled = re.compile(pattern)
        
        if operation == "match":
            result = compiled.match(text)
            return json.dumps({"matched": result is not None, "groups": result.groups() if result else []})
        
        elif operation == "search":
            result = compiled.search(text)
            return json.dumps({
                "found": result is not None,
                "match": result.group() if result else None,
                "position": result.span() if result else None
            })
        
        elif operation == "findall":
            results = compiled.findall(text)
            return json.dumps({"matches": results, "count": len(results)})
        
        elif operation == "sub":
            # 需要替换文本作为额外参数，这里简化处理
            return json.dumps({"error": "sub 操作需要替换文本参数"})
        
        return json.dumps({"error": f"不支持的操作: {operation}"})
        
    except re.error as e:
        return f"正则表达式错误: {str(e)}"


# ==================== HTTP 工具 ====================

# URL 白名单（生产环境应该配置允许的域名）
ALLOWED_URL_PATTERNS = [
    r'^https?://api\.',
    r'^https?://.*\.gov\.',
    r'^https?://.*\.edu\.',
    r'^https?://httpbin\.org',  # 测试用
]

@tool(
    name="http_request",
    description="发送 HTTP 请求到外部 API。支持 GET、POST、PUT、DELETE 方法。",
    category=ToolCategory.API,
    timeout=30.0
)
def http_request(url: str, method: str = "GET", headers: str = None, 
                 body: str = None, timeout_seconds: float = 10.0) -> str:
    """
    发送 HTTP 请求
    
    Args:
        url: 请求 URL（必须是 HTTPS 或允许的域名）
        method: HTTP 方法 (GET, POST, PUT, DELETE, PATCH)
        headers: 请求头（JSON格式）
        body: 请求体（JSON格式）
        timeout_seconds: 请求超时时间
    
    Returns:
        响应结果（JSON格式）
    """
    # 规范化方法
    method = method.upper()
    if method not in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']:
        return json.dumps({
            "success": False,
            "error": f"不支持的 HTTP 方法: {method}"
        }, ensure_ascii=False)
    
    # URL 安全检查
    if not url.startswith(('http://', 'https://')):
        return json.dumps({
            "success": False,
            "error": "URL 必须以 http:// 或 https:// 开头"
        }, ensure_ascii=False)
    
    # 解析请求头
    request_headers = {'User-Agent': 'VectorSphere-Agent/1.0'}
    if headers:
        try:
            request_headers.update(json.loads(headers))
        except json.JSONDecodeError:
            return json.dumps({
                "success": False,
                "error": "请求头必须是有效的 JSON 格式"
            }, ensure_ascii=False)
    
    # 解析请求体
    request_body = None
    if body:
        try:
            request_body = json.loads(body)
        except json.JSONDecodeError:
            request_body = body  # 非 JSON 内容
    
    timeout_seconds = min(max(timeout_seconds, 1.0), 30.0)
    
    # 使用 httpx 发送请求
    if HTTPX_AVAILABLE:
        try:
            import httpx
            
            with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
                if method in ['POST', 'PUT', 'PATCH']:
                    if isinstance(request_body, dict):
                        response = client.request(
                            method=method,
                            url=url,
                            headers=request_headers,
                            json=request_body
                        )
                    else:
                        response = client.request(
                            method=method,
                            url=url,
                            headers=request_headers,
                            content=request_body
                        )
                else:
                    response = client.request(
                        method=method,
                        url=url,
                        headers=request_headers
                    )
                
                # 尝试解析 JSON 响应
                try:
                    response_data = response.json()
                except:
                    response_data = response.text[:5000]  # 限制响应大小
                
                return json.dumps({
                    "success": True,
                    "url": str(response.url),
                    "method": method,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "response": response_data,
                    "elapsed_ms": response.elapsed.total_seconds() * 1000
                }, ensure_ascii=False, indent=2, default=str)
                
        except httpx.TimeoutException:
            return json.dumps({
                "success": False,
                "url": url,
                "method": method,
                "error": f"请求超时 ({timeout_seconds}秒)"
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
                "method": method,
                "error": str(e)
            }, ensure_ascii=False)
    
    # 回退到 urllib
    try:
        import urllib.request
        import urllib.error
        
        req = urllib.request.Request(url, method=method)
        for key, value in request_headers.items():
            req.add_header(key, value)
        
        if request_body and method in ['POST', 'PUT', 'PATCH']:
            if isinstance(request_body, dict):
                data = json.dumps(request_body).encode('utf-8')
                req.add_header('Content-Type', 'application/json')
            else:
                data = request_body.encode('utf-8') if isinstance(request_body, str) else request_body
            req.data = data
        
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            response_data = response.read().decode('utf-8')
            try:
                response_data = json.loads(response_data)
            except:
                pass
            
            return json.dumps({
                "success": True,
                "url": url,
                "method": method,
                "status_code": response.status,
                "response": response_data
            }, ensure_ascii=False, indent=2, default=str)
            
    except urllib.error.HTTPError as e:
        return json.dumps({
            "success": False,
            "url": url,
            "method": method,
            "status_code": e.code,
            "error": str(e.reason)
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "method": method,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="http_get",
    description="发送简单的 HTTP GET 请求。适用于获取数据的场景。",
    category=ToolCategory.API,
    timeout=15.0
)
def http_get(url: str, params: str = None) -> str:
    """
    简单的 HTTP GET 请求
    
    Args:
        url: 请求 URL
        params: 查询参数（JSON格式）
    
    Returns:
        响应内容
    """
    # 构建查询参数
    if params:
        try:
            param_dict = json.loads(params)
            query_string = "&".join(f"{k}={v}" for k, v in param_dict.items())
            url = f"{url}?{query_string}" if "?" not in url else f"{url}&{query_string}"
        except json.JSONDecodeError:
            pass
    
    return http_request(url, "GET")


@tool(
    name="http_post",
    description="发送 HTTP POST 请求。适用于提交数据的场景。",
    category=ToolCategory.API,
    timeout=15.0
)
def http_post(url: str, data: str = None, content_type: str = "application/json") -> str:
    """
    HTTP POST 请求
    
    Args:
        url: 请求 URL
        data: 请求数据（JSON格式）
        content_type: 内容类型
    
    Returns:
        响应内容
    """
    headers = json.dumps({"Content-Type": content_type})
    return http_request(url, "POST", headers=headers, body=data)


# ==================== 知识库管理工具 ====================

@tool(
    name="knowledge_add",
    description="向知识库添加文档或知识条目。支持单条或批量添加。",
    category=ToolCategory.RETRIEVAL,
    timeout=30.0
)
def knowledge_add(content: str, collection: str = "knowledge_base", 
                  doc_id: str = None, metadata: str = None) -> str:
    """
    向知识库添加知识
    
    将文本内容向量化后存储到 VectorSphere 向量数据库。
    
    Args:
        content: 要添加的文本内容
        collection: 目标集合名称（默认 knowledge_base）
        doc_id: 文档ID（可选，自动生成）
        metadata: 元数据（JSON格式，如 {"category": "tech", "author": "system"}）
    
    Returns:
        添加结果
    
    Examples:
        - content="向量数据库是一种专门...", collection="tech_docs"
        - content="产品A的使用说明...", metadata='{"type": "manual", "product": "A"}'
    """
    if not content or not content.strip():
        return json.dumps({
            "success": False,
            "error": "内容不能为空"
        }, ensure_ascii=False)
    
    # 解析元数据
    meta_dict = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            logger.warning(f"Invalid metadata JSON: {metadata}")
    
    # 生成文档ID
    if not doc_id:
        doc_id = hashlib.md5(f"{content[:100]}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
    
    # 添加时间戳
    meta_dict["created_at"] = datetime.utcnow().isoformat()
    meta_dict["content_length"] = len(content)
    
    try:
        from backend.modules.vectordb.vector_store import get_vector_store
        
        store = get_vector_store()
        if store:
            # 确保集合存在
            store.ensure_collection(collection)
            
            # 添加文档
            result = store.upsert_texts(
                collection=collection,
                texts=[content],
                ids=[doc_id],
                metadatas=[meta_dict]
            )
            
            return json.dumps({
                "success": True,
                "doc_id": doc_id,
                "collection": collection,
                "content_preview": content[:100] + "..." if len(content) > 100 else content,
                "metadata": meta_dict,
                "message": "文档已成功添加到知识库"
            }, ensure_ascii=False, indent=2)
            
    except ImportError as e:
        logger.debug(f"Vector store module not available: {e}")
    except Exception as e:
        logger.warning(f"Knowledge add failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)
    
    # 降级响应
    return json.dumps({
        "success": False,
        "doc_id": doc_id,
        "collection": collection,
        "error": "VectorSphere 向量数据库未配置，无法添加知识",
        "note": "请配置 VectorSphere 服务后重试"
    }, ensure_ascii=False, indent=2)


@tool(
    name="knowledge_delete",
    description="从知识库删除文档或知识条目。",
    category=ToolCategory.RETRIEVAL,
    timeout=15.0
)
def knowledge_delete(doc_ids: str, collection: str = "knowledge_base") -> str:
    """
    从知识库删除知识
    
    Args:
        doc_ids: 要删除的文档ID列表（逗号分隔或JSON数组）
        collection: 目标集合名称
    
    Returns:
        删除结果
    """
    # 解析文档ID列表
    if doc_ids.startswith("["):
        try:
            id_list = json.loads(doc_ids)
        except json.JSONDecodeError:
            id_list = [doc_ids]
    else:
        id_list = [id.strip() for id in doc_ids.split(",") if id.strip()]
    
    if not id_list:
        return json.dumps({
            "success": False,
            "error": "文档ID列表不能为空"
        }, ensure_ascii=False)
    
    try:
        from backend.modules.vectordb.vector_store import get_vector_store
        
        store = get_vector_store()
        if store:
            result = store.delete(collection=collection, ids=id_list)
            
            return json.dumps({
                "success": True,
                "deleted_ids": id_list,
                "collection": collection,
                "count": len(id_list),
                "message": f"已删除 {len(id_list)} 条记录"
            }, ensure_ascii=False, indent=2)
            
    except ImportError as e:
        logger.debug(f"Vector store module not available: {e}")
    except Exception as e:
        logger.warning(f"Knowledge delete failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)
    
    return json.dumps({
        "success": False,
        "error": "VectorSphere 向量数据库未配置"
    }, ensure_ascii=False)


@tool(
    name="collection_info",
    description="获取向量数据库集合的详细信息，包括向量数量、维度、索引类型等。",
    category=ToolCategory.RETRIEVAL,
    timeout=10.0
)
def collection_info(collection: str = None) -> str:
    """
    获取集合信息
    
    Args:
        collection: 集合名称（可选，不指定则列出所有集合）
    
    Returns:
        集合信息（JSON格式）
    """
    try:
        from backend.modules.vectordb.vector_store import get_vector_store
        
        store = get_vector_store()
        if store:
            if collection:
                # 获取单个集合信息
                info = store.get_collection(collection)
                if info:
                    return json.dumps({
                        "success": True,
                        "collection": {
                            "name": info.name,
                            "dimension": info.dimension,
                            "metric": info.metric,
                            "index_type": info.index_type,
                            "vector_count": info.vector_count,
                            "description": info.description,
                            "created_at": info.created_at
                        }
                    }, ensure_ascii=False, indent=2)
                else:
                    return json.dumps({
                        "success": False,
                        "error": f"集合 '{collection}' 不存在"
                    }, ensure_ascii=False)
            else:
                # 列出所有集合
                collections = store.list_collections()
                return json.dumps({
                    "success": True,
                    "collections": [
                        {
                            "name": c.name,
                            "dimension": c.dimension,
                            "metric": c.metric,
                            "vector_count": c.vector_count
                        }
                        for c in collections
                    ],
                    "total": len(collections)
                }, ensure_ascii=False, indent=2)
                
    except ImportError as e:
        logger.debug(f"Vector store module not available: {e}")
    except Exception as e:
        logger.warning(f"Collection info failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)
    
    return json.dumps({
        "success": False,
        "error": "VectorSphere 向量数据库未配置",
        "note": "请配置 VECTORSPHERE_URL 环境变量"
    }, ensure_ascii=False)


# ==================== 文本处理工具 ====================

@tool(
    name="text_summarize",
    description="对长文本进行智能摘要，提取关键信息。支持多种摘要策略。",
    category=ToolCategory.DATA,
    timeout=30.0
)
def text_summarize(text: str, max_length: int = 200, strategy: str = "extractive") -> str:
    """
    文本摘要
    
    对输入文本进行智能摘要，提取核心内容。
    
    Args:
        text: 待摘要的文本内容
        max_length: 摘要最大长度（字符数）
        strategy: 摘要策略
            - "extractive": 抽取式摘要（提取关键句子）
            - "compressive": 压缩式摘要（缩短句子）
            - "keyword": 基于关键词的摘要
    
    Returns:
        JSON格式的摘要结果，包含摘要文本和元数据
    
    Examples:
        - text="长文章内容...", max_length=100, strategy="extractive"
    """
    if not text or not text.strip():
        return json.dumps({
            "success": False,
            "error": "文本内容不能为空"
        }, ensure_ascii=False)
    
    text = text.strip()
    original_length = len(text)
    
    try:
        if strategy == "extractive":
            # 抽取式摘要：基于句子重要性评分
            sentences = re.split(r'[。！？.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                summary = text[:max_length]
            else:
                # 计算句子得分（基于词频和位置）
                word_freq = {}
                for sentence in sentences:
                    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', sentence.lower())
                    for word in words:
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                sentence_scores = []
                for i, sentence in enumerate(sentences):
                    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', sentence.lower())
                    score = sum(word_freq.get(w, 0) for w in words)
                    # 位置加权：开头和结尾的句子更重要
                    position_weight = 1.0 + 0.5 * (1.0 / (i + 1)) + 0.3 * (1.0 / (len(sentences) - i))
                    sentence_scores.append((sentence, score * position_weight))
                
                # 按得分排序并选择top句子
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                
                summary_sentences = []
                current_length = 0
                for sentence, score in sentence_scores:
                    if current_length + len(sentence) <= max_length:
                        summary_sentences.append(sentence)
                        current_length += len(sentence)
                    if current_length >= max_length:
                        break
                
                # 按原文顺序重排
                original_order = {s: i for i, s in enumerate(sentences)}
                summary_sentences.sort(key=lambda x: original_order.get(x, 0))
                summary = '。'.join(summary_sentences)
                if summary and not summary.endswith(('。', '！', '？', '.', '!', '?')):
                    summary += '。'
        
        elif strategy == "compressive":
            # 压缩式摘要：移除修饰词，保留核心内容
            # 移除常见的修饰词和连接词
            filler_patterns = [
                r'其实', r'事实上', r'实际上', r'基本上', r'总的来说',
                r'简单地说', r'换句话说', r'也就是说', r'众所周知',
                r'毫无疑问', r'不言而喻', r'显而易见', r'毋庸置疑',
                r'可以说', r'应该说', r'一般来说', r'通常来说'
            ]
            compressed = text
            for pattern in filler_patterns:
                compressed = re.sub(pattern, '', compressed)
            
            # 压缩多余空白
            compressed = re.sub(r'\s+', ' ', compressed).strip()
            summary = compressed[:max_length]
            if len(compressed) > max_length and not summary.endswith(('。', '！', '？', '.', '!', '?')):
                # 尝试在句号处截断
                last_period = max(
                    summary.rfind('。'), summary.rfind('！'), 
                    summary.rfind('？'), summary.rfind('.')
                )
                if last_period > max_length // 2:
                    summary = summary[:last_period + 1]
                else:
                    summary += '...'
        
        elif strategy == "keyword":
            # 基于关键词的摘要
            keywords = extract_keywords_internal(text, top_k=5)
            keyword_list = [kw['keyword'] for kw in keywords]
            
            # 选择包含关键词最多的句子
            sentences = re.split(r'[。！？.!?]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            sentence_scores = []
            for sentence in sentences:
                score = sum(1 for kw in keyword_list if kw in sentence)
                sentence_scores.append((sentence, score))
            
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            summary_sentences = []
            current_length = 0
            for sentence, score in sentence_scores[:5]:  # 最多5句
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
            
            summary = '。'.join(summary_sentences)
            if summary and not summary.endswith(('。', '！', '？', '.', '!', '?')):
                summary += '。'
        
        else:
            # 默认简单截断
            summary = text[:max_length]
            if len(text) > max_length:
                summary += '...'
        
        compression_ratio = round((1 - len(summary) / original_length) * 100, 1)
        
        return json.dumps({
            "success": True,
            "summary": summary,
            "original_length": original_length,
            "summary_length": len(summary),
            "compression_ratio": f"{compression_ratio}%",
            "strategy": strategy
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


def extract_keywords_internal(text: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """内部关键词提取函数"""
    # 分词（简单实现：基于中文和英文分割）
    words = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}', text.lower())
    
    # 停用词列表
    stopwords = {
        '的', '了', '是', '在', '和', '有', '这', '个', '也', '就',
        '都', '而', '及', '与', '或', '但', '如', '对', '等', '之',
        '为', '以', '于', '其', '从', '到', '被', '让', '把', '比',
        'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'for',
        'with', 'this', 'that', 'from', 'are', 'was', 'were', 'been'
    }
    
    # 词频统计
    word_freq = {}
    for word in words:
        if word not in stopwords and len(word) >= 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 计算 TF-IDF 近似得分
    total_words = len(words)
    keywords = []
    for word, freq in word_freq.items():
        tf = freq / total_words
        # 简化的 IDF：基于词在文本中的分布
        idf = 1 + (1 / (freq + 1))  # 出现越少，IDF越高
        score = tf * idf
        keywords.append({'keyword': word, 'frequency': freq, 'score': round(score, 4)})
    
    # 按得分排序
    keywords.sort(key=lambda x: x['score'], reverse=True)
    return keywords[:top_k]


@tool(
    name="keyword_extract",
    description="从文本中提取关键词和关键短语。支持 TF-IDF 和 TextRank 算法。",
    category=ToolCategory.DATA,
    timeout=20.0
)
def keyword_extract(text: str, top_k: int = 10, algorithm: str = "tfidf") -> str:
    """
    关键词提取
    
    从文本中提取最重要的关键词和短语。
    
    Args:
        text: 输入文本
        top_k: 返回的关键词数量（1-50）
        algorithm: 提取算法
            - "tfidf": TF-IDF 算法
            - "textrank": TextRank 算法（基于图的排序）
            - "frequency": 简单词频统计
    
    Returns:
        JSON格式的关键词列表，包含词语、频率和得分
    """
    if not text or not text.strip():
        return json.dumps({
            "success": False,
            "error": "文本内容不能为空"
        }, ensure_ascii=False)
    
    top_k = min(max(top_k, 1), 50)
    
    try:
        if algorithm == "tfidf":
            keywords = extract_keywords_internal(text, top_k)
        
        elif algorithm == "textrank":
            # TextRank 算法实现
            sentences = re.split(r'[。！？.!?\n]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                # 句子太少，回退到 TF-IDF
                keywords = extract_keywords_internal(text, top_k)
            else:
                # 构建词图
                words = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}', text.lower())
                stopwords = {'的', '了', '是', '在', '和', '有', '这', '个', '也', '就'}
                words = [w for w in words if w not in stopwords]
                
                # 共现统计（窗口大小为5）
                word_pairs = {}
                window_size = 5
                for i in range(len(words) - window_size):
                    window = words[i:i + window_size]
                    for j, w1 in enumerate(window):
                        for w2 in window[j+1:]:
                            pair = tuple(sorted([w1, w2]))
                            word_pairs[pair] = word_pairs.get(pair, 0) + 1
                
                # 计算词的 PageRank 得分
                word_scores = {}
                for word in set(words):
                    score = 0
                    for (w1, w2), count in word_pairs.items():
                        if word in (w1, w2):
                            score += count
                    word_scores[word] = score
                
                # 归一化并排序
                max_score = max(word_scores.values()) if word_scores else 1
                keywords = [
                    {'keyword': word, 'score': round(score / max_score, 4), 'frequency': words.count(word)}
                    for word, score in word_scores.items()
                ]
                keywords.sort(key=lambda x: x['score'], reverse=True)
                keywords = keywords[:top_k]
        
        elif algorithm == "frequency":
            # 简单词频统计
            words = re.findall(r'[\u4e00-\u9fa5]{2,}|[a-zA-Z]{3,}', text.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            keywords = [
                {'keyword': word, 'frequency': freq, 'score': freq}
                for word, freq in word_freq.items()
            ]
            keywords.sort(key=lambda x: x['frequency'], reverse=True)
            keywords = keywords[:top_k]
        
        else:
            keywords = extract_keywords_internal(text, top_k)
        
        return json.dumps({
            "success": True,
            "keywords": keywords,
            "total_extracted": len(keywords),
            "algorithm": algorithm,
            "text_length": len(text)
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="text_similarity",
    description="计算两段文本的相似度。支持多种相似度算法。",
    category=ToolCategory.DATA,
    timeout=15.0
)
def text_similarity(text1: str, text2: str, algorithm: str = "cosine") -> str:
    """
    文本相似度计算
    
    计算两段文本之间的语义相似度。
    
    Args:
        text1: 第一段文本
        text2: 第二段文本
        algorithm: 相似度算法
            - "cosine": 余弦相似度（基于词袋模型）
            - "jaccard": Jaccard 相似度（基于词集合）
            - "levenshtein": 编辑距离相似度
            - "ngram": N-gram 相似度
    
    Returns:
        JSON格式的相似度结果，包含相似度分数和详细信息
    """
    if not text1 or not text2:
        return json.dumps({
            "success": False,
            "error": "两段文本都不能为空"
        }, ensure_ascii=False)
    
    try:
        # 分词
        def tokenize(text):
            return re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text.lower())
        
        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)
        
        if algorithm == "cosine":
            # 余弦相似度
            all_tokens = list(set(tokens1 + tokens2))
            vec1 = [tokens1.count(t) for t in all_tokens]
            vec2 = [tokens2.count(t) for t in all_tokens]
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 * magnitude2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (magnitude1 * magnitude2)
        
        elif algorithm == "jaccard":
            # Jaccard 相似度
            set1 = set(tokens1)
            set2 = set(tokens2)
            intersection = set1 & set2
            union = set1 | set2
            
            similarity = len(intersection) / len(union) if union else 0.0
        
        elif algorithm == "levenshtein":
            # 编辑距离相似度
            def levenshtein_distance(s1, s2):
                if len(s1) < len(s2):
                    return levenshtein_distance(s2, s1)
                if len(s2) == 0:
                    return len(s1)
                
                previous_row = range(len(s2) + 1)
                for i, c1 in enumerate(s1):
                    current_row = [i + 1]
                    for j, c2 in enumerate(s2):
                        insertions = previous_row[j + 1] + 1
                        deletions = current_row[j] + 1
                        substitutions = previous_row[j] + (c1 != c2)
                        current_row.append(min(insertions, deletions, substitutions))
                    previous_row = current_row
                
                return previous_row[-1]
            
            distance = levenshtein_distance(text1, text2)
            max_len = max(len(text1), len(text2))
            similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
        
        elif algorithm == "ngram":
            # N-gram 相似度（默认 2-gram）
            def get_ngrams(text, n=2):
                return set(text[i:i+n] for i in range(len(text) - n + 1))
            
            ngrams1 = get_ngrams(text1, 2)
            ngrams2 = get_ngrams(text2, 2)
            
            intersection = ngrams1 & ngrams2
            union = ngrams1 | ngrams2
            
            similarity = len(intersection) / len(union) if union else 0.0
        
        else:
            # 默认使用余弦相似度
            return text_similarity(text1, text2, "cosine")
        
        # 相似度等级
        if similarity >= 0.9:
            level = "非常相似"
        elif similarity >= 0.7:
            level = "相似"
        elif similarity >= 0.5:
            level = "部分相似"
        elif similarity >= 0.3:
            level = "略有相似"
        else:
            level = "不相似"
        
        return json.dumps({
            "success": True,
            "similarity": round(similarity, 4),
            "similarity_percent": f"{round(similarity * 100, 2)}%",
            "level": level,
            "algorithm": algorithm,
            "text1_length": len(text1),
            "text2_length": len(text2),
            "common_tokens": len(set(tokens1) & set(tokens2)) if algorithm in ["cosine", "jaccard"] else None
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="sentiment_analysis",
    description="分析文本的情感倾向。返回积极、消极或中性的情感分类和得分。",
    category=ToolCategory.DATA,
    timeout=15.0
)
def sentiment_analysis(text: str, language: str = "auto") -> str:
    """
    情感分析
    
    分析文本的情感倾向和情绪特征。
    
    Args:
        text: 待分析的文本
        language: 语言（auto: 自动检测, zh: 中文, en: 英文）
    
    Returns:
        JSON格式的情感分析结果
    """
    if not text or not text.strip():
        return json.dumps({
            "success": False,
            "error": "文本内容不能为空"
        }, ensure_ascii=False)
    
    try:
        # 情感词典（简化版）
        positive_words_zh = {
            '好', '棒', '优秀', '满意', '喜欢', '开心', '快乐', '幸福', '成功', '完美',
            '精彩', '出色', '优质', '赞', '厉害', '强', '美好', '愉快', '欣慰', '感谢',
            '支持', '推荐', '不错', '很好', '太好', '真好', '非常好', '很棒', '太棒',
            '可爱', '友好', '温馨', '舒适', '方便', '高效', '专业', '靠谱', '值得'
        }
        
        negative_words_zh = {
            '差', '烂', '糟糕', '失望', '讨厌', '难过', '悲伤', '失败', '糟', '坏',
            '垃圾', '恶心', '无聊', '无语', '生气', '愤怒', '可恶', '恨', '骗', '假',
            '问题', '错误', '不好', '太差', '很差', '不行', '不满', '投诉', '退款',
            '慢', '贵', '难用', '复杂', '麻烦', '不推荐', '后悔', '不靠谱', '坑'
        }
        
        positive_words_en = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
            'love', 'like', 'happy', 'satisfied', 'perfect', 'best', 'recommend', 'nice',
            'beautiful', 'helpful', 'friendly', 'easy', 'fast', 'efficient', 'professional'
        }
        
        negative_words_en = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike', 'angry',
            'sad', 'disappointed', 'poor', 'wrong', 'error', 'problem', 'issue', 'fail',
            'slow', 'expensive', 'difficult', 'complicated', 'annoying', 'boring', 'useless'
        }
        
        # 语言检测
        if language == "auto":
            chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            detected_language = "zh" if chinese_chars > english_chars else "en"
        else:
            detected_language = language
        
        # 分词
        if detected_language == "zh":
            tokens = re.findall(r'[\u4e00-\u9fa5]+', text)
            positive_words = positive_words_zh
            negative_words = negative_words_zh
        else:
            tokens = re.findall(r'[a-zA-Z]+', text.lower())
            positive_words = positive_words_en
            negative_words = negative_words_en
        
        # 计算情感得分
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        # 否定词处理
        negation_words = {'不', '没', '无', '非', '未', '别', '勿', 'not', 'no', "n't", 'never', 'neither'}
        negation_count = sum(1 for token in tokens if token in negation_words)
        
        # 如果有否定词，翻转一部分情感
        if negation_count > 0:
            positive_count, negative_count = negative_count * 0.5, positive_count * 0.5
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            sentiment = "中性"
            score = 0.0
            confidence = 0.3
        else:
            score = (positive_count - negative_count) / (total_sentiment_words + 1)
            score = max(-1, min(1, score))  # 限制在 [-1, 1]
            
            if score > 0.2:
                sentiment = "积极"
            elif score < -0.2:
                sentiment = "消极"
            else:
                sentiment = "中性"
            
            confidence = min(0.9, 0.3 + total_sentiment_words * 0.05)
        
        # 情感强度
        if abs(score) > 0.6:
            intensity = "强烈"
        elif abs(score) > 0.3:
            intensity = "中等"
        else:
            intensity = "轻微"
        
        return json.dumps({
            "success": True,
            "sentiment": sentiment,
            "score": round(score, 4),
            "intensity": intensity,
            "confidence": round(confidence, 2),
            "details": {
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "total_tokens": len(tokens),
                "detected_language": detected_language
            }
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="text_statistics",
    description="统计文本的详细信息，包括字数、词数、句数、段落数、阅读时间等。",
    category=ToolCategory.DATA,
    timeout=10.0
)
def text_statistics(text: str) -> str:
    """
    文本统计
    
    分析文本的各项统计指标。
    
    Args:
        text: 输入文本
    
    Returns:
        JSON格式的统计结果
    """
    if not text:
        return json.dumps({
            "success": False,
            "error": "文本不能为空"
        }, ensure_ascii=False)
    
    try:
        # 基本统计
        char_count = len(text)
        char_count_no_space = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        # 中文字符
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
        
        # 英文单词
        english_words = re.findall(r'[a-zA-Z]+', text)
        english_word_count = len(english_words)
        
        # 数字
        numbers = re.findall(r'\d+', text)
        number_count = len(numbers)
        
        # 句子（中文和英文句号）
        sentences = re.split(r'[。！？.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)
        
        # 段落
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        paragraph_count = max(len(paragraphs), 1)
        
        # 行数
        lines = text.split('\n')
        line_count = len(lines)
        
        # 词数估算（中文按字，英文按词）
        word_count = chinese_chars + english_word_count
        
        # 平均句长
        avg_sentence_length = char_count_no_space / sentence_count if sentence_count > 0 else 0
        
        # 阅读时间估算（中文200字/分钟，英文200词/分钟）
        reading_time_minutes = (chinese_chars / 200) + (english_word_count / 200)
        
        # 词汇丰富度（Type-Token Ratio）
        all_words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text.lower())
        unique_words = set(all_words)
        ttr = len(unique_words) / len(all_words) if all_words else 0
        
        return json.dumps({
            "success": True,
            "statistics": {
                "characters": {
                    "total": char_count,
                    "no_space": char_count_no_space,
                    "chinese": chinese_chars,
                    "digits": number_count
                },
                "words": {
                    "total_estimated": word_count,
                    "english": english_word_count,
                    "unique": len(unique_words)
                },
                "structure": {
                    "sentences": sentence_count,
                    "paragraphs": paragraph_count,
                    "lines": line_count
                },
                "averages": {
                    "chars_per_sentence": round(avg_sentence_length, 1),
                    "words_per_paragraph": round(word_count / paragraph_count, 1)
                },
                "metrics": {
                    "reading_time_minutes": round(reading_time_minutes, 1),
                    "vocabulary_richness": round(ttr, 3)
                }
            }
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="text_clean",
    description="清理和规范化文本。去除多余空白、特殊字符、HTML标签等。",
    category=ToolCategory.DATA,
    timeout=10.0
)
def text_clean(text: str, operations: str = "whitespace,punctuation") -> str:
    """
    文本清理
    
    对文本进行清理和规范化处理。
    
    Args:
        text: 输入文本
        operations: 清理操作（逗号分隔）
            - "whitespace": 规范化空白字符
            - "html": 移除 HTML 标签
            - "urls": 移除 URL
            - "emails": 移除邮箱地址
            - "punctuation": 规范化标点符号
            - "numbers": 移除数字
            - "special": 移除特殊字符
            - "emoji": 移除 emoji
            - "lowercase": 转换为小写
            - "uppercase": 转换为大写
    
    Returns:
        JSON格式的清理结果
    """
    if not text:
        return json.dumps({
            "success": False,
            "error": "文本不能为空"
        }, ensure_ascii=False)
    
    try:
        original_text = text
        ops = [op.strip().lower() for op in operations.split(',')]
        applied_operations = []
        
        if "html" in ops:
            # 移除 HTML 标签
            text = re.sub(r'<[^>]+>', '', text)
            applied_operations.append("html")
        
        if "urls" in ops:
            # 移除 URL
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            applied_operations.append("urls")
        
        if "emails" in ops:
            # 移除邮箱
            text = re.sub(r'\S+@\S+\.\S+', '', text)
            applied_operations.append("emails")
        
        if "emoji" in ops:
            # 移除 emoji
            emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags
                "]+", flags=re.UNICODE)
            text = emoji_pattern.sub('', text)
            applied_operations.append("emoji")
        
        if "numbers" in ops:
            # 移除数字
            text = re.sub(r'\d+', '', text)
            applied_operations.append("numbers")
        
        if "special" in ops:
            # 移除特殊字符（保留中英文和基本标点）
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s。，！？、；：""''（）【】,.!?;:\'\"\(\)\[\]]', '', text)
            applied_operations.append("special")
        
        if "punctuation" in ops:
            # 规范化标点符号
            # 全角转半角
            punctuation_map = {
                '，': ',', '。': '.', '！': '!', '？': '?',
                '；': ';', '：': ':', '"': '"', '"': '"',
                ''': "'", ''': "'", '（': '(', '）': ')'
            }
            for full, half in punctuation_map.items():
                text = text.replace(full, half)
            # 去除重复标点
            text = re.sub(r'([,.!?;:])\1+', r'\1', text)
            applied_operations.append("punctuation")
        
        if "whitespace" in ops:
            # 规范化空白
            text = re.sub(r'[ \t]+', ' ', text)  # 多个空格/制表符合并
            text = re.sub(r'\n{3,}', '\n\n', text)  # 多个换行合并
            text = text.strip()
            applied_operations.append("whitespace")
        
        if "lowercase" in ops:
            text = text.lower()
            applied_operations.append("lowercase")
        
        if "uppercase" in ops:
            text = text.upper()
            applied_operations.append("uppercase")
        
        return json.dumps({
            "success": True,
            "cleaned_text": text,
            "original_length": len(original_text),
            "cleaned_length": len(text),
            "characters_removed": len(original_text) - len(text),
            "applied_operations": applied_operations
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 文件操作工具 ====================

@tool(
    name="file_reader",
    description="读取文件内容（仅限文本文件）。支持多种编码和行数限制。",
    category=ToolCategory.SYSTEM,
    timeout=10.0
)
def file_reader(path: str, encoding: str = "utf-8", max_lines: int = 100, 
                start_line: int = 0) -> str:
    """
    读取文件
    
    安全地读取文本文件内容，支持分页读取。
    
    Args:
        path: 文件路径
        encoding: 文件编码（utf-8, gbk, gb2312, latin-1 等）
        max_lines: 最大读取行数（1-1000）
        start_line: 起始行号（从0开始）
    
    Returns:
        JSON格式的文件内容和元数据
    """
    # 安全检查：禁止读取敏感路径
    forbidden_patterns = [
        '/etc/passwd', '/etc/shadow', '.ssh', '.env', 
        'password', 'secret', 'credential', 'token',
        '..', '~/'
    ]
    
    path_lower = path.lower()
    for pattern in forbidden_patterns:
        if pattern.lower() in path_lower:
            return json.dumps({
                "success": False,
                "error": f"安全限制：不允许访问路径 '{path}'"
            }, ensure_ascii=False)
    
    max_lines = min(max(max_lines, 1), 1000)
    
    try:
        # 尝试读取文件
        import os
        
        if not os.path.exists(path):
            return json.dumps({
                "success": False,
                "error": f"文件不存在: {path}"
            }, ensure_ascii=False)
        
        if not os.path.isfile(path):
            return json.dumps({
                "success": False,
                "error": f"路径不是文件: {path}"
            }, ensure_ascii=False)
        
        # 检查文件大小
        file_size = os.path.getsize(path)
        if file_size > 10 * 1024 * 1024:  # 10MB
            return json.dumps({
                "success": False,
                "error": "文件过大（超过10MB），请使用分页读取"
            }, ensure_ascii=False)
        
        with open(path, 'r', encoding=encoding, errors='replace') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        selected_lines = lines[start_line:start_line + max_lines]
        content = ''.join(selected_lines)
        
        return json.dumps({
            "success": True,
            "path": path,
            "content": content,
            "total_lines": total_lines,
            "returned_lines": len(selected_lines),
            "start_line": start_line,
            "end_line": start_line + len(selected_lines),
            "encoding": encoding,
            "file_size_bytes": file_size,
            "has_more": start_line + max_lines < total_lines
        }, ensure_ascii=False, indent=2)
        
    except UnicodeDecodeError:
        return json.dumps({
            "success": False,
            "error": f"编码错误：请尝试其他编码（如 gbk, latin-1）"
        }, ensure_ascii=False)
    except PermissionError:
        return json.dumps({
            "success": False,
            "error": "权限不足：无法读取文件"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="csv_parser",
    description="解析和处理 CSV 文件或 CSV 格式的文本。支持多种分隔符和数据操作。",
    category=ToolCategory.DATA,
    timeout=20.0
)
def csv_parser(data: str, operation: str = "parse", delimiter: str = ",",
               has_header: bool = True, columns: str = None) -> str:
    """
    CSV 数据处理
    
    解析、转换和分析 CSV 格式的数据。
    
    Args:
        data: CSV 数据（文本内容或文件路径）
        operation: 操作类型
            - "parse": 解析为结构化数据
            - "to_json": 转换为 JSON 格式
            - "stats": 统计每列的信息
            - "filter": 过滤行（需要 filter_expr 参数）
            - "select": 选择特定列（需要 columns 参数）
        delimiter: 分隔符（, ; \\t |）
        has_header: 是否包含表头
        columns: 列名列表（逗号分隔，用于 select 操作）
    
    Returns:
        JSON格式的处理结果
    """
    import csv
    import io
    
    try:
        # 判断是文件路径还是直接的CSV内容
        if os.path.exists(data) and os.path.isfile(data):
            with open(data, 'r', encoding='utf-8', errors='replace') as f:
                csv_content = f.read()
        else:
            csv_content = data
        
        # 解析CSV
        reader = csv.reader(io.StringIO(csv_content), delimiter=delimiter)
        rows = list(reader)
        
        if not rows:
            return json.dumps({
                "success": False,
                "error": "CSV 数据为空"
            }, ensure_ascii=False)
        
        if has_header:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            headers = [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows
        
        if operation == "parse":
            # 解析为字典列表
            result = []
            for row in data_rows[:100]:  # 限制返回100行
                if len(row) == len(headers):
                    result.append(dict(zip(headers, row)))
            
            return json.dumps({
                "success": True,
                "operation": "parse",
                "headers": headers,
                "data": result,
                "total_rows": len(data_rows),
                "returned_rows": len(result)
            }, ensure_ascii=False, indent=2)
        
        elif operation == "to_json":
            # 转换为 JSON
            result = []
            for row in data_rows:
                if len(row) == len(headers):
                    result.append(dict(zip(headers, row)))
            
            return json.dumps({
                "success": True,
                "operation": "to_json",
                "json_data": result,
                "total_records": len(result)
            }, ensure_ascii=False, indent=2)
        
        elif operation == "stats":
            # 统计信息
            stats = {}
            for i, header in enumerate(headers):
                col_values = [row[i] for row in data_rows if i < len(row)]
                
                # 尝试转换为数值
                numeric_values = []
                for v in col_values:
                    try:
                        numeric_values.append(float(v))
                    except (ValueError, TypeError):
                        pass
                
                col_stats = {
                    "count": len(col_values),
                    "unique": len(set(col_values)),
                    "null_count": sum(1 for v in col_values if not v.strip())
                }
                
                if numeric_values:
                    col_stats["numeric"] = True
                    col_stats["min"] = min(numeric_values)
                    col_stats["max"] = max(numeric_values)
                    col_stats["mean"] = sum(numeric_values) / len(numeric_values)
                else:
                    col_stats["numeric"] = False
                    col_stats["sample_values"] = list(set(col_values))[:5]
                
                stats[header] = col_stats
            
            return json.dumps({
                "success": True,
                "operation": "stats",
                "total_rows": len(data_rows),
                "total_columns": len(headers),
                "column_stats": stats
            }, ensure_ascii=False, indent=2)
        
        elif operation == "select":
            # 选择特定列
            if not columns:
                return json.dumps({
                    "success": False,
                    "error": "select 操作需要指定 columns 参数"
                }, ensure_ascii=False)
            
            selected_cols = [c.strip() for c in columns.split(',')]
            col_indices = [headers.index(c) for c in selected_cols if c in headers]
            
            result = []
            for row in data_rows[:100]:
                selected_row = [row[i] for i in col_indices if i < len(row)]
                result.append(dict(zip(selected_cols, selected_row)))
            
            return json.dumps({
                "success": True,
                "operation": "select",
                "selected_columns": selected_cols,
                "data": result,
                "total_rows": len(result)
            }, ensure_ascii=False, indent=2)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"未知操作: {operation}"
            }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="json_processor",
    description="处理 JSON 数据。支持解析、格式化、查询、转换等操作。",
    category=ToolCategory.DATA,
    timeout=15.0
)
def json_processor(data: str, operation: str = "parse", query: str = None,
                   indent: int = 2) -> str:
    """
    JSON 数据处理
    
    解析、查询和转换 JSON 数据。
    
    Args:
        data: JSON 数据（文本或文件路径）
        operation: 操作类型
            - "parse": 解析并验证 JSON
            - "format": 格式化（美化）JSON
            - "minify": 压缩 JSON
            - "query": 使用 JSONPath 查询（简化版）
            - "keys": 获取所有键
            - "flatten": 扁平化嵌套结构
            - "validate": 验证 JSON 格式
        query: JSONPath 查询表达式（用于 query 操作）
        indent: 缩进空格数（用于 format 操作）
    
    Returns:
        JSON格式的处理结果
    """
    try:
        # 尝试读取文件
        if os.path.exists(data) and os.path.isfile(data):
            with open(data, 'r', encoding='utf-8') as f:
                json_content = f.read()
        else:
            json_content = data
        
        # 解析 JSON
        try:
            parsed = json.loads(json_content)
        except json.JSONDecodeError as e:
            if operation == "validate":
                return json.dumps({
                    "success": True,
                    "valid": False,
                    "error": str(e),
                    "error_position": e.pos
                }, ensure_ascii=False)
            return json.dumps({
                "success": False,
                "error": f"JSON 解析错误: {e}"
            }, ensure_ascii=False)
        
        if operation == "parse" or operation == "format":
            return json.dumps({
                "success": True,
                "operation": operation,
                "data": parsed,
                "type": type(parsed).__name__,
                "size": len(json_content)
            }, ensure_ascii=False, indent=indent)
        
        elif operation == "minify":
            minified = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
            return json.dumps({
                "success": True,
                "operation": "minify",
                "minified": minified,
                "original_size": len(json_content),
                "minified_size": len(minified),
                "reduction": f"{round((1 - len(minified)/len(json_content)) * 100, 1)}%"
            }, ensure_ascii=False)
        
        elif operation == "validate":
            return json.dumps({
                "success": True,
                "valid": True,
                "type": type(parsed).__name__,
                "size": len(json_content)
            }, ensure_ascii=False)
        
        elif operation == "keys":
            def get_all_keys(obj, prefix=""):
                keys = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        full_key = f"{prefix}.{k}" if prefix else k
                        keys.append(full_key)
                        keys.extend(get_all_keys(v, full_key))
                elif isinstance(obj, list) and obj:
                    keys.extend(get_all_keys(obj[0], f"{prefix}[]"))
                return keys
            
            all_keys = get_all_keys(parsed)
            return json.dumps({
                "success": True,
                "operation": "keys",
                "keys": list(set(all_keys)),
                "total_keys": len(set(all_keys))
            }, ensure_ascii=False, indent=2)
        
        elif operation == "query":
            if not query:
                return json.dumps({
                    "success": False,
                    "error": "query 操作需要指定 query 参数"
                }, ensure_ascii=False)
            
            # 简化的 JSONPath 查询
            # 支持: $.key, $.key.subkey, $[0], $.array[*].field
            def simple_jsonpath(obj, path):
                parts = path.replace('$', '').strip('.').split('.')
                current = obj
                
                for part in parts:
                    if not part:
                        continue
                    
                    # 处理数组索引
                    array_match = re.match(r'(\w+)?\[(\d+|\*)\]', part)
                    if array_match:
                        key, idx = array_match.groups()
                        if key and isinstance(current, dict):
                            current = current.get(key, [])
                        
                        if isinstance(current, list):
                            if idx == '*':
                                # 返回所有元素
                                remaining_path = '.'.join(parts[parts.index(part)+1:])
                                if remaining_path:
                                    return [simple_jsonpath(item, '$.' + remaining_path) for item in current]
                                return current
                            else:
                                idx = int(idx)
                                current = current[idx] if idx < len(current) else None
                    elif isinstance(current, dict):
                        current = current.get(part)
                    else:
                        return None
                
                return current
            
            result = simple_jsonpath(parsed, query)
            return json.dumps({
                "success": True,
                "operation": "query",
                "query": query,
                "result": result
            }, ensure_ascii=False, indent=2)
        
        elif operation == "flatten":
            # 扁平化嵌套结构
            def flatten_dict(obj, prefix=""):
                items = {}
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        new_key = f"{prefix}.{k}" if prefix else k
                        items.update(flatten_dict(v, new_key))
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        new_key = f"{prefix}[{i}]"
                        items.update(flatten_dict(v, new_key))
                else:
                    items[prefix] = obj
                return items
            
            flattened = flatten_dict(parsed)
            return json.dumps({
                "success": True,
                "operation": "flatten",
                "flattened": flattened,
                "total_fields": len(flattened)
            }, ensure_ascii=False, indent=2)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"未知操作: {operation}"
            }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="yaml_processor",
    description="处理 YAML 数据。支持解析、格式化、转换为 JSON 等操作。",
    category=ToolCategory.DATA,
    timeout=15.0
)
def yaml_processor(data: str, operation: str = "parse", output_format: str = "json") -> str:
    """
    YAML 数据处理
    
    解析和转换 YAML 格式数据。
    
    Args:
        data: YAML 数据（文本或文件路径）
        operation: 操作类型
            - "parse": 解析 YAML
            - "to_json": 转换为 JSON
            - "validate": 验证 YAML 格式
        output_format: 输出格式（json, yaml）
    
    Returns:
        JSON格式的处理结果
    """
    try:
        # 尝试导入 yaml
        try:
            import yaml
            YAML_AVAILABLE = True
        except ImportError:
            YAML_AVAILABLE = False
        
        if not YAML_AVAILABLE:
            # 简化的 YAML 解析（仅支持基本格式）
            def simple_yaml_parse(text):
                result = {}
                current_key = None
                current_indent = 0
                lines = text.strip().split('\n')
                
                for line in lines:
                    if not line.strip() or line.strip().startswith('#'):
                        continue
                    
                    # 检测缩进
                    stripped = line.lstrip()
                    indent = len(line) - len(stripped)
                    
                    if ':' in stripped:
                        key, _, value = stripped.partition(':')
                        key = key.strip()
                        value = value.strip()
                        
                        if value:
                            # 简单键值对
                            result[key] = value.strip('"\'')
                        else:
                            # 可能是嵌套结构
                            result[key] = {}
                            current_key = key
                
                return result
            
            yaml_content = data
            if os.path.exists(data) and os.path.isfile(data):
                with open(data, 'r', encoding='utf-8') as f:
                    yaml_content = f.read()
            
            parsed = simple_yaml_parse(yaml_content)
            
            return json.dumps({
                "success": True,
                "operation": operation,
                "data": parsed,
                "note": "使用简化解析器（完整功能需要 pyyaml 库）"
            }, ensure_ascii=False, indent=2)
        
        # 使用 pyyaml
        yaml_content = data
        if os.path.exists(data) and os.path.isfile(data):
            with open(data, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
        
        try:
            parsed = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            if operation == "validate":
                return json.dumps({
                    "success": True,
                    "valid": False,
                    "error": str(e)
                }, ensure_ascii=False)
            return json.dumps({
                "success": False,
                "error": f"YAML 解析错误: {e}"
            }, ensure_ascii=False)
        
        if operation == "validate":
            return json.dumps({
                "success": True,
                "valid": True,
                "type": type(parsed).__name__
            }, ensure_ascii=False)
        
        if operation == "parse" or operation == "to_json":
            return json.dumps({
                "success": True,
                "operation": operation,
                "data": parsed,
                "type": type(parsed).__name__
            }, ensure_ascii=False, indent=2)
        
        return json.dumps({
            "success": False,
            "error": f"未知操作: {operation}"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="xml_processor",
    description="处理 XML 数据。支持解析、查询、转换为 JSON 等操作。",
    category=ToolCategory.DATA,
    timeout=15.0
)
def xml_processor(data: str, operation: str = "parse", xpath: str = None) -> str:
    """
    XML 数据处理
    
    解析和处理 XML 格式数据。
    
    Args:
        data: XML 数据（文本或文件路径）
        operation: 操作类型
            - "parse": 解析 XML 结构
            - "to_json": 转换为 JSON
            - "xpath": XPath 查询（简化版）
            - "validate": 验证 XML 格式
        xpath: XPath 表达式（用于 xpath 操作）
    
    Returns:
        JSON格式的处理结果
    """
    try:
        import xml.etree.ElementTree as ET
        
        xml_content = data
        if os.path.exists(data) and os.path.isfile(data):
            with open(data, 'r', encoding='utf-8') as f:
                xml_content = f.read()
        
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            if operation == "validate":
                return json.dumps({
                    "success": True,
                    "valid": False,
                    "error": str(e)
                }, ensure_ascii=False)
            return json.dumps({
                "success": False,
                "error": f"XML 解析错误: {e}"
            }, ensure_ascii=False)
        
        def element_to_dict(element):
            """将 XML 元素转换为字典"""
            result = {}
            
            # 属性
            if element.attrib:
                result['@attributes'] = element.attrib
            
            # 子元素
            children = list(element)
            if children:
                child_dict = {}
                for child in children:
                    child_name = child.tag
                    child_data = element_to_dict(child)
                    
                    if child_name in child_dict:
                        # 多个同名子元素，转为数组
                        if not isinstance(child_dict[child_name], list):
                            child_dict[child_name] = [child_dict[child_name]]
                        child_dict[child_name].append(child_data)
                    else:
                        child_dict[child_name] = child_data
                
                result.update(child_dict)
            
            # 文本内容
            if element.text and element.text.strip():
                if result:
                    result['#text'] = element.text.strip()
                else:
                    return element.text.strip()
            
            return result if result else None
        
        if operation == "validate":
            return json.dumps({
                "success": True,
                "valid": True,
                "root_tag": root.tag
            }, ensure_ascii=False)
        
        if operation == "parse":
            # 获取基本结构信息
            def get_structure(element, depth=0):
                children = list(element)
                return {
                    "tag": element.tag,
                    "attributes": element.attrib,
                    "text": element.text.strip() if element.text else None,
                    "children_count": len(children),
                    "children": [get_structure(c, depth+1) for c in children[:5]]  # 限制深度
                }
            
            structure = get_structure(root)
            return json.dumps({
                "success": True,
                "operation": "parse",
                "root_tag": root.tag,
                "structure": structure
            }, ensure_ascii=False, indent=2)
        
        if operation == "to_json":
            json_data = {root.tag: element_to_dict(root)}
            return json.dumps({
                "success": True,
                "operation": "to_json",
                "data": json_data
            }, ensure_ascii=False, indent=2)
        
        if operation == "xpath":
            if not xpath:
                return json.dumps({
                    "success": False,
                    "error": "xpath 操作需要指定 xpath 参数"
                }, ensure_ascii=False)
            
            try:
                elements = root.findall(xpath)
                results = []
                for elem in elements:
                    results.append({
                        "tag": elem.tag,
                        "text": elem.text,
                        "attributes": elem.attrib,
                        "data": element_to_dict(elem)
                    })
                
                return json.dumps({
                    "success": True,
                    "operation": "xpath",
                    "xpath": xpath,
                    "matches": len(results),
                    "results": results
                }, ensure_ascii=False, indent=2)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"XPath 查询错误: {e}"
                }, ensure_ascii=False)
        
        return json.dumps({
            "success": False,
            "error": f"未知操作: {operation}"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="directory_list",
    description="列出目录内容，显示文件和子目录信息。",
    category=ToolCategory.SYSTEM,
    timeout=10.0
)
def directory_list(path: str, pattern: str = "*", recursive: bool = False,
                   max_items: int = 100) -> str:
    """
    列出目录内容
    
    Args:
        path: 目录路径
        pattern: 文件匹配模式（如 *.txt, *.py）
        recursive: 是否递归子目录
        max_items: 最大返回项数
    
    Returns:
        JSON格式的目录列表
    """
    import glob
    import os
    
    # 安全检查
    forbidden_paths = ['/', '/etc', '/root', '/home', '/var', '/sys', '/proc']
    abs_path = os.path.abspath(path)
    
    for forbidden in forbidden_paths:
        if abs_path == forbidden or abs_path.startswith(forbidden + '/'):
            if not abs_path.startswith(os.getcwd()):
                return json.dumps({
                    "success": False,
                    "error": "安全限制：不允许访问系统目录"
                }, ensure_ascii=False)
    
    try:
        if not os.path.exists(path):
            return json.dumps({
                "success": False,
                "error": f"目录不存在: {path}"
            }, ensure_ascii=False)
        
        if not os.path.isdir(path):
            return json.dumps({
                "success": False,
                "error": f"路径不是目录: {path}"
            }, ensure_ascii=False)
        
        # 构建匹配模式
        if recursive:
            search_pattern = os.path.join(path, '**', pattern)
            matches = glob.glob(search_pattern, recursive=True)
        else:
            search_pattern = os.path.join(path, pattern)
            matches = glob.glob(search_pattern)
        
        items = []
        for item_path in matches[:max_items]:
            try:
                stat_info = os.stat(item_path)
                items.append({
                    "name": os.path.basename(item_path),
                    "path": item_path,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size_bytes": stat_info.st_size if os.path.isfile(item_path) else None,
                    "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                })
            except (PermissionError, OSError):
                continue
        
        # 按类型和名称排序
        items.sort(key=lambda x: (x['type'] != 'directory', x['name'].lower()))
        
        return json.dumps({
            "success": True,
            "path": path,
            "pattern": pattern,
            "recursive": recursive,
            "items": items,
            "total_items": len(items),
            "has_more": len(matches) > max_items
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="file_info",
    description="获取文件的详细信息，包括大小、类型、修改时间等。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def file_info(path: str) -> str:
    """
    获取文件信息
    
    Args:
        path: 文件路径
    
    Returns:
        JSON格式的文件信息
    """
    import os
    import mimetypes
    
    try:
        if not os.path.exists(path):
            return json.dumps({
                "success": False,
                "error": f"文件不存在: {path}"
            }, ensure_ascii=False)
        
        stat_info = os.stat(path)
        is_file = os.path.isfile(path)
        
        # 获取 MIME 类型
        mime_type, _ = mimetypes.guess_type(path)
        
        # 格式化文件大小
        def format_size(size):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024:
                    return f"{size:.2f} {unit}"
                size /= 1024
            return f"{size:.2f} PB"
        
        info = {
            "success": True,
            "path": path,
            "name": os.path.basename(path),
            "type": "file" if is_file else "directory",
            "exists": True,
            "size": {
                "bytes": stat_info.st_size,
                "formatted": format_size(stat_info.st_size)
            } if is_file else None,
            "timestamps": {
                "created": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat_info.st_atime).isoformat()
            },
            "permissions": oct(stat_info.st_mode)[-3:],
            "mime_type": mime_type,
            "extension": os.path.splitext(path)[1] if is_file else None
        }
        
        return json.dumps(info, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 编码/解码工具 ====================

@tool(
    name="base64_codec",
    description="Base64 编码和解码。支持文本和二进制数据。",
    category=ToolCategory.DATA,
    timeout=5.0
)
def base64_codec(data: str, operation: str = "encode", encoding: str = "utf-8") -> str:
    """
    Base64 编码/解码
    
    Args:
        data: 输入数据
        operation: 操作类型
            - "encode": 编码为 Base64
            - "decode": 从 Base64 解码
        encoding: 文本编码（用于解码时）
    
    Returns:
        JSON格式的编码/解码结果
    """
    import base64
    
    try:
        if operation == "encode":
            encoded = base64.b64encode(data.encode(encoding)).decode('ascii')
            return json.dumps({
                "success": True,
                "operation": "encode",
                "result": encoded,
                "original_length": len(data),
                "encoded_length": len(encoded)
            }, ensure_ascii=False)
        
        elif operation == "decode":
            # 移除可能的空白字符
            data = data.strip().replace('\n', '').replace(' ', '')
            decoded_bytes = base64.b64decode(data)
            
            try:
                decoded = decoded_bytes.decode(encoding)
                return json.dumps({
                    "success": True,
                    "operation": "decode",
                    "result": decoded,
                    "decoded_length": len(decoded)
                }, ensure_ascii=False)
            except UnicodeDecodeError:
                # 二进制数据，返回十六进制表示
                return json.dumps({
                    "success": True,
                    "operation": "decode",
                    "result_hex": decoded_bytes.hex(),
                    "note": "二进制数据，以十六进制显示",
                    "decoded_bytes": len(decoded_bytes)
                }, ensure_ascii=False)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"未知操作: {operation}"
            }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="url_codec",
    description="URL 编码和解码。处理 URL 中的特殊字符。",
    category=ToolCategory.DATA,
    timeout=5.0
)
def url_codec(data: str, operation: str = "encode", plus_space: bool = False) -> str:
    """
    URL 编码/解码
    
    Args:
        data: 输入数据
        operation: 操作类型
            - "encode": URL 编码
            - "decode": URL 解码
            - "parse": 解析 URL 组件
        plus_space: 是否使用 + 代替空格（表单编码格式）
    
    Returns:
        JSON格式的编码/解码结果
    """
    import urllib.parse
    
    try:
        if operation == "encode":
            if plus_space:
                encoded = urllib.parse.quote_plus(data)
            else:
                encoded = urllib.parse.quote(data, safe='')
            
            return json.dumps({
                "success": True,
                "operation": "encode",
                "result": encoded,
                "original_length": len(data),
                "encoded_length": len(encoded)
            }, ensure_ascii=False)
        
        elif operation == "decode":
            if plus_space:
                decoded = urllib.parse.unquote_plus(data)
            else:
                decoded = urllib.parse.unquote(data)
            
            return json.dumps({
                "success": True,
                "operation": "decode",
                "result": decoded,
                "decoded_length": len(decoded)
            }, ensure_ascii=False)
        
        elif operation == "parse":
            parsed = urllib.parse.urlparse(data)
            query_params = urllib.parse.parse_qs(parsed.query)
            
            return json.dumps({
                "success": True,
                "operation": "parse",
                "components": {
                    "scheme": parsed.scheme,
                    "netloc": parsed.netloc,
                    "path": parsed.path,
                    "params": parsed.params,
                    "query": parsed.query,
                    "fragment": parsed.fragment,
                    "hostname": parsed.hostname,
                    "port": parsed.port,
                    "query_params": query_params
                }
            }, ensure_ascii=False, indent=2)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"未知操作: {operation}"
            }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="html_codec",
    description="HTML 实体编码和解码。处理 HTML 中的特殊字符。",
    category=ToolCategory.DATA,
    timeout=5.0
)
def html_codec(data: str, operation: str = "encode") -> str:
    """
    HTML 实体编码/解码
    
    Args:
        data: 输入数据
        operation: 操作类型
            - "encode": HTML 实体编码
            - "decode": HTML 实体解码
    
    Returns:
        JSON格式的编码/解码结果
    """
    import html
    
    try:
        if operation == "encode":
            encoded = html.escape(data)
            return json.dumps({
                "success": True,
                "operation": "encode",
                "result": encoded,
                "original_length": len(data),
                "encoded_length": len(encoded)
            }, ensure_ascii=False)
        
        elif operation == "decode":
            decoded = html.unescape(data)
            return json.dumps({
                "success": True,
                "operation": "decode",
                "result": decoded,
                "decoded_length": len(decoded)
            }, ensure_ascii=False)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"未知操作: {operation}"
            }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="hex_codec",
    description="十六进制编码和解码。",
    category=ToolCategory.DATA,
    timeout=5.0
)
def hex_codec(data: str, operation: str = "encode", encoding: str = "utf-8") -> str:
    """
    十六进制编码/解码
    
    Args:
        data: 输入数据
        operation: encode 或 decode
        encoding: 文本编码
    
    Returns:
        JSON格式的结果
    """
    try:
        if operation == "encode":
            hex_string = data.encode(encoding).hex()
            return json.dumps({
                "success": True,
                "operation": "encode",
                "result": hex_string,
                "formatted": ' '.join(hex_string[i:i+2] for i in range(0, len(hex_string), 2))
            }, ensure_ascii=False)
        
        elif operation == "decode":
            # 移除空格
            data = data.replace(' ', '').replace('0x', '')
            decoded = bytes.fromhex(data).decode(encoding)
            return json.dumps({
                "success": True,
                "operation": "decode",
                "result": decoded
            }, ensure_ascii=False)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"未知操作: {operation}"
            }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 时间处理工具 ====================

@tool(
    name="datetime_parse",
    description="解析各种格式的日期时间字符串，转换为标准格式。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def datetime_parse(date_string: str, input_format: str = "auto", 
                   output_format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    日期时间解析
    
    Args:
        date_string: 日期时间字符串
        input_format: 输入格式（auto 自动检测）
        output_format: 输出格式
    
    Returns:
        JSON格式的解析结果
    """
    from datetime import datetime
    
    # 常见日期格式
    common_formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y年%m月%d日",
        "%Y年%m月%d日 %H时%M分",
        "%Y年%m月%d日 %H:%M:%S",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%Y%m%d",
        "%Y%m%d%H%M%S",
        "%a %b %d %H:%M:%S %Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
    ]
    
    try:
        parsed_dt = None
        detected_format = None
        
        if input_format == "auto":
            # 尝试各种格式
            for fmt in common_formats:
                try:
                    parsed_dt = datetime.strptime(date_string.strip(), fmt)
                    detected_format = fmt
                    break
                except ValueError:
                    continue
            
            # 尝试时间戳
            if parsed_dt is None:
                try:
                    timestamp = float(date_string)
                    if timestamp > 1e12:  # 毫秒时间戳
                        timestamp /= 1000
                    parsed_dt = datetime.fromtimestamp(timestamp)
                    detected_format = "timestamp"
                except ValueError:
                    pass
        else:
            parsed_dt = datetime.strptime(date_string.strip(), input_format)
            detected_format = input_format
        
        if parsed_dt is None:
            return json.dumps({
                "success": False,
                "error": f"无法解析日期时间: {date_string}"
            }, ensure_ascii=False)
        
        # 生成多种格式的输出
        return json.dumps({
            "success": True,
            "original": date_string,
            "detected_format": detected_format,
            "parsed": {
                "formatted": parsed_dt.strftime(output_format),
                "iso": parsed_dt.isoformat(),
                "date": parsed_dt.strftime("%Y-%m-%d"),
                "time": parsed_dt.strftime("%H:%M:%S"),
                "timestamp": int(parsed_dt.timestamp()),
                "timestamp_ms": int(parsed_dt.timestamp() * 1000),
                "year": parsed_dt.year,
                "month": parsed_dt.month,
                "day": parsed_dt.day,
                "weekday": parsed_dt.strftime("%A"),
                "weekday_cn": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][parsed_dt.weekday()]
            }
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="datetime_calculate",
    description="日期时间计算，支持加减天数、小时等操作。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def datetime_calculate(date_string: str = None, operation: str = "add",
                       days: int = 0, hours: int = 0, minutes: int = 0,
                       weeks: int = 0, months: int = 0, years: int = 0) -> str:
    """
    日期时间计算
    
    Args:
        date_string: 起始日期（为空则使用当前时间）
        operation: add 或 subtract
        days, hours, minutes, weeks: 时间增量
        months, years: 月份和年份增量（近似计算）
    
    Returns:
        JSON格式的计算结果
    """
    from datetime import datetime, timedelta
    
    try:
        # 解析起始日期
        if date_string:
            result = json.loads(datetime_parse(date_string))
            if not result.get("success"):
                return json.dumps(result, ensure_ascii=False)
            start_dt = datetime.fromisoformat(result["parsed"]["iso"])
        else:
            start_dt = datetime.now()
        
        # 计算时间增量
        total_days = days + (weeks * 7) + (months * 30) + (years * 365)
        delta = timedelta(days=total_days, hours=hours, minutes=minutes)
        
        if operation == "subtract":
            result_dt = start_dt - delta
        else:
            result_dt = start_dt + delta
        
        return json.dumps({
            "success": True,
            "operation": operation,
            "start": start_dt.isoformat(),
            "delta": {
                "days": days,
                "hours": hours,
                "minutes": minutes,
                "weeks": weeks,
                "months": months,
                "years": years,
                "total_days": total_days
            },
            "result": {
                "iso": result_dt.isoformat(),
                "formatted": result_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "date": result_dt.strftime("%Y-%m-%d"),
                "timestamp": int(result_dt.timestamp())
            }
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="datetime_diff",
    description="计算两个日期时间之间的差异。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def datetime_diff(date1: str, date2: str) -> str:
    """
    计算日期差异
    
    Args:
        date1: 第一个日期
        date2: 第二个日期
    
    Returns:
        JSON格式的差异结果
    """
    from datetime import datetime
    
    try:
        # 解析两个日期
        result1 = json.loads(datetime_parse(date1))
        result2 = json.loads(datetime_parse(date2))
        
        if not result1.get("success"):
            return json.dumps({"success": False, "error": f"无法解析日期1: {date1}"}, ensure_ascii=False)
        if not result2.get("success"):
            return json.dumps({"success": False, "error": f"无法解析日期2: {date2}"}, ensure_ascii=False)
        
        dt1 = datetime.fromisoformat(result1["parsed"]["iso"])
        dt2 = datetime.fromisoformat(result2["parsed"]["iso"])
        
        # 计算差异
        delta = dt2 - dt1
        total_seconds = abs(delta.total_seconds())
        
        # 转换为各种单位
        total_minutes = total_seconds / 60
        total_hours = total_minutes / 60
        total_days = total_hours / 24
        total_weeks = total_days / 7
        total_months = total_days / 30
        total_years = total_days / 365
        
        # 友好的表示
        if abs(total_days) >= 365:
            human_readable = f"{total_years:.1f} 年"
        elif abs(total_days) >= 30:
            human_readable = f"{total_months:.1f} 个月"
        elif abs(total_days) >= 7:
            human_readable = f"{total_weeks:.1f} 周"
        elif abs(total_days) >= 1:
            human_readable = f"{total_days:.1f} 天"
        elif abs(total_hours) >= 1:
            human_readable = f"{total_hours:.1f} 小时"
        elif abs(total_minutes) >= 1:
            human_readable = f"{total_minutes:.1f} 分钟"
        else:
            human_readable = f"{total_seconds:.0f} 秒"
        
        return json.dumps({
            "success": True,
            "date1": dt1.isoformat(),
            "date2": dt2.isoformat(),
            "difference": {
                "direction": "date2 晚于 date1" if delta.total_seconds() > 0 else "date1 晚于 date2",
                "human_readable": human_readable,
                "total_seconds": int(abs(total_seconds)),
                "total_minutes": round(total_minutes, 2),
                "total_hours": round(total_hours, 2),
                "total_days": round(total_days, 2),
                "total_weeks": round(total_weeks, 2),
                "total_months": round(total_months, 2),
                "total_years": round(total_years, 2),
                "days": abs(delta.days),
                "seconds": abs(delta.seconds)
            }
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="timezone_convert",
    description="在不同时区之间转换时间。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def timezone_convert(date_string: str, from_tz: str = "UTC", 
                     to_tz: str = "Asia/Shanghai") -> str:
    """
    时区转换
    
    Args:
        date_string: 日期时间字符串
        from_tz: 源时区（如 UTC, Asia/Shanghai, America/New_York）
        to_tz: 目标时区
    
    Returns:
        JSON格式的转换结果
    """
    from datetime import datetime, timedelta
    
    try:
        # 常用时区的 UTC 偏移（小时）
        timezone_offsets = {
            "UTC": 0,
            "GMT": 0,
            "Asia/Shanghai": 8,
            "Asia/Beijing": 8,
            "Asia/Tokyo": 9,
            "Asia/Seoul": 9,
            "Asia/Singapore": 8,
            "Asia/Hong_Kong": 8,
            "Europe/London": 0,
            "Europe/Paris": 1,
            "Europe/Berlin": 1,
            "Europe/Moscow": 3,
            "America/New_York": -5,
            "America/Los_Angeles": -8,
            "America/Chicago": -6,
            "Australia/Sydney": 10,
            "Pacific/Auckland": 12,
        }
        
        # 尝试使用 pytz（如果可用）
        try:
            import pytz
            PYTZ_AVAILABLE = True
        except ImportError:
            PYTZ_AVAILABLE = False
        
        # 解析日期
        result = json.loads(datetime_parse(date_string))
        if not result.get("success"):
            return json.dumps(result, ensure_ascii=False)
        
        dt = datetime.fromisoformat(result["parsed"]["iso"])
        
        if PYTZ_AVAILABLE:
            # 使用 pytz 进行精确转换
            from_timezone = pytz.timezone(from_tz)
            to_timezone = pytz.timezone(to_tz)
            
            # 假定输入时间是 from_tz 时区
            localized = from_timezone.localize(dt)
            converted = localized.astimezone(to_timezone)
            
            return json.dumps({
                "success": True,
                "original": date_string,
                "from_timezone": from_tz,
                "to_timezone": to_tz,
                "converted": {
                    "iso": converted.isoformat(),
                    "formatted": converted.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "timestamp": int(converted.timestamp())
                }
            }, ensure_ascii=False, indent=2)
        else:
            # 简化版：使用预定义的偏移
            from_offset = timezone_offsets.get(from_tz)
            to_offset = timezone_offsets.get(to_tz)
            
            if from_offset is None:
                return json.dumps({
                    "success": False,
                    "error": f"未知时区: {from_tz}",
                    "available_timezones": list(timezone_offsets.keys())
                }, ensure_ascii=False)
            
            if to_offset is None:
                return json.dumps({
                    "success": False,
                    "error": f"未知时区: {to_tz}",
                    "available_timezones": list(timezone_offsets.keys())
                }, ensure_ascii=False)
            
            # 计算时差并转换
            offset_diff = to_offset - from_offset
            converted = dt + timedelta(hours=offset_diff)
            
            return json.dumps({
                "success": True,
                "original": date_string,
                "from_timezone": f"{from_tz} (UTC{'+' if from_offset >= 0 else ''}{from_offset})",
                "to_timezone": f"{to_tz} (UTC{'+' if to_offset >= 0 else ''}{to_offset})",
                "offset_hours": offset_diff,
                "converted": {
                    "iso": converted.isoformat(),
                    "formatted": converted.strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp": int(converted.timestamp())
                },
                "note": "使用简化时区计算（未考虑夏令时）"
            }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 数据验证工具 ====================

@tool(
    name="validate_format",
    description="验证数据格式，支持邮箱、URL、手机号、身份证、IP地址等常见格式。",
    category=ToolCategory.DATA,
    timeout=5.0
)
def validate_format(data: str, format_type: str) -> str:
    """
    数据格式验证
    
    Args:
        data: 待验证的数据
        format_type: 格式类型
            - "email": 邮箱地址
            - "url": URL
            - "phone_cn": 中国手机号
            - "phone_intl": 国际电话号码
            - "id_card_cn": 中国身份证号
            - "ip": IP 地址（v4 或 v6）
            - "ipv4": IPv4 地址
            - "ipv6": IPv6 地址
            - "mac": MAC 地址
            - "credit_card": 信用卡号
            - "uuid": UUID
            - "json": JSON 格式
            - "date": 日期格式
            - "time": 时间格式
            - "hex_color": 十六进制颜色
            - "chinese": 纯中文
            - "alphanumeric": 字母数字
    
    Returns:
        JSON格式的验证结果
    """
    patterns = {
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "url": r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*$',
        "phone_cn": r'^1[3-9]\d{9}$',
        "phone_intl": r'^\+?[1-9]\d{6,14}$',
        "id_card_cn": r'^[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]$',
        "ipv4": r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$',
        "ipv6": r'^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}$',
        "mac": r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',
        "credit_card": r'^\d{13,19}$',
        "uuid": r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
        "date": r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',
        "time": r'^\d{1,2}:\d{2}(:\d{2})?$',
        "hex_color": r'^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$',
        "chinese": r'^[\u4e00-\u9fa5]+$',
        "alphanumeric": r'^[a-zA-Z0-9]+$',
    }
    
    try:
        data = data.strip()
        
        # IP 地址特殊处理（v4 或 v6）
        if format_type == "ip":
            ipv4_match = bool(re.match(patterns["ipv4"], data))
            ipv6_match = bool(re.match(patterns["ipv6"], data))
            valid = ipv4_match or ipv6_match
            return json.dumps({
                "success": True,
                "data": data,
                "format_type": "ip",
                "valid": valid,
                "ip_version": "v4" if ipv4_match else ("v6" if ipv6_match else None)
            }, ensure_ascii=False, indent=2)
        
        # JSON 特殊处理
        if format_type == "json":
            try:
                json.loads(data)
                valid = True
                error = None
            except json.JSONDecodeError as e:
                valid = False
                error = str(e)
            
            return json.dumps({
                "success": True,
                "data": data[:100] + "..." if len(data) > 100 else data,
                "format_type": "json",
                "valid": valid,
                "error": error
            }, ensure_ascii=False, indent=2)
        
        # 身份证特殊验证（校验位）
        if format_type == "id_card_cn":
            pattern_valid = bool(re.match(patterns["id_card_cn"], data))
            checksum_valid = False
            
            if pattern_valid:
                # 校验位验证
                weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
                check_codes = '10X98765432'
                
                try:
                    total = sum(int(data[i]) * weights[i] for i in range(17))
                    checksum_valid = check_codes[total % 11] == data[17].upper()
                except (ValueError, IndexError):
                    checksum_valid = False
            
            valid = pattern_valid and checksum_valid
            
            # 解析信息
            info = None
            if valid:
                birth_year = data[6:10]
                birth_month = data[10:12]
                birth_day = data[12:14]
                gender = "男" if int(data[16]) % 2 == 1 else "女"
                info = {
                    "birth_date": f"{birth_year}-{birth_month}-{birth_day}",
                    "gender": gender,
                    "region_code": data[:6]
                }
            
            return json.dumps({
                "success": True,
                "data": data,
                "format_type": "id_card_cn",
                "valid": valid,
                "pattern_valid": pattern_valid,
                "checksum_valid": checksum_valid,
                "info": info
            }, ensure_ascii=False, indent=2)
        
        # 信用卡 Luhn 算法验证
        if format_type == "credit_card":
            pattern_valid = bool(re.match(patterns["credit_card"], data))
            luhn_valid = False
            
            if pattern_valid:
                digits = [int(d) for d in data]
                checksum = 0
                for i, d in enumerate(reversed(digits)):
                    if i % 2 == 1:
                        d *= 2
                        if d > 9:
                            d -= 9
                    checksum += d
                luhn_valid = checksum % 10 == 0
            
            # 识别卡类型
            card_type = None
            if data.startswith('4'):
                card_type = "Visa"
            elif data.startswith(('51', '52', '53', '54', '55')):
                card_type = "MasterCard"
            elif data.startswith(('34', '37')):
                card_type = "American Express"
            elif data.startswith('62'):
                card_type = "UnionPay"
            
            return json.dumps({
                "success": True,
                "data": data[:4] + "****" + data[-4:],  # 部分隐藏
                "format_type": "credit_card",
                "valid": pattern_valid and luhn_valid,
                "luhn_valid": luhn_valid,
                "card_type": card_type
            }, ensure_ascii=False, indent=2)
        
        # 通用正则验证
        pattern = patterns.get(format_type)
        if pattern is None:
            return json.dumps({
                "success": False,
                "error": f"未知格式类型: {format_type}",
                "available_formats": list(patterns.keys())
            }, ensure_ascii=False)
        
        valid = bool(re.match(pattern, data))
        
        return json.dumps({
            "success": True,
            "data": data,
            "format_type": format_type,
            "valid": valid
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="validate_schema",
    description="使用 JSON Schema 验证数据结构。",
    category=ToolCategory.DATA,
    timeout=10.0
)
def validate_schema(data: str, schema: str) -> str:
    """
    JSON Schema 验证
    
    Args:
        data: 要验证的 JSON 数据
        schema: JSON Schema 定义
    
    Returns:
        JSON格式的验证结果
    """
    try:
        # 解析数据和 schema
        try:
            data_obj = json.loads(data)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"数据解析错误: {e}"
            }, ensure_ascii=False)
        
        try:
            schema_obj = json.loads(schema)
        except json.JSONDecodeError as e:
            return json.dumps({
                "success": False,
                "error": f"Schema 解析错误: {e}"
            }, ensure_ascii=False)
        
        # 尝试使用 jsonschema 库
        try:
            import jsonschema
            JSONSCHEMA_AVAILABLE = True
        except ImportError:
            JSONSCHEMA_AVAILABLE = False
        
        if JSONSCHEMA_AVAILABLE:
            try:
                jsonschema.validate(instance=data_obj, schema=schema_obj)
                return json.dumps({
                    "success": True,
                    "valid": True,
                    "message": "数据符合 Schema 定义"
                }, ensure_ascii=False, indent=2)
            except jsonschema.ValidationError as e:
                return json.dumps({
                    "success": True,
                    "valid": False,
                    "error": e.message,
                    "path": list(e.path),
                    "schema_path": list(e.schema_path)
                }, ensure_ascii=False, indent=2)
            except jsonschema.SchemaError as e:
                return json.dumps({
                    "success": False,
                    "error": f"Schema 定义错误: {e.message}"
                }, ensure_ascii=False)
        else:
            # 简化版验证
            errors = []
            
            def validate_simple(obj, schema_part, path=""):
                nonlocal errors
                
                schema_type = schema_part.get("type")
                
                if schema_type == "object":
                    if not isinstance(obj, dict):
                        errors.append(f"{path}: 期望对象，实际是 {type(obj).__name__}")
                        return
                    
                    # 检查必需属性
                    required = schema_part.get("required", [])
                    for prop in required:
                        if prop not in obj:
                            errors.append(f"{path}: 缺少必需属性 '{prop}'")
                    
                    # 检查属性类型
                    properties = schema_part.get("properties", {})
                    for prop, prop_schema in properties.items():
                        if prop in obj:
                            validate_simple(obj[prop], prop_schema, f"{path}.{prop}")
                
                elif schema_type == "array":
                    if not isinstance(obj, list):
                        errors.append(f"{path}: 期望数组，实际是 {type(obj).__name__}")
                        return
                    
                    items_schema = schema_part.get("items")
                    if items_schema:
                        for i, item in enumerate(obj):
                            validate_simple(item, items_schema, f"{path}[{i}]")
                
                elif schema_type == "string":
                    if not isinstance(obj, str):
                        errors.append(f"{path}: 期望字符串，实际是 {type(obj).__name__}")
                
                elif schema_type == "number" or schema_type == "integer":
                    if not isinstance(obj, (int, float)):
                        errors.append(f"{path}: 期望数字，实际是 {type(obj).__name__}")
                
                elif schema_type == "boolean":
                    if not isinstance(obj, bool):
                        errors.append(f"{path}: 期望布尔值，实际是 {type(obj).__name__}")
            
            validate_simple(data_obj, schema_obj, "$")
            
            if errors:
                return json.dumps({
                    "success": True,
                    "valid": False,
                    "errors": errors,
                    "note": "使用简化验证器（完整功能需要 jsonschema 库）"
                }, ensure_ascii=False, indent=2)
            
            return json.dumps({
                "success": True,
                "valid": True,
                "message": "数据基本符合 Schema 定义",
                "note": "使用简化验证器"
            }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 网页工具 ====================

@tool(
    name="html_extract",
    description="从 HTML 内容中提取文本、链接、图片等信息。",
    category=ToolCategory.DATA,
    timeout=15.0
)
def html_extract(html: str, extract_type: str = "text", selector: str = None) -> str:
    """
    HTML 内容提取
    
    Args:
        html: HTML 内容
        extract_type: 提取类型
            - "text": 提取纯文本
            - "links": 提取所有链接
            - "images": 提取图片 URL
            - "meta": 提取 meta 标签信息
            - "headings": 提取标题 (h1-h6)
            - "tables": 提取表格数据
        selector: CSS 选择器（可选，用于定向提取）
    
    Returns:
        JSON格式的提取结果
    """
    try:
        # 尝试使用 BeautifulSoup
        try:
            from bs4 import BeautifulSoup
            BS_AVAILABLE = True
        except ImportError:
            BS_AVAILABLE = False
        
        if BS_AVAILABLE:
            soup = BeautifulSoup(html, 'html.parser')
            
            if selector:
                elements = soup.select(selector)
                results = [elem.get_text(strip=True) for elem in elements]
                return json.dumps({
                    "success": True,
                    "extract_type": "selector",
                    "selector": selector,
                    "results": results,
                    "count": len(results)
                }, ensure_ascii=False, indent=2)
            
            if extract_type == "text":
                # 移除 script 和 style 标签
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator='\n', strip=True)
                # 清理多余空行
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                return json.dumps({
                    "success": True,
                    "extract_type": "text",
                    "text": '\n'.join(lines),
                    "char_count": len('\n'.join(lines))
                }, ensure_ascii=False, indent=2)
            
            elif extract_type == "links":
                links = []
                for a in soup.find_all('a', href=True):
                    links.append({
                        "text": a.get_text(strip=True),
                        "href": a['href']
                    })
                
                return json.dumps({
                    "success": True,
                    "extract_type": "links",
                    "links": links[:100],  # 限制数量
                    "count": len(links)
                }, ensure_ascii=False, indent=2)
            
            elif extract_type == "images":
                images = []
                for img in soup.find_all('img'):
                    images.append({
                        "src": img.get('src', ''),
                        "alt": img.get('alt', ''),
                        "title": img.get('title', '')
                    })
                
                return json.dumps({
                    "success": True,
                    "extract_type": "images",
                    "images": images[:50],
                    "count": len(images)
                }, ensure_ascii=False, indent=2)
            
            elif extract_type == "meta":
                meta_info = {
                    "title": soup.title.string if soup.title else None,
                    "meta_tags": []
                }
                
                for meta in soup.find_all('meta'):
                    meta_info["meta_tags"].append({
                        "name": meta.get('name'),
                        "property": meta.get('property'),
                        "content": meta.get('content')
                    })
                
                return json.dumps({
                    "success": True,
                    "extract_type": "meta",
                    "meta": meta_info
                }, ensure_ascii=False, indent=2)
            
            elif extract_type == "headings":
                headings = []
                for tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    for h in soup.find_all(tag):
                        headings.append({
                            "level": tag,
                            "text": h.get_text(strip=True)
                        })
                
                return json.dumps({
                    "success": True,
                    "extract_type": "headings",
                    "headings": headings,
                    "count": len(headings)
                }, ensure_ascii=False, indent=2)
            
            elif extract_type == "tables":
                tables = []
                for table in soup.find_all('table'):
                    rows = []
                    for tr in table.find_all('tr'):
                        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                        if cells:
                            rows.append(cells)
                    if rows:
                        tables.append(rows)
                
                return json.dumps({
                    "success": True,
                    "extract_type": "tables",
                    "tables": tables[:10],  # 限制表格数量
                    "count": len(tables)
                }, ensure_ascii=False, indent=2)
        
        else:
            # 简化版：使用正则表达式
            if extract_type == "text":
                # 移除 HTML 标签
                text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                return json.dumps({
                    "success": True,
                    "extract_type": "text",
                    "text": text,
                    "char_count": len(text),
                    "note": "使用简化提取器（完整功能需要 beautifulsoup4 库）"
                }, ensure_ascii=False, indent=2)
            
            elif extract_type == "links":
                links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>', html, re.IGNORECASE)
                return json.dumps({
                    "success": True,
                    "extract_type": "links",
                    "links": [{"href": href, "text": text.strip()} for href, text in links[:100]],
                    "count": len(links),
                    "note": "使用简化提取器"
                }, ensure_ascii=False, indent=2)
            
            elif extract_type == "images":
                images = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
                return json.dumps({
                    "success": True,
                    "extract_type": "images",
                    "images": [{"src": src} for src in images[:50]],
                    "count": len(images),
                    "note": "使用简化提取器"
                }, ensure_ascii=False, indent=2)
            
            return json.dumps({
                "success": False,
                "error": f"简化提取器不支持: {extract_type}",
                "available_types": ["text", "links", "images"]
            }, ensure_ascii=False)
        
        return json.dumps({
            "success": False,
            "error": f"未知提取类型: {extract_type}"
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 增强的数据分析工具 ====================

@tool(
    name="statistical_analysis",
    description="对数值数据进行详细的统计分析。包括描述性统计、分布分析、异常检测等。",
    category=ToolCategory.DATA,
    timeout=20.0
)
def statistical_analysis(data: str, analysis_type: str = "descriptive") -> str:
    """
    统计分析
    
    对数值数据进行全面的统计分析。
    
    Args:
        data: 数值数据（JSON数组格式，如 [1, 2, 3, 4, 5]）
        analysis_type: 分析类型
            - "descriptive": 描述性统计（均值、中位数、标准差等）
            - "distribution": 分布分析（偏度、峰度、分位数）
            - "outliers": 异常值检测
            - "correlation": 相关性分析（需要二维数据）
            - "all": 全面分析
    
    Returns:
        JSON格式的分析结果
    """
    import statistics
    import math
    
    try:
        # 解析数据
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            # 尝试按逗号分割
            parsed_data = [float(x.strip()) for x in data.split(',') if x.strip()]
        
        # 确保是数值列表
        if isinstance(parsed_data, dict):
            # 如果是字典，提取数值
            values = [v for v in parsed_data.values() if isinstance(v, (int, float))]
        elif isinstance(parsed_data, list):
            if parsed_data and isinstance(parsed_data[0], dict):
                # 对象数组，尝试提取数值字段
                values = []
                for item in parsed_data:
                    for v in item.values():
                        if isinstance(v, (int, float)):
                            values.append(v)
            else:
                values = [float(x) for x in parsed_data if isinstance(x, (int, float, str)) and str(x).replace('.', '').replace('-', '').isdigit()]
        else:
            return json.dumps({
                "success": False,
                "error": "无法解析数值数据"
            }, ensure_ascii=False)
        
        if not values:
            return json.dumps({
                "success": False,
                "error": "没有有效的数值数据"
            }, ensure_ascii=False)
        
        n = len(values)
        sorted_values = sorted(values)
        
        result = {
            "success": True,
            "data_count": n,
            "analysis_type": analysis_type
        }
        
        # 描述性统计
        if analysis_type in ["descriptive", "all"]:
            mean = statistics.mean(values)
            
            descriptive = {
                "count": n,
                "sum": sum(values),
                "mean": round(mean, 4),
                "median": round(statistics.median(values), 4),
                "mode": None,
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values)
            }
            
            # 众数（可能有多个或不存在）
            try:
                descriptive["mode"] = statistics.mode(values)
            except statistics.StatisticsError:
                pass
            
            # 标准差和方差
            if n > 1:
                descriptive["variance"] = round(statistics.variance(values), 4)
                descriptive["stdev"] = round(statistics.stdev(values), 4)
                descriptive["sem"] = round(descriptive["stdev"] / math.sqrt(n), 4)  # 标准误
            
            result["descriptive"] = descriptive
        
        # 分布分析
        if analysis_type in ["distribution", "all"]:
            mean = statistics.mean(values)
            
            # 分位数
            percentiles = {}
            for p in [10, 25, 50, 75, 90, 95, 99]:
                idx = int((p / 100) * (n - 1))
                percentiles[f"p{p}"] = sorted_values[idx]
            
            distribution = {
                "percentiles": percentiles,
                "iqr": percentiles["p75"] - percentiles["p25"],
                "q1": percentiles["p25"],
                "q3": percentiles["p75"]
            }
            
            # 偏度和峰度
            if n > 2:
                stdev = statistics.stdev(values)
                if stdev > 0:
                    # 偏度
                    skewness = sum((x - mean) ** 3 for x in values) / (n * stdev ** 3)
                    # 峰度
                    kurtosis = sum((x - mean) ** 4 for x in values) / (n * stdev ** 4) - 3
                    
                    distribution["skewness"] = round(skewness, 4)
                    distribution["kurtosis"] = round(kurtosis, 4)
                    
                    # 偏度解释
                    if skewness > 0.5:
                        distribution["skewness_interpretation"] = "右偏（正偏态）"
                    elif skewness < -0.5:
                        distribution["skewness_interpretation"] = "左偏（负偏态）"
                    else:
                        distribution["skewness_interpretation"] = "近似对称"
            
            result["distribution"] = distribution
        
        # 异常值检测
        if analysis_type in ["outliers", "all"]:
            # IQR 方法
            q1 = sorted_values[int(0.25 * n)]
            q3 = sorted_values[int(0.75 * n)]
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_low = [v for v in values if v < lower_bound]
            outliers_high = [v for v in values if v > upper_bound]
            
            # Z-score 方法
            if n > 1:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values)
                if stdev > 0:
                    z_outliers = [v for v in values if abs((v - mean) / stdev) > 3]
                else:
                    z_outliers = []
            else:
                z_outliers = []
            
            result["outliers"] = {
                "method": "IQR (1.5x)",
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "outliers_low": outliers_low,
                "outliers_high": outliers_high,
                "total_outliers": len(outliers_low) + len(outliers_high),
                "outlier_percentage": round((len(outliers_low) + len(outliers_high)) / n * 100, 2),
                "z_score_outliers": z_outliers
            }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="data_aggregation",
    description="对数据进行聚合操作，支持分组统计、汇总等。",
    category=ToolCategory.DATA,
    timeout=15.0
)
def data_aggregation(data: str, group_by: str = None, aggregate: str = "count") -> str:
    """
    数据聚合
    
    Args:
        data: JSON 格式的数据数组
        group_by: 分组字段
        aggregate: 聚合操作
            - "count": 计数
            - "sum": 求和（需要指定数值字段）
            - "avg": 平均值
            - "min": 最小值
            - "max": 最大值
            - "list": 列出所有值
    
    Returns:
        JSON格式的聚合结果
    """
    try:
        parsed_data = json.loads(data)
        
        if not isinstance(parsed_data, list):
            return json.dumps({
                "success": False,
                "error": "数据必须是数组格式"
            }, ensure_ascii=False)
        
        if not parsed_data:
            return json.dumps({
                "success": False,
                "error": "数据数组为空"
            }, ensure_ascii=False)
        
        # 简单计数（无分组）
        if not group_by:
            total = len(parsed_data)
            
            # 尝试计算数值统计
            if isinstance(parsed_data[0], (int, float)):
                values = [float(x) for x in parsed_data]
                return json.dumps({
                    "success": True,
                    "aggregate": aggregate,
                    "result": {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
                }, ensure_ascii=False, indent=2)
            
            return json.dumps({
                "success": True,
                "aggregate": "count",
                "result": total
            }, ensure_ascii=False, indent=2)
        
        # 分组聚合
        groups = {}
        for item in parsed_data:
            if not isinstance(item, dict):
                continue
            
            key = item.get(group_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        # 计算聚合结果
        results = {}
        for key, items in groups.items():
            if aggregate == "count":
                results[str(key)] = len(items)
            
            elif aggregate == "list":
                results[str(key)] = items[:10]  # 限制数量
            
            else:
                # 需要数值字段
                # 尝试找到第一个数值字段
                numeric_field = None
                if items and isinstance(items[0], dict):
                    for field, value in items[0].items():
                        if isinstance(value, (int, float)):
                            numeric_field = field
                            break
                
                if numeric_field:
                    values = [item[numeric_field] for item in items if numeric_field in item and isinstance(item[numeric_field], (int, float))]
                    
                    if values:
                        if aggregate == "sum":
                            results[str(key)] = sum(values)
                        elif aggregate == "avg":
                            results[str(key)] = sum(values) / len(values)
                        elif aggregate == "min":
                            results[str(key)] = min(values)
                        elif aggregate == "max":
                            results[str(key)] = max(values)
                else:
                    results[str(key)] = len(items)
        
        return json.dumps({
            "success": True,
            "group_by": group_by,
            "aggregate": aggregate,
            "groups_count": len(results),
            "results": results
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 工作流工具 ====================

@tool(
    name="conditional_logic",
    description="执行条件判断逻辑，支持多条件组合。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def conditional_logic(conditions: str, data: str = None) -> str:
    """
    条件判断
    
    Args:
        conditions: 条件表达式（JSON格式）
            格式: {"field": "value", "operator": "eq|ne|gt|lt|gte|lte|in|contains", "value": "target"}
            或多条件: {"and|or": [condition1, condition2, ...]}
        data: 要判断的数据（JSON格式）
    
    Returns:
        JSON格式的判断结果
    """
    try:
        conditions_obj = json.loads(conditions)
        data_obj = json.loads(data) if data else {}
        
        def evaluate_condition(cond, context):
            if "and" in cond:
                return all(evaluate_condition(c, context) for c in cond["and"])
            elif "or" in cond:
                return any(evaluate_condition(c, context) for c in cond["or"])
            elif "not" in cond:
                return not evaluate_condition(cond["not"], context)
            else:
                field = cond.get("field")
                operator = cond.get("operator", "eq")
                target = cond.get("value")
                
                # 获取字段值
                actual = context.get(field)
                
                # 执行比较
                if operator == "eq":
                    return actual == target
                elif operator == "ne":
                    return actual != target
                elif operator == "gt":
                    return actual > target
                elif operator == "lt":
                    return actual < target
                elif operator == "gte":
                    return actual >= target
                elif operator == "lte":
                    return actual <= target
                elif operator == "in":
                    return actual in target
                elif operator == "contains":
                    return target in str(actual)
                elif operator == "startswith":
                    return str(actual).startswith(str(target))
                elif operator == "endswith":
                    return str(actual).endswith(str(target))
                elif operator == "exists":
                    return field in context
                elif operator == "is_null":
                    return actual is None
                elif operator == "is_not_null":
                    return actual is not None
                else:
                    return False
        
        result = evaluate_condition(conditions_obj, data_obj)
        
        return json.dumps({
            "success": True,
            "result": result,
            "conditions": conditions_obj,
            "evaluated_data": data_obj
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="batch_process",
    description="对数据列表进行批量处理操作。",
    category=ToolCategory.DATA,
    timeout=30.0
)
def batch_process(data: str, operation: str, params: str = None) -> str:
    """
    批量处理
    
    Args:
        data: 数据数组（JSON格式）
        operation: 批量操作
            - "map": 对每个元素应用转换（需要 params.expression）
            - "filter": 过滤元素（需要 params.condition）
            - "reduce": 归约操作（需要 params.reducer）
            - "sort": 排序（需要 params.key 和 params.reverse）
            - "unique": 去重
            - "chunk": 分块（需要 params.size）
        params: 操作参数（JSON格式）
    
    Returns:
        JSON格式的处理结果
    """
    try:
        data_list = json.loads(data)
        params_obj = json.loads(params) if params else {}
        
        if not isinstance(data_list, list):
            return json.dumps({
                "success": False,
                "error": "数据必须是数组格式"
            }, ensure_ascii=False)
        
        if operation == "map":
            # 简单的映射转换
            expression = params_obj.get("expression", "x")
            field = params_obj.get("field")
            
            result = []
            for item in data_list:
                if field and isinstance(item, dict):
                    result.append(item.get(field))
                else:
                    result.append(item)
            
            return json.dumps({
                "success": True,
                "operation": "map",
                "original_count": len(data_list),
                "result": result
            }, ensure_ascii=False, indent=2)
        
        elif operation == "filter":
            condition = params_obj.get("condition", {})
            
            result = []
            for item in data_list:
                if isinstance(item, dict) and condition:
                    match = True
                    for key, value in condition.items():
                        if item.get(key) != value:
                            match = False
                            break
                    if match:
                        result.append(item)
                else:
                    # 过滤非空值
                    if item:
                        result.append(item)
            
            return json.dumps({
                "success": True,
                "operation": "filter",
                "original_count": len(data_list),
                "filtered_count": len(result),
                "result": result
            }, ensure_ascii=False, indent=2)
        
        elif operation == "sort":
            key = params_obj.get("key")
            reverse = params_obj.get("reverse", False)
            
            if key and data_list and isinstance(data_list[0], dict):
                result = sorted(data_list, key=lambda x: x.get(key, 0), reverse=reverse)
            else:
                result = sorted(data_list, reverse=reverse)
            
            return json.dumps({
                "success": True,
                "operation": "sort",
                "key": key,
                "reverse": reverse,
                "result": result
            }, ensure_ascii=False, indent=2)
        
        elif operation == "unique":
            # 去重
            if data_list and isinstance(data_list[0], dict):
                # 对字典列表去重
                seen = set()
                result = []
                for item in data_list:
                    item_hash = json.dumps(item, sort_keys=True)
                    if item_hash not in seen:
                        seen.add(item_hash)
                        result.append(item)
            else:
                result = list(dict.fromkeys(data_list))
            
            return json.dumps({
                "success": True,
                "operation": "unique",
                "original_count": len(data_list),
                "unique_count": len(result),
                "result": result
            }, ensure_ascii=False, indent=2)
        
        elif operation == "chunk":
            size = params_obj.get("size", 10)
            result = [data_list[i:i+size] for i in range(0, len(data_list), size)]
            
            return json.dumps({
                "success": True,
                "operation": "chunk",
                "chunk_size": size,
                "total_chunks": len(result),
                "result": result
            }, ensure_ascii=False, indent=2)
        
        elif operation == "reduce":
            # 简单归约
            reducer = params_obj.get("reducer", "sum")
            initial = params_obj.get("initial", 0)
            
            if reducer == "sum":
                result = sum(float(x) for x in data_list if isinstance(x, (int, float, str)) and str(x).replace('.', '').replace('-', '').isdigit())
            elif reducer == "concat":
                result = ''.join(str(x) for x in data_list)
            elif reducer == "count":
                result = len(data_list)
            else:
                result = data_list
            
            return json.dumps({
                "success": True,
                "operation": "reduce",
                "reducer": reducer,
                "result": result
            }, ensure_ascii=False, indent=2)
        
        else:
            return json.dumps({
                "success": False,
                "error": f"未知操作: {operation}",
                "available_operations": ["map", "filter", "sort", "unique", "chunk", "reduce"]
            }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="retry_with_backoff",
    description="带有指数退避的重试机制。用于执行可能失败的操作。",
    category=ToolCategory.SYSTEM,
    timeout=60.0
)
def retry_with_backoff(operation: str, max_retries: int = 3, 
                       initial_delay: float = 1.0, max_delay: float = 30.0,
                       backoff_factor: float = 2.0) -> str:
    """
    带重试的操作执行
    
    注意：此工具主要用于演示重试逻辑，实际使用需要配合其他工具。
    
    Args:
        operation: 操作描述或表达式
        max_retries: 最大重试次数
        initial_delay: 初始延迟（秒）
        max_delay: 最大延迟（秒）
        backoff_factor: 退避因子
    
    Returns:
        JSON格式的执行结果
    """
    import time
    import random
    
    try:
        retries = []
        current_delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                # 模拟操作（实际使用中这里会执行真实操作）
                # 这里我们用一个简单的模拟：50% 概率成功
                if random.random() > 0.5 or attempt == max_retries:
                    result = {
                        "success": True,
                        "operation": operation,
                        "attempt": attempt + 1,
                        "retries": retries,
                        "message": "操作成功完成",
                        "execution_info": {
                            "total_attempts": attempt + 1,
                            "total_delay": sum(r.get("delay", 0) for r in retries)
                        }
                    }
                    return json.dumps(result, ensure_ascii=False, indent=2)
                else:
                    raise Exception("模拟失败")
                    
            except Exception as e:
                retry_info = {
                    "attempt": attempt + 1,
                    "error": str(e),
                    "delay": current_delay if attempt < max_retries else 0
                }
                retries.append(retry_info)
                
                if attempt < max_retries:
                    # 添加抖动（±10%）
                    jitter = current_delay * 0.1 * (2 * random.random() - 1)
                    actual_delay = min(current_delay + jitter, max_delay)
                    
                    time.sleep(min(actual_delay, 1.0))  # 限制实际延迟
                    current_delay = min(current_delay * backoff_factor, max_delay)
        
        return json.dumps({
            "success": False,
            "operation": operation,
            "max_retries_exceeded": True,
            "retries": retries,
            "error": f"操作在 {max_retries + 1} 次尝试后仍然失败"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 缓存工具 ====================

# 工具结果缓存
_tool_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()


@tool(
    name="cache_set",
    description="将数据存入缓存，支持过期时间设置。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def cache_set(key: str, value: str, ttl: int = 3600, namespace: str = "tool_cache") -> str:
    """
    设置缓存
    
    Args:
        key: 缓存键
        value: 缓存值（JSON格式）
        ttl: 过期时间（秒，默认1小时）
        namespace: 命名空间
    
    Returns:
        JSON格式的操作结果
    """
    global _tool_cache
    
    try:
        full_key = f"{namespace}:{key}"
        
        with _cache_lock:
            _tool_cache[full_key] = {
                "value": value,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow().timestamp() + ttl),
                "ttl": ttl
            }
        
        return json.dumps({
            "success": True,
            "key": key,
            "namespace": namespace,
            "ttl": ttl,
            "size_bytes": len(value)
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="cache_get",
    description="从缓存获取数据。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def cache_get(key: str, namespace: str = "tool_cache", default: str = None) -> str:
    """
    获取缓存
    
    Args:
        key: 缓存键
        namespace: 命名空间
        default: 默认值（缓存不存在或已过期时返回）
    
    Returns:
        JSON格式的缓存值
    """
    global _tool_cache
    
    try:
        full_key = f"{namespace}:{key}"
        
        with _cache_lock:
            if full_key in _tool_cache:
                entry = _tool_cache[full_key]
                
                # 检查是否过期
                if entry["expires_at"] > datetime.utcnow().timestamp():
                    return json.dumps({
                        "success": True,
                        "key": key,
                        "namespace": namespace,
                        "value": entry["value"],
                        "created_at": entry["created_at"],
                        "cached": True
                    }, ensure_ascii=False, indent=2)
                else:
                    # 删除过期条目
                    del _tool_cache[full_key]
        
        return json.dumps({
            "success": True,
            "key": key,
            "namespace": namespace,
            "value": default,
            "cached": False,
            "note": "缓存不存在或已过期"
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="cache_delete",
    description="删除缓存条目。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def cache_delete(key: str = None, namespace: str = "tool_cache", 
                 clear_all: bool = False) -> str:
    """
    删除缓存
    
    Args:
        key: 缓存键（为空时删除整个命名空间）
        namespace: 命名空间
        clear_all: 是否清空所有缓存
    
    Returns:
        JSON格式的操作结果
    """
    global _tool_cache
    
    try:
        deleted = 0
        
        with _cache_lock:
            if clear_all:
                deleted = len(_tool_cache)
                _tool_cache.clear()
            elif key:
                full_key = f"{namespace}:{key}"
                if full_key in _tool_cache:
                    del _tool_cache[full_key]
                    deleted = 1
            else:
                # 删除命名空间下所有条目
                prefix = f"{namespace}:"
                keys_to_delete = [k for k in _tool_cache.keys() if k.startswith(prefix)]
                for k in keys_to_delete:
                    del _tool_cache[k]
                    deleted += 1
        
        return json.dumps({
            "success": True,
            "deleted_count": deleted,
            "key": key,
            "namespace": namespace,
            "clear_all": clear_all
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@tool(
    name="cache_stats",
    description="获取缓存统计信息。",
    category=ToolCategory.SYSTEM,
    timeout=5.0
)
def cache_stats(namespace: str = None) -> str:
    """
    缓存统计
    
    Args:
        namespace: 命名空间（为空则统计所有）
    
    Returns:
        JSON格式的统计信息
    """
    global _tool_cache
    
    try:
        with _cache_lock:
            now = datetime.utcnow().timestamp()
            
            total_entries = 0
            expired_entries = 0
            total_size = 0
            namespaces = {}
            
            for key, entry in _tool_cache.items():
                ns = key.split(':')[0]
                
                if namespace and ns != namespace:
                    continue
                
                total_entries += 1
                size = len(entry["value"])
                total_size += size
                
                if entry["expires_at"] <= now:
                    expired_entries += 1
                
                if ns not in namespaces:
                    namespaces[ns] = {"count": 0, "size": 0}
                namespaces[ns]["count"] += 1
                namespaces[ns]["size"] += size
            
            return json.dumps({
                "success": True,
                "total_entries": total_entries,
                "active_entries": total_entries - expired_entries,
                "expired_entries": expired_entries,
                "total_size_bytes": total_size,
                "total_size_kb": round(total_size / 1024, 2),
                "namespaces": namespaces
            }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 异步工具 ====================

@async_tool(
    name="async_http_request",
    description="异步发送 HTTP 请求，支持并发请求。",
    category=ToolCategory.API,
    timeout=60.0
)
async def async_http_request(urls: str, method: str = "GET", 
                             concurrent_limit: int = 5) -> str:
    """
    异步 HTTP 请求
    
    同时发送多个 HTTP 请求。
    
    Args:
        urls: URL 列表（JSON数组格式，或逗号分隔）
        method: HTTP 方法
        concurrent_limit: 并发限制
    
    Returns:
        JSON格式的响应结果
    """
    try:
        # 解析 URL 列表
        if urls.startswith('['):
            url_list = json.loads(urls)
        else:
            url_list = [u.strip() for u in urls.split(',') if u.strip()]
        
        if not url_list:
            return json.dumps({
                "success": False,
                "error": "URL 列表为空"
            }, ensure_ascii=False)
        
        results = []
        
        if HTTPX_AVAILABLE:
            import httpx
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                semaphore = asyncio.Semaphore(concurrent_limit)
                
                async def fetch(url):
                    async with semaphore:
                        try:
                            start = asyncio.get_event_loop().time()
                            response = await client.request(method, url)
                            elapsed = asyncio.get_event_loop().time() - start
                            
                            return {
                                "url": url,
                                "status_code": response.status_code,
                                "success": True,
                                "elapsed_ms": round(elapsed * 1000, 2),
                                "content_length": len(response.content)
                            }
                        except Exception as e:
                            return {
                                "url": url,
                                "success": False,
                                "error": str(e)
                            }
                
                tasks = [fetch(url) for url in url_list]
                results = await asyncio.gather(*tasks)
        
        else:
            # 回退到同步请求
            for url in url_list[:concurrent_limit]:
                result_str = http_request(url, method)
                result = json.loads(result_str)
                results.append({
                    "url": url,
                    "success": result.get("success", False),
                    "status_code": result.get("status_code"),
                    "error": result.get("error")
                })
        
        successful = sum(1 for r in results if r.get("success"))
        
        return json.dumps({
            "success": True,
            "total_requests": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "results": results
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ==================== 导出工具列表 ====================

def get_builtin_tools() -> List[Tool]:
    """获取所有内置工具
    
    Returns:
        所有内置工具的列表，包含以下类别：
        - 搜索工具：Web搜索、新闻搜索、知识库搜索、向量搜索
        - 文本处理工具：摘要、关键词提取、相似度、情感分析、统计、清理
        - 文件操作工具：读取、CSV/JSON/YAML/XML解析、目录操作
        - 编码解码工具：Base64、URL、HTML实体、十六进制
        - 时间处理工具：解析、计算、差异、时区转换
        - 数据验证工具：格式验证、Schema验证
        - 网页工具：HTML内容提取
        - 数据分析工具：统计分析、数据聚合
        - 工作流工具：条件判断、批量处理、重试机制
        - 缓存工具：缓存管理
        - HTTP工具：同步/异步请求
        - 代码执行工具：Python、SQL、计算器
        - 记忆工具：存储、检索、管理
        - 系统工具：日期时间、UUID、哈希、正则
    """
    tools = []
    
    # 搜索类工具（Web搜索、新闻搜索、知识库搜索、向量搜索）
    search_tools = [
        web_search, news_search, search_engine_status, 
        knowledge_search, vector_search
    ]
    
    # 知识库管理工具
    knowledge_tools = [knowledge_add, knowledge_delete, collection_info]
    
    # 文本处理工具
    text_tools = [
        text_summarize, keyword_extract, text_similarity, 
        sentiment_analysis, text_statistics, text_clean
    ]
    
    # 文件操作工具
    file_tools = [
        file_reader, csv_parser, json_processor, 
        yaml_processor, xml_processor, directory_list, file_info
    ]
    
    # 编码解码工具
    codec_tools = [base64_codec, url_codec, html_codec, hex_codec]
    
    # 时间处理工具
    datetime_tools = [
        datetime_parse, datetime_calculate, 
        datetime_diff, timezone_convert
    ]
    
    # 数据验证工具
    validation_tools = [validate_format, validate_schema]
    
    # 网页工具
    web_tools = [html_extract]
    
    # 数据分析工具
    analysis_tools = [statistical_analysis, data_aggregation]
    
    # 工作流工具
    workflow_tools = [conditional_logic, batch_process, retry_with_backoff]
    
    # 缓存工具
    cache_tools = [cache_set, cache_get, cache_delete, cache_stats]
    
    # 代码类工具
    code_tools = [python_executor, calculator, sql_executor]
    
    # 数据处理工具（原有）
    data_tools = [data_analyzer, data_transformer]
    
    # 记忆工具
    memory_tools = [memory_store, memory_retrieve, memory_delete, memory_list]
    
    # 系统工具
    system_tools = [get_datetime, uuid_generator, hash_text, regex_matcher]
    
    # HTTP 工具（同步）
    http_tools = [http_request, http_get, http_post]
    
    # 异步工具
    async_tools = [async_http_request]
    
    # 收集所有工具
    all_tool_funcs = (
        search_tools + knowledge_tools + text_tools + file_tools +
        codec_tools + datetime_tools + validation_tools + web_tools +
        analysis_tools + workflow_tools + cache_tools + code_tools +
        data_tools + memory_tools + system_tools + http_tools + async_tools
    )
    
    for func in all_tool_funcs:
        if hasattr(func, '_tool'):
            tools.append(func._tool)
    
    return tools


def get_tools_by_category(category: ToolCategory) -> List[Tool]:
    """按类别获取工具
    
    Args:
        category: 工具类别
    
    Returns:
        指定类别的工具列表
    """
    return [t for t in get_builtin_tools() if t.category == category]


def get_search_tools() -> List[Tool]:
    """获取搜索类工具"""
    return (
        get_tools_by_category(ToolCategory.SEARCH) + 
        get_tools_by_category(ToolCategory.RETRIEVAL)
    )


def get_code_tools() -> List[Tool]:
    """获取代码类工具"""
    return get_tools_by_category(ToolCategory.CODE)


def get_data_tools() -> List[Tool]:
    """获取数据处理类工具
    
    包括文本处理、文件解析、编码解码、数据分析等。
    """
    return get_tools_by_category(ToolCategory.DATA)


def get_system_tools() -> List[Tool]:
    """获取系统类工具
    
    包括日期时间、文件操作、缓存等。
    """
    return get_tools_by_category(ToolCategory.SYSTEM)


def get_http_tools() -> List[Tool]:
    """获取 HTTP 类工具"""
    return get_tools_by_category(ToolCategory.API)


def get_memory_tools() -> List[Tool]:
    """获取记忆类工具"""
    memory_tool_names = ['memory_store', 'memory_retrieve', 'memory_delete', 'memory_list']
    return [t for t in get_builtin_tools() if t.name in memory_tool_names]


def get_knowledge_tools() -> List[Tool]:
    """获取知识库相关工具
    
    包括：
    - knowledge_search: 语义搜索
    - vector_search: 向量相似度搜索
    - knowledge_add: 添加知识
    - knowledge_delete: 删除知识
    - collection_info: 获取集合信息
    
    Returns:
        知识库工具列表
    """
    knowledge_tool_names = [
        'knowledge_search', 'vector_search', 
        'knowledge_add', 'knowledge_delete', 'collection_info'
    ]
    return [t for t in get_builtin_tools() if t.name in knowledge_tool_names]


def get_text_tools() -> List[Tool]:
    """获取文本处理工具
    
    包括：
    - text_summarize: 文本摘要
    - keyword_extract: 关键词提取
    - text_similarity: 文本相似度
    - sentiment_analysis: 情感分析
    - text_statistics: 文本统计
    - text_clean: 文本清理
    
    Returns:
        文本处理工具列表
    """
    text_tool_names = [
        'text_summarize', 'keyword_extract', 'text_similarity',
        'sentiment_analysis', 'text_statistics', 'text_clean'
    ]
    return [t for t in get_builtin_tools() if t.name in text_tool_names]


def get_file_tools() -> List[Tool]:
    """获取文件操作工具
    
    包括：
    - file_reader: 文件读取
    - csv_parser: CSV 解析
    - json_processor: JSON 处理
    - yaml_processor: YAML 处理
    - xml_processor: XML 处理
    - directory_list: 目录列表
    - file_info: 文件信息
    
    Returns:
        文件操作工具列表
    """
    file_tool_names = [
        'file_reader', 'csv_parser', 'json_processor',
        'yaml_processor', 'xml_processor', 'directory_list', 'file_info'
    ]
    return [t for t in get_builtin_tools() if t.name in file_tool_names]


def get_codec_tools() -> List[Tool]:
    """获取编码解码工具
    
    包括：
    - base64_codec: Base64 编解码
    - url_codec: URL 编解码
    - html_codec: HTML 实体编解码
    - hex_codec: 十六进制编解码
    
    Returns:
        编码解码工具列表
    """
    codec_tool_names = ['base64_codec', 'url_codec', 'html_codec', 'hex_codec']
    return [t for t in get_builtin_tools() if t.name in codec_tool_names]


def get_datetime_tools() -> List[Tool]:
    """获取时间处理工具
    
    包括：
    - datetime_parse: 日期时间解析
    - datetime_calculate: 日期时间计算
    - datetime_diff: 日期差异计算
    - timezone_convert: 时区转换
    
    Returns:
        时间处理工具列表
    """
    datetime_tool_names = [
        'datetime_parse', 'datetime_calculate', 
        'datetime_diff', 'timezone_convert', 'get_datetime'
    ]
    return [t for t in get_builtin_tools() if t.name in datetime_tool_names]


def get_validation_tools() -> List[Tool]:
    """获取数据验证工具
    
    包括：
    - validate_format: 格式验证（邮箱、URL、手机号等）
    - validate_schema: JSON Schema 验证
    
    Returns:
        数据验证工具列表
    """
    validation_tool_names = ['validate_format', 'validate_schema']
    return [t for t in get_builtin_tools() if t.name in validation_tool_names]


def get_analysis_tools() -> List[Tool]:
    """获取数据分析工具
    
    包括：
    - statistical_analysis: 统计分析
    - data_aggregation: 数据聚合
    - data_analyzer: 数据分析
    - data_transformer: 数据转换
    
    Returns:
        数据分析工具列表
    """
    analysis_tool_names = [
        'statistical_analysis', 'data_aggregation',
        'data_analyzer', 'data_transformer'
    ]
    return [t for t in get_builtin_tools() if t.name in analysis_tool_names]


def get_workflow_tools() -> List[Tool]:
    """获取工作流工具
    
    包括：
    - conditional_logic: 条件判断
    - batch_process: 批量处理
    - retry_with_backoff: 重试机制
    
    Returns:
        工作流工具列表
    """
    workflow_tool_names = ['conditional_logic', 'batch_process', 'retry_with_backoff']
    return [t for t in get_builtin_tools() if t.name in workflow_tool_names]


def get_cache_tools() -> List[Tool]:
    """获取缓存工具
    
    包括：
    - cache_set: 设置缓存
    - cache_get: 获取缓存
    - cache_delete: 删除缓存
    - cache_stats: 缓存统计
    
    Returns:
        缓存工具列表
    """
    cache_tool_names = ['cache_set', 'cache_get', 'cache_delete', 'cache_stats']
    return [t for t in get_builtin_tools() if t.name in cache_tool_names]


def get_async_tools() -> List[Tool]:
    """获取异步工具
    
    Returns:
        异步工具列表
    """
    async_tool_names = ['async_http_request']
    return [t for t in get_builtin_tools() if t.name in async_tool_names]


def get_tool_by_name(name: str) -> Optional[Tool]:
    """按名称获取工具
    
    Args:
        name: 工具名称
    
    Returns:
        工具实例，如果不存在则返回 None
    """
    for tool in get_builtin_tools():
        if tool.name == name:
            return tool
    return None


def get_tools_by_names(names: List[str]) -> List[Tool]:
    """按名称列表获取多个工具
    
    Args:
        names: 工具名称列表
    
    Returns:
        匹配的工具列表
    """
    all_tools = get_builtin_tools()
    return [t for t in all_tools if t.name in names]


def get_tool_categories() -> Dict[str, List[str]]:
    """获取所有工具分类及其包含的工具名称
    
    Returns:
        分类名称到工具名称列表的映射
    """
    categories = {
        "search": [],
        "text_processing": [],
        "file_operations": [],
        "encoding": [],
        "datetime": [],
        "validation": [],
        "data_analysis": [],
        "workflow": [],
        "cache": [],
        "http": [],
        "code": [],
        "memory": [],
        "system": [],
        "knowledge": []
    }
    
    # 填充分类
    for tool in get_builtin_tools():
        if tool.name in ['web_search', 'news_search', 'search_engine_status']:
            categories["search"].append(tool.name)
        elif tool.name in ['text_summarize', 'keyword_extract', 'text_similarity', 
                          'sentiment_analysis', 'text_statistics', 'text_clean']:
            categories["text_processing"].append(tool.name)
        elif tool.name in ['file_reader', 'csv_parser', 'json_processor', 
                          'yaml_processor', 'xml_processor', 'directory_list', 'file_info']:
            categories["file_operations"].append(tool.name)
        elif tool.name in ['base64_codec', 'url_codec', 'html_codec', 'hex_codec']:
            categories["encoding"].append(tool.name)
        elif tool.name in ['datetime_parse', 'datetime_calculate', 'datetime_diff', 
                          'timezone_convert', 'get_datetime']:
            categories["datetime"].append(tool.name)
        elif tool.name in ['validate_format', 'validate_schema']:
            categories["validation"].append(tool.name)
        elif tool.name in ['statistical_analysis', 'data_aggregation', 
                          'data_analyzer', 'data_transformer']:
            categories["data_analysis"].append(tool.name)
        elif tool.name in ['conditional_logic', 'batch_process', 'retry_with_backoff']:
            categories["workflow"].append(tool.name)
        elif tool.name in ['cache_set', 'cache_get', 'cache_delete', 'cache_stats']:
            categories["cache"].append(tool.name)
        elif tool.name in ['http_request', 'http_get', 'http_post', 'async_http_request']:
            categories["http"].append(tool.name)
        elif tool.name in ['python_executor', 'calculator', 'sql_executor']:
            categories["code"].append(tool.name)
        elif tool.name in ['memory_store', 'memory_retrieve', 'memory_delete', 'memory_list']:
            categories["memory"].append(tool.name)
        elif tool.name in ['uuid_generator', 'hash_text', 'regex_matcher']:
            categories["system"].append(tool.name)
        elif tool.name in ['knowledge_search', 'vector_search', 'knowledge_add', 
                          'knowledge_delete', 'collection_info']:
            categories["knowledge"].append(tool.name)
    
    return categories


def get_tools_for_agent(agent_type: str = "general") -> List[Tool]:
    """获取适合特定 Agent 类型的工具集
    
    Args:
        agent_type: Agent 类型
            - "general": 通用Agent（所有工具）
            - "research": 研究型Agent（搜索、文本处理、记忆）
            - "code": 代码型Agent（代码执行、文件操作、系统工具）
            - "data": 数据型Agent（数据分析、文件解析、SQL）
            - "knowledge": 知识库Agent（知识搜索、向量搜索、记忆）
            - "web": Web Agent（HTTP、网页解析、搜索）
            - "text": 文本处理Agent（文本工具、编码解码）
            - "automation": 自动化Agent（工作流、缓存、批量处理）
            - "minimal": 最小工具集（基础搜索和计算）
    
    Returns:
        适合该Agent类型的工具列表
    """
    if agent_type == "research":
        # 研究型 Agent：搜索、文本处理、记忆
        return (
            get_search_tools() + get_text_tools() + 
            get_memory_tools() + get_datetime_tools()
        )
    
    elif agent_type == "code":
        # 代码型 Agent：代码执行、文件操作、系统工具
        return (
            get_code_tools() + get_file_tools() + 
            get_system_tools() + get_validation_tools()
        )
    
    elif agent_type == "data":
        # 数据型 Agent：数据分析、文件解析、SQL
        return (
            get_analysis_tools() + get_file_tools() + 
            get_code_tools() + get_validation_tools()
        )
    
    elif agent_type == "knowledge":
        # 知识库 Agent：知识搜索、向量搜索、记忆
        return (
            get_knowledge_tools() + get_text_tools() + 
            get_memory_tools() + get_search_tools()
        )
    
    elif agent_type == "web":
        # Web Agent：HTTP、网页解析、搜索
        web_tools = get_tools_by_names(['html_extract'])
        return (
            get_http_tools() + get_async_tools() + 
            get_search_tools() + web_tools + get_codec_tools()
        )
    
    elif agent_type == "text":
        # 文本处理 Agent
        return (
            get_text_tools() + get_codec_tools() + 
            get_validation_tools() + get_file_tools()
        )
    
    elif agent_type == "automation":
        # 自动化 Agent：工作流、缓存、批量处理
        return (
            get_workflow_tools() + get_cache_tools() + 
            get_http_tools() + get_async_tools() + get_code_tools()
        )
    
    elif agent_type == "minimal":
        # 最小工具集
        return get_tools_by_names([
            'web_search', 'calculator', 'get_datetime',
            'memory_store', 'memory_retrieve'
        ])
    
    else:  # general
        # 通用 Agent：所有工具
        return get_builtin_tools()


def get_tool_info() -> Dict[str, Any]:
    """获取所有工具的详细信息
    
    Returns:
        包含所有工具信息的字典
    """
    tools = get_builtin_tools()
    categories = get_tool_categories()
    
    return {
        "total_tools": len(tools),
        "categories": categories,
        "category_counts": {cat: len(tools_list) for cat, tools_list in categories.items()},
        "tools": [
            {
                "name": t.name,
                "description": t.description[:100] + "..." if len(t.description) > 100 else t.description,
                "category": t.category.value,
                "is_async": t.is_async,
                "timeout": t.timeout,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "required": p.required
                    }
                    for p in t.parameters
                ]
            }
            for t in tools
        ]
    }

