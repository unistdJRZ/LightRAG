"""
Shared LLM helper utilities.
"""
import asyncio
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set

import fitz  # PyMuPDF

from backend.mineru_config_reader import get_formula_enable, get_table_enable
from backend.mineru_enum_class import BlockType, MakeMode
from backend.mineru_utils import mk_blocks_to_markdown
from backend.PDF_factory import _fit_font_size, _normalize_bbox
from backend.cachelib import async_cached


try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import markdown as md  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    md = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from playwright.async_api import async_playwright  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    async_playwright = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sync_playwright = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from weasyprint import CSS, HTML  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CSS = None  # type: ignore
    HTML = None  # type: ignore


def get_llm_client() -> "OpenAI":
    """
    Lazily construct an OpenAI (DeepSeek) client using environment settings.
    """
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_key = 'sk-efa599ed81c9492580cb4849d5bad62f'#防止univcorn无法获取key，测试用
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")

    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


class LLMRequest:
    """
    Thread-backed async LLM requester.

    Provides an async `request` method that runs the blocking LLM call in a thread
    so multiple requests can execute concurrently. A blocking helper is also
    provided for backward compatibility with sync callers.
    """

    def __init__(self, max_workers: int | None = None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        client = get_llm_client()
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content if resp.choices else ""

    async def request(self, system_prompt: str, user_prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor, self._call, system_prompt, user_prompt
        )

# Shared requester for the module
llm_requester = LLMRequest(max_workers=20)


async def translate_to_zh(content: str) -> str:
    """
    Call LLM to translate content.
    """
    if not content:
        return ""

    system_content = (
        "你是一个专业的学术研究助手，你需要将一段专业性文本翻译为中文。"
        "翻译内容应准确传达原文的专业术语和含义，同时确保中文表达流畅自然。"
        "如果内容格式为Markdown，如果内容中包含公式，文件路径等信息，无需处理这些内容。"
        "[OUTPUT FORMAT] 仅返回翻译后的Markdown文本，不要包含任何多余信息。"
    )

    user_content = "请翻译以下内容：\n\n"
    user_content += content

    try:
        translated = await llm_requester.request(system_content, user_content)
    except Exception:
        translated = ""

    return translated

@async_cached(
    ttl=float(os.getenv("SUMMARY_CACHE_TTL", "86400")),
    version="2",
    file_cache_dir=os.getenv("CACHE_DIR", ".cache"),
)
async def do_summary_zh(content: str) -> str:
    """
    Call LLM to summarize content.
    """
    if not content:
        return ""

    system_content = """
    ### Role Definition

    你是一位专业的学术导读助手，拥有深厚的物理学、材料科学及工程学背景。你的任务是阅读用户上传的专业文献（通常为Markdown格式），并生成一份结构清晰、深度且易于理解的**AI导读报告**。

    ### Task Objective

    你需要深入理解文献的核心逻辑、物理机制、实验数据和结论，忽略非核心的格式符号，输出一份包含以下板块的导读：

    1. **全文概述**：系统性总结文章核心内容。
    2. **名词解释**：解释文中的专业术语。
    3. **AI大纲**：梳理文章的逻辑结构。
    4. **关键要点**：提炼文章的核心价值与结论。
    5. **关键图表解读**：说明文中图表的含义（如有）。
    6. **相关组织/人物**：提取文中提到的重要实体。
    7. **关键问题与回答**：基于原文生成深度问答。

    ### Output Guidelines

    请严格按照以下步骤和格式生成输出：

    #### 1. 全文概述 (Executive Summary)

    * **字数**：300-500字。
    * **内容**：不要仅仅是简单的摘要。你需要逻辑严密地串联起文章的研究背景、核心理论（如物理机制）、关键方法（如优化策略）、主要发现（如特定材料性能）以及最终结论。
    * **风格**：学术、客观、凝练。

    #### 2. 名词解释 (Terminology)

    * **筛选**：选取文中出现的核心专业术语（3-5个），特别是缩写或特定领域的概念（例如：Seebeck系数, ZT值, 声子玻璃等）。
    * **解释**：基于原文上下文进行解释，说明其物理意义或在文中的作用。

    #### 3. AI大纲 (AI Outline)

    * **结构**：使用Markdown的多级列表。
    * **内容**：重构文章的逻辑骨架。不要只列标题，要简要概括该层级下的核心讨论点。

    #### 4. 关键要点 (Key Takeaways)

    * **形式**：3-5个Bullet points。
    * **内容**：提炼文章的“干货”。例如：性能优化的具体三要素、某个理论的具体预测结果、不同材料体系的对比结论等。

    #### 5. 关键图表解读 (Key Chart Interpretation)

    * **触发条件**：如果文中提及了图表（Figure X），请执行此步骤。
    * **内容**：列出图表编号，并一句话概括该图表揭示了什么科学事实或数据趋势（例如：“该图验证了...”，“揭示了...关系”）。

    #### 6. 相关组织与人物 (Related Entities)

    * **分类**：分为“相关组织”和“相关人物”。
    * **内容**：提取文中提及的研究机构（如NASA, DOE）和关键科学家，并简述他们与本文内容的关联或贡献。

    #### 7. 关键问题与回答 (Key Q&A)

    * **数量**：3个深度问题。
    * **设计**：提出“为什么 (Why)”或“如何 (How)”类型的核心问题，这些问题应当是读者阅读本文最想搞懂的难点。
    * **回答**：基于原文内容给出精准、概括性的回答。

    ---

    ### Response Format Example

    (请严格遵守此Markdown格式输出)

    # 全文概述

    [在此处撰写全文概述内容...]

    # 名词解释

    **[术语1]**：[解释内容]
    **[术语2]**：[解释内容]

    # AI大纲

    * [一级标题]
    * [二级论点]
    * [二级论点]



    # 关键要点

    * **[要点标题]**：[详细描述]
    * **[要点标题]**：[详细描述]

    # 关键图表解读

    * **图1**：[解读内容]
    * **图2**：[解读内容]

    # 相关组织与人物

    * **相关组织**：
    * [组织名]：[简介]


    * **相关人物**：
    * [人名]：[简介]



    # 关键问题与回答

    **Q1：[问题描述]？**
    A：[答案内容]

    **Q2：[问题描述]？**
    A：[答案内容]

    ---

    ### Constraints

    * **语言**：除非用户另有要求，否则请使用**中文**进行导读（保留专有名词的英文原名）。
    * **真实性**：所有解释和结论必须基于提供的文本内容，严禁编造文中不存在的数据或理论。
    * **公式**：如果涉及核心物理公式（如ZT值计算），请使用LaTeX格式展示。
    """

    user_content = "请翻译以下内容：\n\n"
    user_content += content

    try:
        translated = await llm_requester.request(system_content, user_content)
    except Exception:
        translated = ""

    return translated

class MetaLoader:
    """
    Load OCR meta data and provide translated variants.
    """

    def __init__(self) -> None:
        self.blocks: List[Dict[str, object]] = []
        self.markdown = ""
        self.markdown_zh = ""
        self._loaded = False
        self._translated = False
        self._katex_cache: Optional[Dict[str, str]] = None

    def load(self) -> None:
        ocr_path = os.getenv("OCR_JSON_PATH", "./backend/data/sample.json")
        md_data_url = os.getenv("MD_DATA_URL", "/api/images").rstrip("/")

        if not os.path.exists(ocr_path):
            raise FileNotFoundError("ocr json not found")

        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "pdf_info" not in data:
            self.blocks = []
            self.markdown = ""
            self._loaded = True
            self._translated = False
            return

        formula_enable = get_formula_enable(
            os.getenv("MINERU_VLM_FORMULA_ENABLE", "True").lower() == "true"
        )
        table_enable = get_table_enable(
            os.getenv("MINERU_VLM_TABLE_ENABLE", "True").lower() == "true"
        )

        blocks: List[Dict[str, object]] = []
        markdown_parts: List[str] = []
        for page_idx, page_info in enumerate(data.get("pdf_info", [])):
            para_blocks = page_info.get("para_blocks") or []
            page_size = page_info.get("page_size") or []
            page_index = page_info.get("page_idx", page_idx)
            for para_block in para_blocks:
                md_list = mk_blocks_to_markdown(
                    [para_block],
                    MakeMode.MM_MD,
                    formula_enable,
                    table_enable,
                    img_buket_path=md_data_url,
                )
                if not md_list:
                    continue
                md = md_list[0]
                block_id = len(blocks)
                blocks.append(
                    {
                        "block_id": block_id,
                        "page_idx": page_index,
                        "page_size": page_size,
                        "bbox": para_block.get("bbox", []),
                        "markdown": md,
                        "content_type": para_block.get("type"),
                    }
                )
                markdown_parts.append(md)

        self.blocks = blocks
        self.markdown = "\n\n".join(markdown_parts)
        self._loaded = True
        self._translated = False

    async def translate_zh(self) -> None:
        if not self._loaded:
            self.load()

        non_text_types: Set[str] = {
            BlockType.IMAGE,
            BlockType.TABLE,
            BlockType.CODE,
            BlockType.IMAGE_BODY,
            BlockType.TABLE_BODY,
            BlockType.CODE_BODY,
            BlockType.IMAGE_CAPTION,
            BlockType.TABLE_CAPTION,
            BlockType.CODE_CAPTION,
            BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_FOOTNOTE,
            BlockType.INTERLINE_EQUATION,
            BlockType.ALGORITHM,
            BlockType.REF_TEXT,
            BlockType.PHONETIC,
            BlockType.HEADER,
            BlockType.FOOTER,
            BlockType.PAGE_NUMBER,
            BlockType.ASIDE_TEXT,
            BlockType.PAGE_FOOTNOTE,
        }

        translated_parts: List[str] = []
        translate_tasks: List[asyncio.Task[str]] = []
        task_indices: List[int] = []

        for index, block in enumerate(self.blocks):
            content_type = str(block.get("content_type") or "")
            markdown = str(block.get("markdown") or "")
            if content_type in non_text_types:
                block["markdown_zh"] = markdown
                translated_parts.append(markdown)
            else:
                task_indices.append(index)
                translate_tasks.append(asyncio.create_task(translate_to_zh(markdown)))
                translated_parts.append("")

        if translate_tasks:
            results = await asyncio.gather(*translate_tasks, return_exceptions=True)
            for index, result in zip(task_indices, results):
                markdown = str(self.blocks[index].get("markdown") or "")
                if isinstance(result, Exception) or not result:
                    translated = markdown
                else:
                    translated = result
                self.blocks[index]["markdown_zh"] = translated
                translated_parts[index] = translated

        self.markdown_zh = "\n\n".join(translated_parts)
        self._translated = True

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    async def ensure_translated(self) -> None:
        if not self._translated:
            await self.translate_zh()

    def _markdown_to_text(self, content: str) -> str:
        if not content:
            return ""
        cleaned = re.sub(r"<!--[\s\S]*?-->", "", content)
        cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
        cleaned = re.sub(r"`{3}[\s\S]*?`{3}", "", cleaned)
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\[(.*?)\]\([^)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _render_markdown_image(
        self, content: str, width_pt: float, height_pt: float
    ) -> Optional[bytes]:
        if not content or width_pt <= 0 or height_pt <= 0:
            return None
        if md is None or HTML is None or CSS is None:
            return None

        sanitized = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
        extensions = ["fenced_code", "tables"]
        try:
            html_body = md.markdown(sanitized, extensions=extensions)
        except Exception:
            return None

        #将markdown转换为纯文本，用于计算合适的字体大小
        plain_text = self._markdown_to_text(content)
        fit_size = 0.0
        if plain_text:
            fit_size = _fit_font_size(plain_text, fitz.Rect(0, 0, width_pt, height_pt))
        base_font_size = max(4.0, min(12.0, height_pt / 3.6))
        if fit_size:
            base_font_size = min(base_font_size, fit_size)
        padding_x = max(2.0, min(6.0, width_pt * 0.02))
        padding_y = max(2.0, min(6.0, height_pt * 0.02))
        html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
  </head>
  <body>
    <div id="content">{html_body}</div>
  </body>
</html>
"""
        css = f"""
@page {{ size: {width_pt}pt {height_pt}pt; margin: 0; }}
html, body {{
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  background: #ffffff;
  color: #111111;
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimSun", sans-serif;
  font-size: {base_font_size}pt;
  line-height: 1.45;
}}
#content {{
  box-sizing: border-box;
  padding: {padding_y}pt {padding_x}pt;
}}
h1, h2, h3, h4, h5, h6 {{
  margin: 0 0 4pt 0;
  font-weight: 600;
}}
h1 {{ font-size: 1.2em; }}
h2 {{ font-size: 1.15em; }}
h3 {{ font-size: 1.1em; }}
h4 {{ font-size: 1.05em; }}
h5 {{ font-size: 1.0em; }}
h6 {{ font-size: 0.95em; }}
p {{ margin: 0 0 4pt 0; }}
ul, ol {{ margin: 0 0 4pt 12pt; padding: 0; }}
pre {{
  background: #f5f5f5;
  padding: 4pt 6pt;
  border-radius: 4pt;
  white-space: pre-wrap;
}}
code {{
  font-family: "Consolas", "Courier New", monospace;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 9pt;
}}
td, th {{
  border: 0.5pt solid #d0d0d0;
  padding: 2pt 4pt;
  vertical-align: top;
}}
"""
        try:
            pdf_bytes = HTML(string=html).write_pdf(stylesheets=[CSS(string=css)])
        except Exception:
            return None

        try:
            with fitz.open("pdf", pdf_bytes) as tmp_doc:
                page = tmp_doc[0]
                pixmap = page.get_pixmap(dpi=144, alpha=False)
                return pixmap.tobytes("png")#最终转换为图像返回
        except Exception:
            return None

    def _load_katex_assets(self) -> Dict[str, str]:
        if self._katex_cache is not None:
            return self._katex_cache
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        katex_root = os.path.join(root, "node_modules", "katex", "dist")
        assets = {"css": "", "katex_js": "", "auto_js": ""}
        try:
            with open(os.path.join(katex_root, "katex.min.css"), "r", encoding="utf-8") as f:
                assets["css"] = f.read()
        except Exception:
            assets["css"] = ""
        try:
            with open(os.path.join(katex_root, "katex.min.js"), "r", encoding="utf-8") as f:
                assets["katex_js"] = f.read()
        except Exception:
            assets["katex_js"] = ""
        try:
            with open(
                os.path.join(katex_root, "contrib", "auto-render.min.js"),
                "r",
                encoding="utf-8",
            ) as f:
                assets["auto_js"] = f.read()
        except Exception:
            assets["auto_js"] = ""
        self._katex_cache = assets
        return assets

    def _build_html_assets(
        self,
        html_body: str,
        width_pt: float,
        height_pt: float,
        fit_size: float,
        include_katex: bool,
    ) -> tuple[str, str]:
        base_font_size = max(4.0, min(12.0, height_pt / 3.6))
        if fit_size:
            base_font_size = min(base_font_size, fit_size)
        padding_x = max(2.0, min(6.0, width_pt * 0.02))
        padding_y = max(2.0, min(6.0, height_pt * 0.02))
        katex_assets = self._load_katex_assets() if include_katex else {"css": "", "katex_js": "", "auto_js": ""}
        katex_css = katex_assets.get("css", "")
        katex_js = katex_assets.get("katex_js", "")
        auto_js = katex_assets.get("auto_js", "")
        html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>{katex_css}</style>
  </head>
  <body>
    <div id="content">{html_body}</div>
    <script>{katex_js}</script>
    <script>{auto_js}</script>
    <script>
      try {{
        if (typeof renderMathInElement === "function") {{
          renderMathInElement(document.body, {{
            delimiters: [
              {{ left: "$$", right: "$$", display: true }},
              {{ left: "$", right: "$", display: false }},
            ],
            throwOnError: false
          }});
        }}
      }} finally {{
        window.__katex_done = true;
      }}
    </script>
  </body>
</html>
"""
        css = f"""
@page {{ size: {width_pt}pt {height_pt}pt; margin: 0; }}
html, body {{
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  background: #ffffff;
  color: #111111;
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimSun", sans-serif;
  font-size: {base_font_size}pt;
  line-height: 1.45;
  overflow: hidden;
}}
#content {{
  box-sizing: border-box;
  padding: {padding_y}pt {padding_x}pt;
  width: 100%;
  height: 100%;
  word-break: break-word;
  overflow-wrap: anywhere;
}}
h1, h2, h3, h4, h5, h6 {{
  margin: 0 0 4pt 0;
  font-weight: 600;
}}
h1 {{ font-size: 1.2em; }}
h2 {{ font-size: 1.15em; }}
h3 {{ font-size: 1.1em; }}
h4 {{ font-size: 1.05em; }}
h5 {{ font-size: 1.0em; }}
h6 {{ font-size: 0.95em; }}
p {{ margin: 0 0 4pt 0; }}
ul, ol {{ margin: 0 0 4pt 12pt; padding: 0; }}
pre {{
  background: #f5f5f5;
  padding: 4pt 6pt;
  border-radius: 4pt;
  white-space: pre-wrap;
}}
code {{
  font-family: "Consolas", "Courier New", monospace;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 9pt;
}}
td, th {{
  border: 0.5pt solid #d0d0d0;
  padding: 2pt 4pt;
  vertical-align: top;
}}
"""
        return html, css

    async def _render_markdown_image_playwright(
        self,
        page: object,
        content: str,
        width_pt: float,
        height_pt: float,
    ) -> Optional[bytes]:
        if not content or width_pt <= 0 or height_pt <= 0:
            return None
        if md is None:
            return None

        sanitized = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
        extensions = ["fenced_code", "tables"]
        try:
            html_body = md.markdown(sanitized, extensions=extensions)
        except Exception:
            return None

        plain_text = self._markdown_to_text(content)
        fit_size = 0.0
        if plain_text:
            fit_size = _fit_font_size(plain_text, fitz.Rect(0, 0, width_pt, height_pt))

        html, css = self._build_html_assets(
            html_body,
            width_pt,
            height_pt,
            fit_size,
            include_katex=True,
        )
        if "</head>" in html:
            html = html.replace("</head>", f"<style>{css}</style></head>")
        else:
            html = f"<style>{css}</style>{html}"

        px_scale = 96 / 72
        width_px = max(1, int(width_pt * px_scale))
        height_px = max(1, int(height_pt * px_scale))
        try:
            await page.set_viewport_size({"width": width_px, "height": height_px})
            await page.set_content(
                html,
                wait_until="load",
            )
            try:
                await page.wait_for_function("window.__katex_done === true", timeout=1500)
            except Exception:
                pass
            try:
                await page.evaluate("() => document.fonts && document.fonts.ready")
            except Exception:
                pass
            await page.evaluate(
                """
() => {
  const container = document.getElementById('content');
  if (!container) return 1;
  const maxH = window.innerHeight;
  const maxW = window.innerWidth;
  const rect = container.getBoundingClientRect();
  const height = Math.max(container.scrollHeight, rect.height);
  const width = Math.max(container.scrollWidth, rect.width);
  let scale = Math.min(maxH / height, maxW / width);
  if (!Number.isFinite(scale) || scale <= 0) scale = 1;
  container.style.transformOrigin = 'top left';
  container.style.transform = `scale(${scale})`;
  container.style.width = `${maxW / scale}px`;
  container.style.height = `${maxH / scale}px`;
  document.body.style.height = `${maxH}px`;
  document.body.style.width = `${maxW}px`;
  return scale;
}
"""
            )
            await page.wait_for_timeout(30)
            return await page.screenshot(type="png", full_page=False)
        except Exception:
            return None

    def _render_markdown_image_playwright_sync(
        self,
        page: object,
        content: str,
        width_pt: float,
        height_pt: float,
    ) -> Optional[bytes]:
        if not content or width_pt <= 0 or height_pt <= 0:
            return None
        if md is None:
            return None

        sanitized = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", content)
        extensions = ["fenced_code", "tables"]
        try:
            html_body = md.markdown(sanitized, extensions=extensions)
        except Exception:
            return None

        plain_text = self._markdown_to_text(content)
        fit_size = 0.0
        if plain_text:
            fit_size = _fit_font_size(plain_text, fitz.Rect(0, 0, width_pt, height_pt))

        html, css = self._build_html_assets(
            html_body,
            width_pt,
            height_pt,
            fit_size,
            include_katex=True,
        )
        if "</head>" in html:
            html = html.replace("</head>", f"<style>{css}</style></head>")
        else:
            html = f"<style>{css}</style>{html}"

        px_scale = 96 / 72
        width_px = max(1, int(width_pt * px_scale))
        height_px = max(1, int(height_pt * px_scale))
        try:
            page.set_viewport_size({"width": width_px, "height": height_px})
            page.set_content(html, wait_until="load")
            try:
                page.wait_for_function("window.__katex_done === true", timeout=1500)
            except Exception:
                pass
            try:
                page.evaluate("() => document.fonts && document.fonts.ready")
            except Exception:
                pass
            page.evaluate(
                """
() => {
  const container = document.getElementById('content');
  if (!container) return 1;
  const maxH = window.innerHeight;
  const maxW = window.innerWidth;
  const rect = container.getBoundingClientRect();
  const height = Math.max(container.scrollHeight, rect.height);
  const width = Math.max(container.scrollWidth, rect.width);
  let scale = Math.min(maxH / height, maxW / width);
  if (!Number.isFinite(scale) || scale <= 0) scale = 1;
  container.style.transformOrigin = 'top left';
  container.style.transform = `scale(${scale})`;
  container.style.width = `${maxW / scale}px`;
  container.style.height = `${maxH / scale}px`;
  document.body.style.height = `${maxH}px`;
  document.body.style.width = `${maxW}px`;
  return scale;
}
"""
            )
            page.wait_for_timeout(30)
            return page.screenshot(type="png", full_page=False)
        except Exception:
            return None

    def _render_images_sync_playwright(
        self, render_requests: List[tuple[int, str, float, float]]
    ) -> Dict[int, bytes]:
        results: Dict[int, bytes] = {}
        if sync_playwright is None:
            return results
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(device_scale_factor=2)
            page = context.new_page()
            for index, markdown, width_pt, height_pt in render_requests:
                image_bytes = self._render_markdown_image_playwright_sync(
                    page, markdown, width_pt, height_pt
                )
                if image_bytes:
                    results[index] = image_bytes
            page.close()
            context.close()
            browser.close()
        return results

    async def render_translation_pdf(self, pdf_path: str) -> bytes:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("pdf not found")
        await self.ensure_translated()

        non_text_types: Set[str] = {
            BlockType.IMAGE,
            BlockType.TABLE,
            BlockType.CODE,
            BlockType.IMAGE_BODY,
            BlockType.TABLE_BODY,
            BlockType.CODE_BODY,
            BlockType.IMAGE_CAPTION,
            BlockType.TABLE_CAPTION,
            BlockType.CODE_CAPTION,
            BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_FOOTNOTE,
            BlockType.INTERLINE_EQUATION,
            BlockType.ALGORITHM,
            BlockType.REF_TEXT,
            BlockType.PHONETIC,
            BlockType.HEADER,
            BlockType.FOOTER,
            BlockType.PAGE_NUMBER,
            BlockType.ASIDE_TEXT,
            BlockType.PAGE_FOOTNOTE,
        }

        with fitz.open(pdf_path) as doc:
            render_requests: List[tuple[int, str, float, float]] = []
            rects: Dict[int, fitz.Rect] = {}
            for index, block in enumerate(self.blocks):
                content_type = str(block.get("content_type") or "")
                if content_type in non_text_types:#跳过没被翻译的部分的渲染工作
                    continue
                bbox = block.get("bbox") or []
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                page_index = int(block.get("page_idx") or 0)
                if page_index < 0 or page_index >= len(doc):
                    continue
                page = doc[page_index]
                try:
                    rect = _normalize_bbox(page, bbox)
                except ValueError:
                    continue
                markdown = str(block.get("markdown_zh") or block.get("markdown") or "")
                rects[index] = rect
                render_requests.append((index, markdown, rect.width, rect.height))#收集需要渲染的的内容

            images: Dict[int, bytes] = {}
            loop = asyncio.get_running_loop()
            proactor_type = getattr(asyncio, "ProactorEventLoop", None)
            supports_subprocess = True
            if proactor_type is not None:
                supports_subprocess = isinstance(loop, proactor_type)

            if async_playwright is not None and supports_subprocess:
                try:
                    async with async_playwright() as p:
                        browser = await p.chromium.launch()
                        context = await browser.new_context(device_scale_factor=2)
                        page_handle = await context.new_page()
                        for index, markdown, width_pt, height_pt in render_requests:
                            image_bytes = await self._render_markdown_image_playwright(
                                page_handle, markdown, width_pt, height_pt
                            )
                            if image_bytes:
                                images[index] = image_bytes
                        await page_handle.close()
                        await context.close()
                        await browser.close()
                except Exception:
                    images = await asyncio.to_thread(
                        self._render_images_sync_playwright, render_requests
                    )
            elif sync_playwright is not None:
                images = await asyncio.to_thread(
                    self._render_images_sync_playwright, render_requests
                )

            for index, rect in rects.items():
                page_index = int(self.blocks[index].get("page_idx") or 0)
                if page_index < 0 or page_index >= len(doc):
                    continue
                page = doc[page_index]
                image_bytes = images.get(index)
                if image_bytes:
                    page.draw_rect(
                        rect,
                        color=(1, 1, 1),
                        fill=(1, 1, 1),
                        fill_opacity=1.0,
                        overlay=True,
                    )
                    page.insert_image(rect, stream=image_bytes, keep_proportion=False)
                    continue
                markdown = str(self.blocks[index].get("markdown_zh") or self.blocks[index].get("markdown") or "")
                image_bytes = self._render_markdown_image(
                    markdown, rect.width, rect.height
                )
                if image_bytes:
                    page.draw_rect(
                        rect,
                        color=(1, 1, 1),
                        fill=(1, 1, 1),
                        fill_opacity=1.0,
                        overlay=True,
                    )
                    page.insert_image(rect, stream=image_bytes, keep_proportion=False)

            pdf_bytes = doc.tobytes()

        return pdf_bytes


if __name__ == "__main__":
    async def _demo() -> None:
        sample_text = "This is a sample text to be translated into Chinese."
        translated_text = await translate_to_zh(sample_text)
        print("Translated Text:\n", translated_text)

    asyncio.run(_demo())
