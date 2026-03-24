from __future__ import annotations

import base64
import html
import json
import math
import os
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF


def _normalize_bbox(page: fitz.Page, bbox: List[float]) -> fitz.Rect:
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 elements")
    x0, y0, x1, y1 = bbox
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)

    if x1 <= 1 and y1 <= 1 and x0 >= 0 and y0 >= 0:
        x0 *= page_width
        x1 *= page_width
        y0 *= page_height
        y1 *= page_height

    rect_w = abs(x1 - x0)
    rect_h = abs(y1 - y0)
    
    #以下内容处理页面旋转
    rotation = int(page.rotation or 0) % 360
    actual_width = page_width
    actual_height = page_height
    if rotation in (90, 270):
        actual_width, actual_height = actual_height, actual_width

    if rotation == 270:
        rect_w, rect_h = rect_h, rect_w
        x0 = actual_height - y1
        y0 = actual_width - x1
    elif rotation == 180:
        x0 = page_width - x1
    elif rotation == 90:
        rect_w, rect_h = rect_h, rect_w
        x0, y0 = y0, x0
    else:
        y0 = page_height - y1

    y0 = actual_height - (y0 + rect_h)
    return fitz.Rect(x0, y0, x0 + rect_w, y0 + rect_h)

def _fit_font_size(text: str, rect: fitz.Rect, fontname: str = "helv") -> float:
    if not text.strip():
        return 1
    base_len = fitz.get_text_length(text, fontname=fontname, fontsize=1)
    if base_len <= 0:
        return 1

    min_size = 4.0
    max_size = max(min(rect.height, rect.width), min_size)
    line_height_ratio = 1.15

    size_w = rect.width / base_len
    est_lines = max(1, math.ceil((base_len * size_w) / max(rect.width, 1)))
    size_h = rect.height / max(est_lines * line_height_ratio, 1)

    size = min(size_w, size_h, max_size)
    return max(min_size, size * 0.95)

def pdf_bytes_to_base64(pdf_bytes: bytes) -> str:
    return base64.b64encode(pdf_bytes).decode("ascii")

def load_middle_ocr_json(ocr_json_path: str) -> List[List[Dict[str, Any]]]:
    if not os.path.exists(ocr_json_path):
        return []
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if not isinstance(data, dict) or "pdf_info" not in data:
        return []

    pages_info = data.get("pdf_info", [])
    max_index = max((p.get("page_idx", idx) for idx, p in enumerate(pages_info)), default=-1)
    pages: List[List[Dict[str, Any]]] = [[] for _ in range(max_index + 1)]

    for idx, page_info in enumerate(pages_info):#遍历页
        page_idx = page_info.get("page_idx", idx)
        page_size = page_info.get("page_size") or [1, 1]
        page_width = float(page_size[0] or 1)
        page_height = float(page_size[1] or 1)
        items: List[Dict[str, Any]] = []

        for block in page_info.get("para_blocks", []) + page_info.get("discarded_blocks", []):#遍历页内所有paragraph
            spans = block.get("lines", [])[0].get("spans", []) if block.get("lines") else []
            seen = {}  # bbox(tuple) -> index in items
            for item in spans:
                bbox = item.get("bbox")
                content = (item.get("content") or "").strip()
                block_type = item.get("type")
                image_path = item.get("image_path", "")

                # bbox 可能是 list，需要转成可 hash 的 tuple
                bbox_key = tuple(bbox) if bbox is not None else None

                if bbox_key is None:
                    # 没 bbox：直接 append
                    items.append({
                        "bbox": bbox,
                        "content": content,
                        "type": block_type,
                        "image_path": image_path,
                    })
                    continue

                if bbox_key not in seen:
                    # 第一次出现：建立合并对象
                    seen[bbox_key] = len(items)
                    items.append({
                        "bbox": list(bbox_key),          # 保持输出为 list
                        "content": content,
                        "type": block_type,              # ✅ 用第一个块的
                        "image_path": image_path,         # ✅ 用第一个块的
                    })
                else:
                    # 重复 bbox：合并 content（空格分割），其余字段不动
                    idx = seen[bbox_key]
                    if content:
                        if items[idx]["content"]:
                            items[idx]["content"] += " " + content
                        else:
                            items[idx]["content"] = content

        if 0 <= page_idx < len(pages):
            pages[page_idx] = items

    return pages

#content json 太粗粒度了，不能用
def load_content_ocr_json(ocr_json_path: str) -> List[List[Dict[str, Any]]]:
    if not os.path.exists(ocr_json_path):
        return []
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    max_index = max((p.get("page_idx", -1) for p in data), default=-1)
    pages: List[List[Dict[str, Any]]] = [[] for _ in range(max_index + 1)]
    print(data[:5])
    for content_meta in data:#遍历页
        page_idx = content_meta.get("page_idx", -1)
        block_bbox = content_meta.get("bbox")
        block_type = content_meta.get("type")
        block_content = content_meta.get("text", "")
        pages[page_idx].append([v for v in block_bbox])
    return pages

# depracated
def render_ocr_on_pdf(
    pdf_path: str,
    ocr_json_path: str,
    output_path: Optional[str] = None,
) -> bytes:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"pdf not found: {pdf_path}")
    if not os.path.exists(ocr_json_path):
        raise FileNotFoundError(f"ocr json not found: {ocr_json_path}")

    pages_info = load_middle_ocr_json(ocr_json_path)
    with fitz.open(pdf_path) as doc:
        for page_index, page_info in enumerate(pages_info):
            page = doc[page_index]#取出对应页面
            for content in page_info:
                try:    
                    rect = _normalize_bbox(page, content["bbox"])
                except ValueError:
                    continue
                if content.get("content") is None:
                    continue
                page.draw_rect(
                    rect,
                    color=(1, 0, 0),
                    fill=(1, 0, 0),
                    fill_opacity=0.2,
                )
                font_size = _fit_font_size(content["content"], rect)
                # page.insert_textbox(
                #     rect,
                #     content['content'],
                #     fontsize=10,
                #     rotate=0,
                #     color=(0, 0.5, 0),
                #     render_mode=0,   # 0 = fill text（可见）
                #     overlay=True,    # 画在原内容之上
                # )
                rc = page.insert_textbox(rect, content["content"], fontsize=8, color=(0,0.5,0), overlay=True)
                if rc < 0:
                    print("textbox overflow:", rc, "rect:", rect, "text:", content["content"][:50])
                print('rendered:', content['content'])
        pdf_bytes = doc.tobytes()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)
    return pdf_bytes



# md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
#         md_writer.write_string(
#             f"{pdf_file_name}.md",
#             md_content_str,
#         )
if __name__ == "__main__":
    ocr_content_path = "./backend/data/sample.json"
    pdf_path = "./backend/data/sample.pdf"
    render_ocr_on_pdf(pdf_path, ocr_content_path, "./backend/data/output.pdf")
