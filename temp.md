| 关键词 | 指代内容 | 结构形态 | 典型用例位置 |
|---|---|---|---|
| metadata | 请求级元数据容器，挂载本次请求的 files、tool_ids 等 | dict，例：{"files": [...], "tool_ids": [...], ...}
| process_chat_payload 里构建 form_data["metadata"]，随后 chat_completion_files_handler 通过 body["metadata"]["files"]
| files | 需要做 RAG 的“附件条目”列表 | list[dict]，条目常含 type、id、name、context、file、collection_name(s)、docs
| file | 这是 get_sources_from_items 在内部把原始 item 附回到 query_result 里，后续作为 source 的来源 | dict，即原始
files 条目 | get_sources_from_items 末尾 query_results.append({**query_result, "file": item})。backend/open_webui/
retrieval/utils.py |
| documents | 检索或直接抽取到的文档内容集合（原始结果） | list[list[str]]，外层通常只有一组 | get_sources_from_items
返回的 query_result 结构。backend/open_webui/retrieval/utils.py |
query_result 结构。backend/open_webui/retrieval/utils.py |
| metadata（在 source 内） | 已被“拍平”的元数据列表，和 document 一一对应 | list[dict]，来自 metadatas[0] |
get_sources_from_items 组装 sources 时："metadata": query_result["metadatas"][0]。backend/open_webui/retrieval/
utils.py |
query_result["file"]。工具来源：chat_completion_tools_handler 构建。backend/open_webui/retrieval/utils.py、backend/
open_webui/utils/middleware.py |
| sources | 最终用于上下文拼接和前端引用的来源列表 | list[dict]，元素结构：{"source": ..., "document": [...],
open_webui/utils/middleware.py |
backend/open_webui/retrieval/utils.py |
| context | 是否要求“全量上下文” | str，常见值 full | get_sources_from_items 决定走“全文直接注入”还是“检索”。backend/
open_webui/retrieval/utils.py |
| docs | 直接携带文档内容的旁路输入 | list[dict]，每个 doc 有 content 和 metadata | get_sources_from_items 在
item.get("docs") 分支直接构建 query_result。backend/open_webui/retrieval/utils.py |
| collection_name(s) | 指定向量库集合名 | str 或 list[str] | 走向量检索路径时使用。backend/open_webui/retrieval/
utils.py |

这些关键词在 RAG 中如何被使用并插入上下文（关键链路）

process_chat_payload 把 files 合并进 form_data["metadata"]，之后调用 chat_completion_files_handler。backend/
open_webui/utils/middleware.py
chat_completion_files_handler 把 metadata.files 传给 get_sources_from_items，得到 sources。backend/open_webui/
utils/middleware.py
get_sources_from_items 产出 query_result（documents/metadatas），再组装为 sources 列表（document/metadata/
source）。backend/open_webui/retrieval/utils.py
process_chat_payload 遍历 sources，对每个 document + metadata 生成 文档文本，拼
成 context_string。backend/open_webui/utils/middleware.py
用 rag_template 把 context_string 和用户原始 prompt 合并为新的用户消息，再写回 messages。backend/open_webui/utils/
middleware.py、backend/open_webui/utils/task.py