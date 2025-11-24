import os
from typing import Iterable, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

docs = [
    Document(id="doc_001", page_content="「月面ピザ」は、2035年に月面基地で初めて作られた料理で、無重力でも具材が飛び散らない特殊なチーズが使われています。"),
    Document(id="doc_002", page_content="新種の動物「カメレオン猫」は、気分によって毛の色が変わる特性を持っており、嬉しいときはピンク色になります。"),
    Document(id="doc_003", page_content="「レインボー・エナジー」は、雨上がりの虹から抽出されるクリーンエネルギーで、1回の抽出で街全体の電力を1週間まかなえます。"),
    Document(id="doc_004", page_content="伝説の果物「ウタウメロン」は、熟すと美しいメロディのような音を発するため、収穫の時期が音でわかります。"),
    Document(id="doc_005", page_content="スポーツ競技「エア・スイミング」は、強力な風で体を浮かせながら空中で泳ぐ競技で、水に濡れることなく楽しめます。")
]

model = os.getenv("OPENAI_MODEL")
base_url = os.getenv("OPENAI_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model=model,               # 互換API側で定義されているモデル名
    base_url=base_url,   # OpenAI互換のエンドポイント
    api_key=api_key,               # そのサービスのAPIキー
)

embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
embeddings = OpenAIEmbeddings(model=embedding_model, base_url=base_url, api_key=api_key)
vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def format_messages(messages: Iterable) -> str:
    """
    Convert a sequence of LangChain message objects into a readable transcript.
    Handles HumanMessage, AIMessage, ToolMessage, SystemMessage, FunctionMessage.
    """
    def short(text: str, limit: int = 240) -> str:
        text = text or ""
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def role_name(msg) -> str:
        if isinstance(msg, HumanMessage):
            return "Human"
        if isinstance(msg, AIMessage):
            return "AI"
        if isinstance(msg, ToolMessage):
            return f"Tool[{msg.name or 'unknown'}]"
        if isinstance(msg, FunctionMessage):
            return f"Function[{msg.name}]"
        if isinstance(msg, SystemMessage):
            return "System"
        return msg.__class__.__name__

    lines: List[str] = []
    tool_labels = {}
    label_counter = 1

    def label_for_tool_call(call_id: str) -> str:
        nonlocal label_counter
        if call_id not in tool_labels:
            tool_labels[call_id] = f"TC{label_counter}"
            label_counter += 1
        return tool_labels[call_id]

    idx = 1
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)

        # If AI message only issues tool calls, skip showing AI content line.
        if isinstance(msg, AIMessage) and tool_calls:
            for call_idx, tc in enumerate(tool_calls, start=1):
                name = tc.get("name") or tc.get("type") or "tool_call"
                args = tc.get("args", {})
                call_id = tc.get("id") or tc.get("tool_call_id") or f"auto_{call_idx}"
                label = label_for_tool_call(call_id)
                lines.append(f"[{idx}] ToolCall {label} {name} args={args}")
                idx += 1
            continue

        prefix = f"[{idx}] {role_name(msg)}"
        content = short(getattr(msg, "content", ""))

        # Tool messages: show only results, not the original tool content line.
        if isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", None):
            label = label_for_tool_call(msg.tool_call_id)
            lines.append(f"{prefix} Result for {label}")

            artifacts = getattr(msg, "artifact", None)
            if artifacts and isinstance(artifacts, list):
                for doc in artifacts:
                    doc_id = getattr(doc, "id", "unknown")
                    doc_excerpt = short(getattr(doc, "page_content", ""))
                    lines.append(f"    • doc {doc_id}: {doc_excerpt}")
            else:
                # Fallback: show truncated tool content if no artifacts.
                if content:
                    lines.append(f"    {content}")

            idx += 1
            continue

        # Default rendering for other messages.
        lines.append(f"{prefix}: {content}")
        idx += 1

    return "\n".join(lines)


tools = [retrieve_context]

prompt = (
    "あなたは未知の情報についてRAGコンテキストにアクセスするツールを持っています。"
    "質問に対して、このツールを必ず利用して回答してください。"
)

agent = create_agent(llm, tools, system_prompt=prompt)

query = "カメレオン猫が嬉しいとき、毛の色は何色になりますか？"

response = agent.invoke({"messages": [{"role": "user", "content": query}]})
pretty = format_messages(response["messages"])
print(pretty)
# print(response["messages"])
