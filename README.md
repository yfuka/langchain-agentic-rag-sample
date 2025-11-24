langchain-agentic-rag-sample
============================

LangChainの軽量サンプルです。OpenAI互換エンドポイントと埋め込みモデルを指定し、インメモリVector Storeに入れたダミー文書をRAGツールで検索するエージェントを動かします。`agentic_rag.py`を実行すると、ツール呼び出しを含む対話のトレースをそのままコンソールに出力します。

前提
- Python 3.13 以上
- OpenAI互換APIのエンドポイントとAPIキー（OpenRouterなど）
- `uv`

セットアップ
1. 依存関係をインストール: `uv sync`
2. 環境変数ファイルを作成: `.env.template` を `.env` にコピーし、以下を埋める
   - `OPENAI_ENDPOINT`: 互換APIのベースURL
   - `OPENAI_API_KEY`: APIキー
   - `OPENAI_MODEL`: チャットモデル名（例: `gpt-4o`, `openai/gpt-oss-20b`）
   - `OPENAI_EMBEDDING_MODEL`: 埋め込みモデル名（例: `openai/text-embedding-3-small`）

動作確認と実行
- 接続テスト: `python api_check.py` でモデルに簡単なメッセージを送信
- RAGエージェント実行: `python agentic_rag.py`
  - インメモリのダミー文書を類似検索し、ツール呼び出し→回答までのメッセージを整形表示します。
  - 質問内容は `agentic_rag.py` 内の `query` 変数を書き換えて試せます。

主なファイル
- `agentic_rag.py`: RAGツール（`retrieve_context`）付きLangChainエージェントとログ整形関数
- `api_check.py`: OpenAI互換APIへの単純なチャットリクエストで疎通確認
- `.env.template`: 必要な環境変数のサンプル
- `pyproject.toml`: 依存関係定義
