# Welfog AI Agent

AI-powered customer support and product search chatbot for Welfog.

## Setup (local)

```bash
copy .env.example .env
# Edit .env — set MySQL, OpenAI/Groq keys, keep Qdrant settings below
pip install -r requirements.txt
docker compose up -d
python scripts/reindex_all.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Deployment

Minimum path after `git clone` (MySQL must already be running, e.g. XAMPP):

```bash
cd ai-agent-for-welfog   # this repo root

copy .env.example .env
# Fill MYSQL_*, OPENAI_API_KEY (and other LLM keys). Leave Qdrant defaults.

docker compose up -d
# Starts container welfog-qdrant on ports 6333 (HTTP) and 6334 (gRPC)
# Data persists in Docker volume welfog_qdrant_data

pip install -r requirements.txt

# Optional: import knowledge table DDL (also auto-created on app start)
# mysql -u root -p welfog_ai < migrations/20260707_create_knowledge_documents.sql
# mysql -u root -p welfog_ai < migrations/20260707_create_knowledge_document_chunks.sql

# If knowledge_documents is empty, either use Admin UI to add docs, or:
# python migrations/20260707_import_knowledge_txt_to_mysql.py

python scripts/reindex_all.py
# Chunks + embeds all active MySQL knowledge docs into Qdrant (required once on a fresh machine)

python app.py
```

### What starts automatically

| Component | When |
|-----------|------|
| MySQL `chat_sessions` / `chats` | `python app.py` startup |
| MySQL `knowledge_documents` | `python app.py` startup |
| Qdrant collection + payload indexes | `python app.py` startup (`init_qdrant_on_startup`) if `QDRANT_ENABLED=1` |
| Knowledge vectors (embeddings) | **Not** on app start — run `python scripts/reindex_all.py` once (or after empty Qdrant volume) |
| Admin create/update/delete knowledge | Automatic re-chunk + re-embed + Qdrant upsert/delete (no extra command) |

### Required `.env` (Qdrant)

```env
QDRANT_ENABLED=1
QDRANT_URL=http://127.0.0.1:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=welfog_knowledge_chunks
QDRANT_VECTOR_SIZE=1536
QDRANT_DISTANCE=Cosine
QDRANT_TIMEOUT_SEC=5
KB_RETRIEVAL_BACKEND=qdrant
```

App connects to Qdrant at `QDRANT_URL` after `docker compose up -d`. Do not use a one-off `docker run` in production — use Compose so the volume and restart policy stay consistent.

### Docker Compose commands

```bash
docker compose up -d          # start Qdrant
docker compose ps             # status (container name: welfog-qdrant)
docker compose logs -f qdrant
docker compose down           # stop (keeps volume)
docker compose down -v        # stop + wipe vector data (re-run reindex_all after)
```

If an old one-off `docker run --name welfog-qdrant` container conflicts:

```bash
docker rm -f welfog-qdrant
docker compose up -d
python scripts/reindex_all.py   # only if the old volume/data was not reused
```

### MySQL

1. Create database `welfog_ai` (or set `MYSQL_DATABASE` in `.env`).
2. Chat tables are created automatically on app start.
3. Knowledge DDL is in `migrations/*.sql` and is also created by the app / reindex script.
4. Content must exist in `knowledge_documents` (Admin CRUD or import script) before RAG answers work.

### One-time indexing command

```bash
python scripts/reindex_all.py
```

Run this after first clone, after wiping the Qdrant volume, or when MySQL has docs but Qdrant is empty. Ongoing Admin edits reindex themselves.

## Deploy (server)

Upload this folder as-is. Set `.env` on the server, install dependencies, run `docker compose up -d`, run `python scripts/reindex_all.py` once, then `python app.py` (or gunicorn behind nginx).
