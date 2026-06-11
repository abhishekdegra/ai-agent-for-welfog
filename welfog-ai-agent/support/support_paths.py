import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
