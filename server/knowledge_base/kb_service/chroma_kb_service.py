import os
import shutil

from configs import SCORE_THRESHOLD
from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter
from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool, ThreadSafeFaiss
from server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path
from server.utils import torch_gc
from langchain.docstore.document import Document
from typing import List, Dict, Optional, Tuple
