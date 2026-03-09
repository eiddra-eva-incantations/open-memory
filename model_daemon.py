import json
import time
import threading
import gc
import sys
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# --- Configuration & Models ---
EMBED_MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'
RERANK_MODEL_NAME = 'Qwen/Qwen3-Reranker-4B'
VEC_DIM = 1024 
PORT = 50051

# Detect CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model_daemon")

model = None
reranker = None

class ModelVRAMManager:
    def __init__(self):
        self.last_used = time.time()
        self.active_count = 0 
        self.lock = threading.RLock()
        self.on_cuda = False
        
        # Start watchdog
        self.watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog.start()
        
    def acquire(self):
        with self.lock:
            self.active_count += 1
            self.last_used = time.time()
            if self.active_count == 1:
                try:
                    global model, reranker
                    if model is not None:
                        model.to(device)
                    if reranker is not None:
                        reranker.model.to(device)
                    self.on_cuda = True
                    logger.debug(f"Models moved to GPU. VRAM activated.")
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        logger.warning("CUDA OOM detected! Falling back to CPU inference...")
                        if model is not None:
                            model.to('cpu')
                        if reranker is not None:
                            reranker.model.to('cpu')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.on_cuda = False
                    else:
                        raise e

    def release(self):
        with self.lock:
            self.active_count -= 1
            self.last_used = time.time()
            if self.active_count == 0:
                global model, reranker
                try:
                    if model is not None:
                        model.to('cpu')
                    if reranker is not None:
                        reranker.model.to('cpu')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.debug("Models offloaded to CPU. VRAM freed.")
                except Exception as e:
                    logger.warning(f"Error during VRAM CPU offload: {e}")
                finally:
                    self.on_cuda = False

    def _watchdog_loop(self):
        while True:
            time.sleep(10)
            with self.lock:
                if self.active_count > 0:
                    continue
                # 5 minutes of complete inactivity
                if time.time() - self.last_used > 300: 
                    global model, reranker
                    if model is not None or reranker is not None:
                        logger.info("Models idle for 5 minutes. Unloading completely...")
                        model = None
                        reranker = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()


vram_manager = ModelVRAMManager()

class GPUContext:
    def __enter__(self):
        vram_manager.acquire()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        vram_manager.release()

def get_embedding_model():
    """Lazy-load only the embedding model into system RAM."""
    global model
    if model is None:
        logger.info("Loading embedding model into system RAM (float16)...")
        model = SentenceTransformer(
            EMBED_MODEL_NAME, 
            device='cpu', 
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16}
        )
        if vram_manager.on_cuda:
            model.to(device)
    return model

def get_reranker():
    """Lazy-load only the reranker model into system RAM."""
    global reranker
    if reranker is None:
        logger.info("Loading reranker model into system RAM (float16)...")
        reranker = CrossEncoder(
            RERANK_MODEL_NAME, 
            device='cpu', 
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16}
        )
        if reranker.tokenizer.pad_token is None:
            reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
        if reranker.model.config.pad_token_id is None:
            reranker.model.config.pad_token_id = reranker.tokenizer.pad_token_id or 0
        if vram_manager.on_cuda:
            reranker.model.to(device)
    return reranker

class ModelRequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, response_code, data):
        self.send_response(response_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_POST(self):
        if self.path == '/encode':
            self._handle_encode()
        elif self.path == '/rerank':
            self._handle_rerank()
        else:
            self._send_response(404, {"error": "Not found. Supported paths: /encode, /rerank"})

    def _handle_encode(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            req_data = json.loads(body)
            text = req_data.get('text')
            
            if not text:
                self._send_response(400, {"error": "Missing 'text' in request body"})
                return

            with GPUContext():
                m = get_embedding_model()
                full_vec = m.encode(text).astype(float).tolist()
                
            self._send_response(200, {"vector": full_vec})
            
        except Exception as e:
            logger.error(f"Error during /encode: {e}", exc_info=True)
            self._send_response(500, {"error": str(e)})

    def _handle_rerank(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            req_data = json.loads(body)
            query = req_data.get('query')
            documents = req_data.get('documents')
            
            if not query or not isinstance(documents, list):
                self._send_response(400, {"error": "Missing 'query' or 'documents' (list) in request body"})
                return
            if not documents:
                self._send_response(200, {"scores": []})
                return

            pairs = [(query, doc) for doc in documents]
            
            with GPUContext():
                reranker_model = get_reranker()
                rerank_scores = reranker_model.predict(pairs, batch_size=4).tolist()
                
            self._send_response(200, {"scores": rerank_scores})
            
        except Exception as e:
            logger.error(f"Error during /rerank: {e}", exc_info=True)
            self._send_response(500, {"error": str(e)})

    def log_message(self, format, *args):
        # Mute standard HTTP server logs to keep it clean, or use logger
        logger.debug(f"{self.client_address[0]} - - [{self.log_date_time_string()}] {format%args}")


if __name__ == "__main__":
    server_address = ('127.0.0.1', PORT)
    httpd = HTTPServer(server_address, ModelRequestHandler)
    logger.info(f"OpenMemory Model Daemon starting on {server_address[0]}:{PORT}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping...")
    finally:
        httpd.server_close()
        logger.info("Daemon stopped.")
