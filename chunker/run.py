"""
LLM-Assisted Chunker - Launcher Script

This script provides both CLI and API interfaces.

Usage:
    CLI mode:  python run.py
    API mode:  python run.py --api
"""

import sys
import argparse
from pathlib import Path
import tempfile
import shutil
import json

# Add src and utils directories to Python path
src_path = Path(__file__).parent / "src"
utils_path = Path(__file__).parent / "utils"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(utils_path))

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import existing functions from utils and src
from utils.text_utils import extract_text, count_tokens, get_text_excerpt, get_file_info, SUPPORTED_EXTENSIONS
from utils.llm_utils import suggest_chunking_strategy
from src.chunking_strategies import CHUNKING_STRATEGIES
from src.main import main as cli_main


# FastAPI app
app = FastAPI(
    title="LLM-Assisted Chunker",
    description="Upload a file and get intelligent chunking",
    version="1.0.0",
    #docs_url=None,  # Disable /docs
    #redoc_url=None  # Disable /redoc
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# HTML Interface at root
#@app.get("/", response_class=HTMLResponse)
async def root():
    """Main interface - Upload file and process"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM-Assisted Chunker</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 800px;
                width: 100%;
            }
            h1 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 20px;
                cursor: pointer;
                transition: all 0.3s;
            }
            .upload-area:hover {
                background: #f8f9ff;
                border-color: #764ba2;
            }
            .upload-area.dragover {
                background: #e8ebff;
                border-color: #667eea;
            }
            input[type="file"] {
                display: none;
            }
            .file-icon {
                font-size: 4em;
                margin-bottom: 20px;
            }
            .upload-text {
                font-size: 1.2em;
                color: #667eea;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .upload-hint {
                color: #999;
                font-size: 0.9em;
            }
            .selected-file {
                background: #f0f7ff;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
            }
            .selected-file.show {
                display: block;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 10px;
                font-size: 1.1em;
                cursor: pointer;
                width: 100%;
                font-weight: bold;
                transition: transform 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            .btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .results {
                display: none;
                margin-top: 30px;
            }
            .results.show {
                display: block;
            }
            .step {
                background: #f8f9ff;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 5px solid #667eea;
            }
            .step h3 {
                color: #667eea;
                margin-bottom: 10px;
            }
            .recommendation {
                background: #fff4e6;
                padding: 15px;
                border-radius: 8px;
                border-left: 5px solid #ff9800;
                margin: 10px 0;
            }
            .strategy-option {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                cursor: pointer;
                border: 2px solid #e0e0e0;
                transition: all 0.3s;
            }
            .strategy-option:hover {
                border-color: #667eea;
                box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
            }
            .strategy-option.selected {
                border-color: #667eea;
                background: #f0f7ff;
            }
            .chunks-summary {
                background: #e8f5e9;
                padding: 15px;
                border-radius: 8px;
                border-left: 5px solid #4caf50;
                margin: 10px 0;
            }
            .chunk-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                border-left: 3px solid #667eea;
            }
            .chunk-meta {
                color: #999;
                font-size: 0.9em;
                margin-bottom: 5px;
            }
            .chunk-text {
                color: #333;
                line-height: 1.6;
            }
            .strategy-selector {
                margin: 20px 0;
            }
            .strategy-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                cursor: pointer;
                border: 2px solid #e0e0e0;
                transition: all 0.3s;
            }
            .strategy-card:hover {
                border-color: #667eea;
                box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
                transform: translateX(5px);
            }
            .strategy-card.selected {
                border-color: #667eea;
                background: #f0f7ff;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            .strategy-card.recommended {
                border-color: #ff9800;
                background: #fff4e6;
            }
            .strategy-card.recommended::before {
                content: '⭐ ';
            }
            .btn-secondary {
                background: #4caf50;
                margin-top: 20px;
            }
            .btn-secondary:hover {
                background: #45a049;
            }
            .download-btn {
                background: #4caf50;
                display: inline-block;
                padding: 12px 30px;
                border-radius: 8px;
                color: white;
                text-decoration: none;
                font-weight: bold;
                margin: 10px 0;
                cursor: pointer;
                border: none;
                font-size: 1em;
            }
            .download-btn:hover {
                background: #45a049;
                transform: translateY(-2px);
            }
            .hidden {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 LLM-Assisted Chunker</h1>
            <p class="subtitle">Upload your document and let AI help chunk it intelligently</p>
            
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                <div class="file-icon">📄</div>
                <div class="upload-text">Click or drag file here</div>
                <div class="upload-hint">Supports: PDF, DOCX, TXT, MD, CSV, JSON, XML, HTML</div>
            </div>
            
            <input type="file" id="fileInput" accept=".pdf,.docx,.txt,.md,.csv,.log,.json,.xml,.html">
            
            <div class="selected-file" id="selectedFile">
                <strong>Selected:</strong> <span id="fileName"></span>
            </div>
            
            <button class="btn" id="processBtn" onclick="processFile()" disabled>
                Process Document
            </button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your document...</p>
            </div>
            
            <div class="results" id="results"></div>
        </div>

        <script>
            let selectedFile = null;
            let fileInfo = null;
            let recommendation = null;
            let analysisData = null;
            let selectedStrategy = null;

            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const selectedFileDiv = document.getElementById('selectedFile');
            const fileName = document.getElementById('fileName');
            const processBtn = document.getElementById('processBtn');
            const loading = document.getElementById('loading');
            const results = document.getElementById('results');

            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFileSelect(files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFileSelect(e.target.files[0]);
                }
            });

            function handleFileSelect(file) {
                selectedFile = file;
                fileName.textContent = file.name;
                selectedFileDiv.classList.add('show');
                processBtn.disabled = false;
            }

            async function processFile() {
                if (!selectedFile) return;

                processBtn.disabled = true;
                loading.classList.add('show');
                results.classList.remove('show');
                results.innerHTML = '';

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    // Step 1 & 2: Analyze and get recommendation
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });

                    analysisData = await response.json();
                    displayAnalysisAndStrategies(analysisData);
                } catch (error) {
                    results.innerHTML = `<div class="step" style="border-left-color: red;">
                        <h3>❌ Error</h3>
                        <p>${error.message}</p>
                    </div>`;
                    results.classList.add('show');
                } finally {
                    loading.classList.remove('show');
                    processBtn.disabled = false;
                }
            }

            function displayAnalysisAndStrategies(data) {
                let html = '';

                // Step 1: File Info
                html += `
                    <div class="step">
                        <h3>📄 Step 1: File Analysis</h3>
                        <p><strong>File Type:</strong> ${data.file_info.file_type}</p>
                        <p><strong>File Size:</strong> ${data.file_info.size_kb} KB</p>
                        <p><strong>Characters:</strong> ${data.file_info.text_length.toLocaleString()}</p>
                        <p><strong>Tokens:</strong> ${data.file_info.token_count.toLocaleString()}</p>
                    </div>
                `;

                // Step 2: LLM Recommendation
                html += `
                    <div class="step">
                        <h3>🤖 Step 2: AI Recommendation</h3>
                        <div class="recommendation">
                            <p><strong>✨ Recommended Strategy:</strong> ${data.llm_recommendation.strategy_name}</p>
                            <p><strong>Reason:</strong> ${data.llm_recommendation.reasoning}</p>
                        </div>
                    </div>
                `;

                // Step 3: User Selection
                html += `
                    <div class="step">
                        <h3>⚙️ Step 3: Select Your Chunking Strategy</h3>
                        <p style="margin-bottom: 15px; color: #666;">Click on a strategy to select it:</p>
                        <div class="strategy-selector">
                            ${data.available_strategies.map(s => `
                                <div class="strategy-card ${s.id === data.llm_recommendation.recommended_strategy ? 'recommended' : ''}" 
                                     onclick="selectStrategy(${s.id})" 
                                     id="strategy-${s.id}">
                                    <strong>${s.id}. ${s.name}</strong>
                                    ${s.id === data.llm_recommendation.recommended_strategy ? 
                                        '<span style="color: #ff9800; font-size: 0.9em;"> (AI Recommended)</span>' : ''}
                                </div>
                            `).join('')}
                        </div>
                        <button class="btn btn-secondary hidden" id="chunkBtn" onclick="performChunking()">
                            ✨ Chunk Document
                        </button>
                    </div>
                `;

                results.innerHTML = html;
                results.classList.add('show');
            }

            function selectStrategy(strategyId) {
                selectedStrategy = strategyId;
                
                // Update UI
                document.querySelectorAll('.strategy-card').forEach(card => {
                    card.classList.remove('selected');
                });
                document.getElementById(`strategy-${strategyId}`).classList.add('selected');
                document.getElementById('chunkBtn').classList.remove('hidden');
            }

            async function performChunking() {
                if (!selectedStrategy) return;

                const chunkBtn = document.getElementById('chunkBtn');
                chunkBtn.disabled = true;
                chunkBtn.textContent = 'Chunking...';

                const formData = new FormData();
                formData.append('file', selectedFile);
                formData.append('strategy', selectedStrategy);

                try {
                    const response = await fetch('/chunk', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    displayChunkingResults(data);
                } catch (error) {
                    alert('Error during chunking: ' + error.message);
                } finally {
                    chunkBtn.disabled = false;
                    chunkBtn.textContent = '✨ Chunk Document';
                }
            }

            function displayChunkingResults(data) {
                // Add Step 4 & 5
                let html = `
                    <div class="step">
                        <h3>✅ Step 4: Selected Strategy</h3>
                        <p><strong>${data.selected_strategy.name}</strong></p>
                    </div>

                    <div class="step">
                        <h3>📊 Step 5: Chunking Results</h3>
                        <div class="chunks-summary">
                            <p><strong>Total Chunks Created:</strong> ${data.chunks.length}</p>
                            <p><strong>Total Tokens:</strong> ${data.summary.total_tokens.toLocaleString()}</p>
                            <p><strong>Total Characters:</strong> ${data.summary.total_chars.toLocaleString()}</p>
                        </div>
                        
                        <button class="download-btn" onclick="downloadChunks()">
                            📥 Download Chunks (JSON)
                        </button>

                        <h4 style="margin-top: 20px;">Chunks Preview (showing first 5):</h4>
                        ${data.chunks.slice(0, 5).map(chunk => `
                            <div class="chunk-item">
                                <div class="chunk-meta">
                                    Chunk ${chunk.chunk_id} | ${chunk.token_count} tokens | ${chunk.char_count} chars
                                </div>
                                <div class="chunk-text">
                                    ${chunk.text.substring(0, 200)}${chunk.text.length > 200 ? '...' : ''}
                                </div>
                            </div>
                        `).join('')}
                        ${data.chunks.length > 5 ? `<p><em>... and ${data.chunks.length - 5} more chunks</em></p>` : ''}
                    </div>
                `;

                // Store chunks data for download
                window.chunksData = data;

                // Append to results
                results.innerHTML += html;
            }

            function downloadChunks() {
                if (!window.chunksData) return;

                const dataStr = JSON.stringify(window.chunksData, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `chunks_${window.chunksData.source_file}_${Date.now()}.json`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            }


        </script>
    </body>
    </html>
    """
    return html_content


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Step 1 & 2: Analyze file and get LLM recommendation
    
    This endpoint:
    1. Extracts text from uploaded file
    2. Gets file info (type, size)
    3. Gets LLM recommendation for chunking strategy
    4. Returns available strategies for user selection
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Step 1: Extract text and get file info
        text = extract_text(tmp_path)
        file_info = get_file_info(tmp_path)
        file_info['text_length'] = len(text)
        file_info['token_count'] = count_tokens(text)
        
        # Step 2: Get LLM recommendation
        excerpt = get_text_excerpt(text, max_chars=1000)
        llm_recommendation = suggest_chunking_strategy(
            file_type=file_info['file_type'],
            file_size_kb=file_info['size_kb'],
            text_excerpt=excerpt
        )
        
        # Get available strategies
        available_strategies = [
            {"id": int(k), "name": v[0]} 
            for k, v in CHUNKING_STRATEGIES.items()
        ]
        
        return {
            "source_file": file.filename,
            "file_info": file_info,
            "llm_recommendation": llm_recommendation,
            "available_strategies": available_strategies
        }
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/chunk")
async def chunk_document(file: UploadFile = File(...), strategy: int = Form(...)):
    """
    Step 3, 4 & 5: Perform chunking based on user-selected strategy
    
    This endpoint:
    1. Extracts text from file
    2. Uses user-selected chunking strategy
    3. Performs chunking
    4. Returns chunks with download capability
    """
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    if str(strategy) not in CHUNKING_STRATEGIES:
        raise HTTPException(
            status_code=400, 
            detail="Invalid strategy. Choose 1-4"
        )
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Extract text
        text = extract_text(tmp_path)
        
        # Get selected strategy
        strategy_name, strategy_func = CHUNKING_STRATEGIES[str(strategy)]
        
        # Perform chunking
        chunks = strategy_func(text)
        
        # Prepare chunk data
        chunk_data = [
            {
                "chunk_id": i,
                "text": chunk,
                "token_count": count_tokens(chunk),
                "char_count": len(chunk)
            }
            for i, chunk in enumerate(chunks, 1)
        ]
        
        # Return complete results with all info
        return {
            "source_file": file.filename,
            "selected_strategy": {
                "id": strategy,
                "name": strategy_name
            },
            "chunks": chunk_data,
            "summary": {
                "total_chunks": len(chunk_data),
                "total_tokens": sum(c["token_count"] for c in chunk_data),
                "total_chars": sum(c["char_count"] for c in chunk_data)
            }
        }
        
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_api(host="0.0.0.0", port=8000, reload=False):
    """Run API server"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          LLM-Assisted Chunker API Server                     ║
╚══════════════════════════════════════════════════════════════╝

🚀 Server: http://{host}:{port}
📚 Docs:   http://{host}:{port}/docs

Press CTRL+C to stop
""")
    uvicorn.run("run:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Assisted Chunker")
    parser.add_argument("--api", action="store_true", help="Run API server")
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload")
    
    args = parser.parse_args()
    
    if args.api:
        run_api(host=args.host, port=args.port, reload=args.reload)
    else:
        cli_main()
