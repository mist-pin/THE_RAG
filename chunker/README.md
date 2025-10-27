# LLM-Assisted Chunker

Intelligent text chunking with AI-powered strategy recommendations. The system analyzes your document and suggests the best chunking approach from 4 different strategies.

##  Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Configure API key (optional - works without it)
copy .env.example .env
# Edit .env and add: GEMINI_API_KEY=your-key-here

# 3. Run the application
python run.py
```

**Get API Key:** https://makersuite.google.com/app/apikey

## 📁 Project Structure

```
LLM_Assisted_Chunker/
├── src/
│   ├── main.py                 # Main application logic
│   └── chunking_strategies.py  # 4 chunking implementations
├── utils/
│   ├── llm_utils.py           # LLM integration & recommendations
│   └── text_utils.py          # Text extraction & token counting
├── data/                       # Input documents
├── output/                     # Generated JSON chunks
├── logs/                       # Application logs
├── run.py                      # Application launcher (API Entry Point)
├── requirements.txt            # Dependencies
└── .env                        # API configuration
```

##  Features

### 4 Chunking Strategies

1. **Sentence-based** - Groups by sentences (fine-grained analysis)
2. **Paragraph** - Groups by paragraphs (general documents)
3. **Section-based** - Groups by headers (structured documents like markdown)
4. **Semantic** - Intelligent context-aware chunking (complex texts)

### Key Capabilities

- 🤖 **LLM-powered recommendations** - Get AI suggestions for optimal strategy
- 📄 **Multi-format support** - PDF, DOCX, TXT, MD, CSV, JSON, XML, HTML, LOG
- 🎯 **Token-accurate** - Uses tiktoken for precise token counting
- 🔄 **Works offline** - Mock mode available without API key
- 📊 **JSON output** - Saves chunks with metadata to `output/` directory

## 🎮 Usage

```bash
python run.py
```

**Workflow:**
1. Enter file path (e.g., `data/sample_document.txt`)
2. View LLM recommendation ⭐
3. Select chunking strategy (1-4)
4. Get chunks saved as JSON in `output/`

## 📊 Output Format

```json
{
  "source_file": "/path/to/file.txt",
  "chunking_strategy": "Semantic-based chunking",
  "num_chunks": 8,
  "chunks": [
    {
      "chunk_id": 1,
      "text": "...",
      "token_count": 485,
      "char_count": 2031
    }
  ]
}
```

## 🔧 Configuration

Create `.env` file in project root:

```env
GEMINI_API_KEY=your-actual-api-key-here
GEMINI_MODEL=gemini-2.0-flash-exp
```

**Available Models:**
- `gemini-2.0-flash-exp` ⚡ (Recommended - Fast)
- `gemini-1.5-pro` 🧠 (More capable, slower)
- `gemini-1.5-flash` ⚖️ (Balanced)

**Note:** System works without API key using basic heuristics.

##  Troubleshooting

**"Import could not be resolved"**
```bash
pip install -r requirements.txt
```

**"spaCy model not found"**
```bash
python -m spacy download en_core_web_sm
```

**"Running in MOCK MODE"**
- Add valid API key to `.env` file
- System still works with basic recommendations

##  Tips

- **Markdown files** → Use Section-based chunking
- **CSV files** → Use Paragraph chunking
- **General text** → Use Semantic chunking (recommended)
- **Logs** → Check `logs/chunking.log` for details
- **Output** → All chunks saved to `output/` directory

---

**Happy Chunking! **
