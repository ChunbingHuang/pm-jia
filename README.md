
<p align="center">
  <img src="/src/pm_jia/assets/logo.svg" alt="Logo" width="200" />
</p>

# PM-JIA

Product Management JIA - An AI-powered agent for creating, organizing, and refining structured product design documents.

PM-JIA is an autonomous AI system built without any open-source agent framework. Instead of relying on fixed workflows, it dynamically plans tasks based on each user request. The system allows users to upload additional materials for analysis and uses web search tools to gather real-time information to complete tasks effectively.

A built-in workflow planner:
- Analyzes and understands the user’s input
- Designs a custom workflow
- Orchestrates multiple specialized agents to work together

The system provides both CLI and web interfaces for product management assistance. It features multi-modal input processing, agent-based architecture, and memory management capabilities.

## ✨ Features

- **Multi-modal Input Processing**: Handle text, images, PDFs, tables, and various document formats (under construction)
- **Agent-based Architecture**: Modular design with specialized agents for different tasks
- **Memory System**: Persistent conversation tracking and context management
- **Dual Interfaces**: Both command-line and web-based interfaces
- **Document Generation**: AI-powered product design document creation
- **Session Management**: Persistent project sessions with state management
- **OCR Capabilities**: Extract text from images and PDF documents
- **Progress Tracking**: Real-time progress monitoring for long-running operations
- **Tool System**: Extensible function calling system for agent capabilities (more tools to be added)

## 🏗️ Architecture

PM-JIA follows a modular architecture organized under `src/pm_jia/`:

### Core Components

- **IntelligentOrchestrator** (`agent.py`): Main orchestrator for coordinating specialized agents
- **MaterialProcessor** (`processor.py`): Handles multi-modal input processing with OCR support
- **LLMEngine** (`llm.py`): OpenAI integration layer with structured output support
- **Memory** (`memory.py`): Conversation and step tracking system
- **SessionManager** (`session.py`): Persistent session state management
- **Tools** (`tools.py`): Function calling system for agent capabilities

### Interfaces

- **CLI** (`cli.py`): Rich-based command-line interface with async support
- **FastAPI Backend** (`api.py`): RESTful API with CORS support for web clients
- **Web Frontend** (`web/`): Next.js-based web interface ([web README.md](web/README.md))

### Configuration & Utilities

- **Config** (`config.py`): Centralized configuration classes
- **Prompts** (`prompts/`): YAML-based prompt templates with Jinja2 templating
- **Logger** (`logger.py`): Structured logging configuration
- **Utils** (`utils.py`): Helper functions and utilities

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for general use)
- Tavily API key (for searching)
- Gemini API key (for image analysis)
- Tesseract OCR (for image/PDF processing)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pm-jia
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   - **macOS**: `brew install tesseract`
   - **Ubuntu**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Configure environment:**
   ```bash
   # Create .env file in project root
    cat > .env << 'EOF'
    OPENAI_API_KEY=your_openai_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    EOF
   ```

### Usage

#### Command Line Interface

**Start an interactive session:**
```bash
python cli.py
```

#### FastAPI Backend

**Start the API server:**
```bash
python src/pm_jia/api.py
# or
uvicorn src.pm_jia.api:app --host 0.0.0.0 --port 8000
```

**API Documentation:**
- OpenAPI docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Web Interface

**Start the web frontend:**
```bash
cd web
npm install
npm run dev
```

Navigate to `http://localhost:3000` (requires API backend running on port 8000).

## 📁 Project Structure

```
pm-jia/
├── src/pm_jia/              # Main Python package
│   ├── agent.py             # Agent orchestration
│   ├── api.py               # FastAPI backend
│   ├── config.py            # Configuration classes
│   ├── llm.py               # OpenAI integration
│   ├── memory.py            # Memory management
│   ├── processor.py         # Multi-modal processing
│   ├── session.py           # Session management
│   ├── tools.py             # Function calling system
│   └── prompts/             # YAML prompt templates
├── web/                     # Next.js web interface
├── cli.py                   # Command-line interface
├── sessions/                # Persistent session storage
└── logs/                    # Application logs
```

## 🛠️ Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key    # For web search functionality
GEMINI_API_KEY=your_gemini_api_key    # For image analysis

# Optional
CLI_MODE=True               # Enable CLI-specific features
LOG_LEVEL=INFO              # Logging level
```

### Configuration Classes

- **GeneralConfig**: Project paths and templates
- **AgentConfig**: LLM model settings (defaults to *gpt-4o-mini*)
- **ProcessorConfig**: File processing limits and supported types
- **LoggerConfig**: Logging configuration

## 🧪 Development

### Running Individual Components

**Test agent functionality:**
```bash
python src/pm_jia/agent.py
```

**Test tool system:**
```bash
python src/pm_jia/tools.py
```

### Key Development Notes

- All async functions require `asyncio.run()` when called from synchronous contexts
- The system uses OpenAI's tool calling capabilities extensively
- Environment variables are loaded via python-dotenv
- File uploads are limited to 10MB by default (configurable)
- The agent system supports up to 10 steps per execution (configurable)

## 📋 Supported File Types

- **Text**: TXT, MD, JSON
- **Data**: CSV, XLSX
- **Images**: JPG, PNG (with OCR)
- **Documents**: PDF (with text extraction)

## 🔧 API Endpoints

- `POST /sessions/create` - Create new session
- `GET /sessions/{session_id}` - Get session details
- `POST /sessions/{session_id}/chat` - Send chat message
- `POST /sessions/{session_id}/generate` - Generate document
- `POST /sessions/{session_id}/upload` - Upload materials
- `GET /sessions/{session_id}/memory` - Get conversation history

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request