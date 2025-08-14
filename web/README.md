# PM-JIA Web Interface

A modern React/Next.js web interface for the PM-JIA AI-powered product design assistant.

## Features

- **Session Management**: Create, load, and manage multiple project sessions
- **Real-time Chat**: Interactive chat interface with PM-JIA assistant
- **Document Generation**: Generate comprehensive product design documents
- **File Upload**: Upload and process various file types (images, PDFs, documents, data files)
- **Progress Tracking**: Real-time progress updates during document generation
- **Responsive Design**: Clean, modern UI built with Tailwind CSS

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- PM-JIA FastAPI backend running on `http://localhost:8000`

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env.local
   ```
   
   Update `.env.local` with your backend API URL if different from default.

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open in browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Project Structure

```
web/
├── app/                    # Next.js app directory
│   ├── layout.tsx         # Root layout with providers
│   ├── page.tsx           # Main application page
│   ├── globals.css        # Global styles
│   └── icon.svg           # App icon
├── components/            # React components
│   ├── SessionManager.tsx # Session management sidebar
│   ├── ChatInterface.tsx  # Chat interface
│   ├── DocumentGeneration.tsx # Document generation form
│   ├── FileUpload.tsx     # File upload component
│   ├── ProgressDisplay.tsx # Progress tracking display
│   └── StatisticsDisplay.tsx # Statistics and metrics display
├── contexts/              # React contexts
│   └── SessionContext.tsx # Session state management
├── services/              # API services and utilities
│   └── api.ts             # Backend API service layer
└── public/               # Static assets
    └── jia.svg           # Application logo
```

## Key Components

### SessionManager
- Create new project sessions
- Load existing sessions from backend
- Display session information and status
- Delete sessions with confirmation

### ChatInterface
- Real-time messaging with PM-JIA
- Message history with timestamps
- Typing indicators and loading states
- Support for uploaded materials context

### DocumentGeneration
- Product idea input form
- Advanced configuration options (safety check, workflow steps, temperature, etc.)
- Real-time generation progress
- Document preview and download

### FileUpload
- Drag & drop file upload
- Support for multiple file types (TXT, MD, JSON, CSV, XLSX, JPG, PNG, PDF)
- File processing status
- Material context for chat and document generation

### ProgressDisplay
- Real-time progress tracking for long-running operations
- Visual progress indicators
- Step-by-step progress updates

### StatisticsDisplay
- Session statistics and metrics
- Usage tracking and analytics
- Performance insights

## Technology Stack

- **Framework**: Next.js 15 with App Router and Turbopack
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **UI Components**: Headless UI
- **Icons**: Lucide React
- **HTTP Client**: Axios
- **Markdown Rendering**: React Showdown with GitHub Markdown CSS
- **State Management**: React Context + Hooks

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Configuration

### Environment Variables

- `NEXT_PUBLIC_API_URL` - Backend API URL (default: http://localhost:8000)
- `NODE_ENV` - Environment (development/production)

### Backend Requirements

Ensure the FastAPI backend is running with CORS enabled for the web interface origin.

## API Integration

The web interface connects to the FastAPI backend through:

- **Sessions API**: Create, load, save, and delete sessions
- **Chat API**: Send messages and receive responses
- **Document Generation API**: Generate product design documents
- **File Upload API**: Process and upload materials
- **Memory API**: Load conversation history
