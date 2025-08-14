// API service for connecting to FastAPI backend
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface Session {
  session_id: string;
  project_name: string;
  created_at: string;
  status: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface DocumentGenerationConfig {
  enable_safety_check: boolean;
  max_workflow_steps: number;
  temperature: number;
  max_tokens: number;
}

export interface DocumentGenerationResult {
  session_id: string;
  success: boolean;
  document?: string;
  stage: string;
  message?: string;
  workflow_plan?: Record<string, unknown>;
}

// API functions
export const apiService = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Session management
  async createSession(projectName: string, config?: Record<string, unknown>): Promise<Session> {
    const response = await api.post('/sessions', {
      project_name: projectName,
      config: config || {},
    });
    return response.data;
  },

  async getSession(sessionId: string): Promise<Session> {
    const response = await api.get(`/sessions/${sessionId}`);
    return response.data;
  },

  async listSessions(): Promise<{ sessions: Session[] }> {
    const response = await api.get('/sessions');
    return response.data;
  },

  async deleteSession(sessionId: string) {
    const response = await api.delete(`/sessions/${sessionId}`);
    return response.data;
  },

  async switchToSession(sessionId: string, currentSessionId?: string) {
    const response = await api.post(`/sessions/${sessionId}/switch`, null, {
      params: currentSessionId ? { current_session_id: currentSessionId } : {},
    });
    return response.data;
  },

  async getSessionDocumentData(sessionId: string) {
    const response = await api.get(`/sessions/${sessionId}/document`);
    return response.data;
  },

  // Chat functionality
  async sendChatMessage(sessionId: string, message: string, additionalMaterials?: Record<string, unknown>) {
    const response = await api.post('/chat', {
      session_id: sessionId,
      message,
      additional_materials: additionalMaterials,
    });
    return response.data;
  },

  // Document generation
  async generateDocument(
    sessionId: string,
    userInput: string,
    additionalMaterials?: Record<string, unknown>
  ): Promise<DocumentGenerationResult> {
    const response = await api.post('/documents/generate', {
      session_id: sessionId,
      user_input: userInput,
      additional_materials: additionalMaterials,
    });
    return response.data;
  },

  // File upload
  async uploadFiles(sessionId: string, files: FileList) {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    const response = await api.post(`/documents/upload/${sessionId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Session memory
  async getSessionMemory(sessionId: string) {
    const response = await api.get(`/sessions/${sessionId}/memory`);
    return response.data;
  },

  // Session statistics
  async getSessionStatistics(sessionId: string) {
    const response = await api.get(`/sessions/${sessionId}/statistics`);
    return response.data;
  },
};

export default apiService;