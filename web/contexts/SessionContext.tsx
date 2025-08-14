'use client';

import { apiService, ChatMessage, Session } from '@/services/api';
import { createContext, ReactNode, useContext, useEffect, useState } from 'react';

interface SessionContextType {
  currentSession: Session | null;
  sessions: Session[];
  chatHistory: ChatMessage[];
  uploadedMaterials: { [key: string]: string };
  loading: boolean;
  error: string | null;
  
  // Actions
  createSession: (projectName: string) => Promise<void>;
  loadSession: (sessionId: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  refreshSessions: () => Promise<void>;
  addChatMessage: (message: ChatMessage) => void;
  clearChatHistory: () => void;
  addUploadedMaterials: (materials: { [key: string]: string }) => void;
  setError: (error: string | null) => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

interface SessionProviderProps {
  children: ReactNode;
}

export function SessionProvider({ children }: SessionProviderProps) {
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [uploadedMaterials, setUploadedMaterials] = useState<{ [key: string]: string }>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load sessions on mount
  useEffect(() => {
    refreshSessions();
  }, []);

  const createSession = async (projectName: string) => {
    try {
      setLoading(true);
      setError(null);
      const newSession = await apiService.createSession(projectName);
      setCurrentSession(newSession);
      setChatHistory([]);
      setUploadedMaterials({});
      await refreshSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create session');
    } finally {
      setLoading(false);
    }
  };

  const loadSession = async (sessionId: string) => {
    try {
      setLoading(true);
      setError(null);
      
      // Use switch session API to automatically save current session memory
      const currentSessionId = currentSession?.session_id;
      const switchResponse = await apiService.switchToSession(sessionId, currentSessionId);
      
      setCurrentSession(switchResponse.session);
      
      // Load chat history from memory
      try {
        const memoryData = await apiService.getSessionMemory(sessionId);
        if (memoryData.memory && memoryData.memory.steps) {
          const messages: ChatMessage[] = memoryData.memory.steps
            .filter((step: any) => step.step_type === 'user' || step.step_type === 'assistant')
            .map((step: any) => ({
              role: step.step_type,
              content: step.content || '',
              timestamp: step.timing?.start_time || new Date().toISOString(),
            }));
          setChatHistory(messages);
        } else {
          setChatHistory([]);
        }
      } catch (memoryError) {
        console.warn('Could not load session memory:', memoryError);
        setChatHistory([]);
      }
      
      setUploadedMaterials({});
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load session');
    } finally {
      setLoading(false);
    }
  };

  const deleteSession = async (sessionId: string) => {
    try {
      setLoading(true);
      setError(null);
      await apiService.deleteSession(sessionId);
      
      if (currentSession?.session_id === sessionId) {
        setCurrentSession(null);
        setChatHistory([]);
        setUploadedMaterials({});
      }
      
      await refreshSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete session');
    } finally {
      setLoading(false);
    }
  };

  const refreshSessions = async () => {
    try {
      const data = await apiService.listSessions();
      setSessions(data.sessions || []);
    } catch (err) {
      console.error('Failed to refresh sessions:', err);
      setSessions([]);
    }
  };

  const addChatMessage = (message: ChatMessage) => {
    setChatHistory(prev => [...prev, message]);
  };

  const clearChatHistory = () => {
    setChatHistory([]);
  };

  const addUploadedMaterials = (materials: { [key: string]: string }) => {
    setUploadedMaterials(prev => ({ ...prev, ...materials }));
  };

  const value: SessionContextType = {
    currentSession,
    sessions,
    chatHistory,
    uploadedMaterials,
    loading,
    error,
    createSession,
    loadSession,
    deleteSession,
    refreshSessions,
    addChatMessage,
    clearChatHistory,
    addUploadedMaterials,
    setError,
  };

  return (
    <SessionContext.Provider value={value}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}