'use client';

import { useSession } from '@/contexts/SessionContext';
import { apiService } from '@/services/api';
import { Bot, MessageCircle, Send, Trash2, User } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';

export default function ChatInterface() {
  const {
    currentSession,
    chatHistory,
    uploadedMaterials,
    addChatMessage,
    clearChatHistory,
    setError,
  } = useSession();

  const [message, setMessage] = useState('');
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  // Auto-focus the input when component mounts or current session changes
  useEffect(() => {
    if (currentSession && inputRef.current) {
      inputRef.current.focus();
    }
  }, [currentSession]);

  // Auto-focus the input after sending a message
  useEffect(() => {
    if (!sending && inputRef.current) {
      inputRef.current.focus();
    }
  }, [sending]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim() || !currentSession || sending) return;

    const userMessage = {
      role: 'user' as const,
      content: message,
      timestamp: new Date().toISOString(),
    };

    addChatMessage(userMessage);
    setMessage('');
    setSending(true);

    try {
      const response = await apiService.sendChatMessage(
        currentSession.session_id,
        message,
        uploadedMaterials
      );

      const assistantMessage = {
        role: 'assistant' as const,
        content: response.response,
        timestamp: response.timestamp,
      };

      addChatMessage(assistantMessage);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to send message');
      
      // Add error message to chat
      const errorMessage = {
        role: 'assistant' as const,
        content: 'Sorry, I encountered an error processing your message. Please try again.',
        timestamp: new Date().toISOString(),
      };
      addChatMessage(errorMessage);
    } finally {
      setSending(false);
    }
  };

  const handleClearHistory = () => {
    if (confirm('Are you sure you want to clear the chat history?')) {
      clearChatHistory();
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (!currentSession) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center p-8">
          <MessageCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Session Selected</h3>
          <p className="text-gray-500">
            Please create or select a session to start chatting with PM-JIA.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Chat Header - Fixed */}
      <div className="flex-shrink-0 border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Chat with PM-JIA</h2>
            <p className="text-sm text-gray-500">Project: {currentSession.project_name}</p>
          </div>
          <button
            onClick={handleClearHistory}
            className="flex items-center gap-2 px-3 py-1 text-sm text-red-600 hover:text-red-800 transition-colors"
            disabled={chatHistory.length === 0}
          >
            <Trash2 className="h-4 w-4" />
            Clear History
          </button>
        </div>
      </div>

      {/* Messages Area - Scrollable */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {chatHistory.length === 0 && (
          <div className="text-center py-8">
            <Bot className="h-12 w-12 text-blue-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Welcome to {currentSession.project_name}!
            </h3>
            <p className="text-gray-500">
              I&apos;m here to help you with product design and development. 
              What would you like to work on today?
            </p>
          </div>
        )}

        {chatHistory.map((msg, index) => (
          <div
            key={index}
            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'assistant' && (
              <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <Bot className="h-4 w-4 text-blue-600" />
              </div>
            )}
            
            <div
              className={`max-w-md lg:max-w-lg xl:max-w-xl p-3 rounded-lg ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {msg.content}
              </div>
              <div
                className={`text-xs mt-1 ${
                  msg.role === 'user' ? 'text-blue-200' : 'text-gray-500'
                }`}
              >
                {formatTimestamp(msg.timestamp)}
              </div>
            </div>

            {msg.role === 'user' && (
              <div className="flex-shrink-0 w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                <User className="h-4 w-4 text-white" />
              </div>
            )}
          </div>
        ))}

        {sending && (
          <div className="flex gap-3 justify-start">
            <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
              <Bot className="h-4 w-4 text-blue-600" />
            </div>
            <div className="bg-gray-100 text-gray-900 p-3 rounded-lg">
              <div className="flex items-center gap-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-sm">PM-JIA is thinking...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Message Input - Fixed at bottom */}
      <div className="flex-shrink-0 border-t border-gray-200 p-4">
        <form onSubmit={handleSendMessage} className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={sending}
            autoFocus
          />
          <button
            type="submit"
            disabled={!message.trim() || sending}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>

        {/* Uploaded Materials Indicator */}
        {Object.keys(uploadedMaterials).length > 0 && (
          <div className="mt-2 text-xs text-gray-500">
            📎 {Object.keys(uploadedMaterials).length} file(s) uploaded
          </div>
        )}
      </div>
    </div>
  );
}