'use client';

import { AlertCircle, CheckCircle, Clock, Zap } from 'lucide-react';
import { useEffect, useRef } from 'react';

interface ProgressMessage {
  timestamp: string;
  stage: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'in_progress' | 'calling_function' | 'stage_start' | 'stage_complete' | 'step_start' | 'step_complete' | 'custom';
  agent_name?: string;
  step_count?: number;
  total_steps?: number;
}

interface ProgressDisplayProps {
  messages: ProgressMessage[];
  isGenerating: boolean;
}

export default function ProgressDisplay({ messages, isGenerating }: ProgressDisplayProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  const getProgressIcon = (type: string) => {
    switch (type) {
      case 'success':
      case 'stage_complete':
      case 'step_complete':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'in_progress':
      case 'calling_function':
        return <Zap className="h-4 w-4 text-blue-500" />;
      case 'stage_start':
      case 'step_start':
        return <Clock className="h-4 w-4 text-blue-500" />;
      case 'info':
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getProgressColor = (type: string) => {
    switch (type) {
      case 'success':
      case 'stage_complete':
      case 'step_complete':
        return 'border-green-200 bg-green-50';
      case 'error':
        return 'border-red-200 bg-red-50';
      case 'warning':
        return 'border-yellow-200 bg-yellow-50';
      case 'in_progress':
      case 'calling_function':
        return 'border-blue-200 bg-blue-50';
      case 'stage_start':
      case 'step_start':
        return 'border-indigo-200 bg-indigo-50';
      case 'info':
      default:
        return 'border-gray-200 bg-gray-50';
    }
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  if (!isGenerating && messages.length === 0) {
    return null;
  }

  return (
    <div className="mt-6 p-4 border border-blue-200 rounded-lg bg-blue-50">
      <div className="flex items-center gap-2 mb-3">
        <div className={`flex items-center gap-2 ${isGenerating ? 'animate-pulse' : ''}`}>
          {isGenerating ? (
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          ) : (
            <CheckCircle className="h-4 w-4 text-green-500" />
          )}
          <h4 className="font-medium text-blue-900">
            {isGenerating ? 'Generation Progress' : 'Generation Complete'}
          </h4>
        </div>
      </div>

      <div className="space-y-2 max-h-100 overflow-y-auto">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`flex items-start gap-3 p-2 border rounded ${getProgressColor(msg.type)}`}
          >
            <div className="flex-shrink-0 mt-0.5">
              {getProgressIcon(msg.type)}
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-sm font-medium text-gray-900">
                  {msg.stage}
                </span>
                {msg.agent_name && (
                  <span className="text-xs text-gray-500 bg-gray-200 px-2 py-0.5 rounded">
                    {msg.agent_name}
                  </span>
                )}
                {msg.step_count && msg.total_steps && (
                  <span className="text-xs text-blue-600 bg-blue-200 px-2 py-0.5 rounded">
                    Step {msg.step_count}/{msg.total_steps}
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-700">{msg.message}</p>
              <p className="text-xs text-gray-500 mt-1">
                {formatTime(msg.timestamp)}
              </p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {isGenerating && (
        <div className="mt-3 text-xs text-blue-600">
          <div className="flex items-center gap-2">
            <div className="animate-pulse flex space-x-1">
              <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
            </div>
            <span>PM-JIA is working on your document...</span>
          </div>
        </div>
      )}
    </div>
  );
}