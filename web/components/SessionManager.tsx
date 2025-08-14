'use client';

import { useSession } from '@/contexts/SessionContext';
import { Calendar, FolderOpen, Plus, Trash2 } from 'lucide-react';
import React, { useState } from 'react';

export default function SessionManager() {
  const {
    currentSession,
    sessions,
    loading,
    error,
    createSession,
    loadSession,
    deleteSession,
  } = useSession();

  const [newProjectName, setNewProjectName] = useState('');
  const [showCreateForm, setShowCreateForm] = useState(false);

  const handleCreateSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;

    await createSession(newProjectName);
    setNewProjectName('');
    setShowCreateForm(false);
  };

  const handleLoadSession = async (sessionId: string) => {
    await loadSession(sessionId);
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('Are you sure you want to delete this session?')) {
      await deleteSession(sessionId);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className="bg-white border-r border-gray-200 w-80 p-4 h-full overflow-y-auto">
      <div className="mb-6">
        {/* Logo */}
        <div className="flex justify-center mb-4">
          <img 
            src="/jia.svg" 
            alt="JIA Logo" 
            className="h-30 w-30"
          />
        </div>
        
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Session Management</h2>
        
        {/* Current Session Display */}
        {currentSession && (
          <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="flex items-center gap-2 mb-1">
              <FolderOpen className="h-4 w-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-900">Current Session</span>
            </div>
            <p className="text-sm text-blue-800">{currentSession.project_name}</p>
            <p className="text-xs text-blue-600">
              {formatDate(currentSession.created_at)}
            </p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}

        {/* Create New Session */}
        <div className="mb-4">
          {!showCreateForm ? (
            <button
              onClick={() => setShowCreateForm(true)}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              disabled={loading}
            >
              <Plus className="h-4 w-4" />
              Create New Session
            </button>
          ) : (
            <form onSubmit={handleCreateSession} className="space-y-2">
              <input
                type="text"
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                placeholder="Enter project name..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                autoFocus
                disabled={loading}
              />
              <div className="flex gap-2">
                <button
                  type="submit"
                  className="flex-1 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm"
                  disabled={loading || !newProjectName.trim()}
                >
                  Create
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowCreateForm(false);
                    setNewProjectName('');
                  }}
                  className="flex-1 px-3 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors text-sm"
                  disabled={loading}
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>
      </div>

      {/* Session List */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-3">Existing Sessions</h3>
        
        {loading && (
          <div className="text-center py-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto"></div>
            <p className="text-sm text-gray-500 mt-2">Loading...</p>
          </div>
        )}

        {!loading && sessions.length === 0 && (
          <p className="text-sm text-gray-500 text-center py-4">
            No sessions found. Create your first session to get started.
          </p>
        )}

        <div className="space-y-2">
          {sessions.map((session) => (
            <div
              key={session.session_id}
              className={`p-3 border rounded-lg cursor-pointer hover:bg-gray-50 transition-colors ${
                currentSession?.session_id === session.session_id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200'
              }`}
              onClick={() => handleLoadSession(session.session_id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {session.project_name}
                  </p>
                  <div className="flex items-center gap-1 mt-1">
                    <Calendar className="h-3 w-3 text-gray-400" />
                    <p className="text-xs text-gray-500">
                      {formatDate(session.created_at)}
                    </p>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Status: {session.status}
                  </p>
                </div>
                <button
                  onClick={(e) => handleDeleteSession(session.session_id, e)}
                  className="ml-2 p-1 text-red-400 hover:text-red-600 transition-colors"
                  title="Delete session"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}