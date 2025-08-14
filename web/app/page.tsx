'use client';

import React, { useState } from 'react';
import SessionManager from '@/components/SessionManager';
import ChatInterface from '@/components/ChatInterface';
import DocumentGeneration from '@/components/DocumentGeneration';
import FileUpload from '@/components/FileUpload';
import { MessageCircle, FileText, Rocket } from 'lucide-react';

type TabType = 'chat' | 'document' | 'upload';

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>('chat');

  const tabs = [
    { id: 'chat' as TabType, label: 'Chat', icon: MessageCircle },
    { id: 'document' as TabType, label: 'Generate Document', icon: FileText },
  ];

  return (
    <div className="h-screen flex bg-gray-100 overflow-hidden">
      {/* Session Manager Sidebar - Fixed */}
      <div className="flex-shrink-0">
        <SessionManager />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header - Fixed */}
        <header className="flex-shrink-0 bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Rocket className="h-8 w-8 text-blue-600" />
                <h1 className="text-2xl font-bold text-gray-900">PM-JIA</h1>
              </div>
              <span className="text-sm text-gray-500">
                AI-Powered Product Design Assistant
              </span>
            </div>
          </div>
        </header>

        {/* Tab Navigation - Fixed */}
        <div className="flex-shrink-0 bg-white border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab Content Area - Scrollable */}
        <div className="flex-1 flex min-h-0">
          <div className="flex-1 min-w-0">
            {activeTab === 'chat' && <ChatInterface />}
            {activeTab === 'document' && <DocumentGeneration />}
          </div>

          {/* File Upload Sidebar - Fixed */}
          <div className="flex-shrink-0">
            <FileUpload />
          </div>
        </div>
      </div>
    </div>
  );
}
