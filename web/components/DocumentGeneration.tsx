'use client';

import { useSession } from '@/contexts/SessionContext';
import { apiService, DocumentGenerationConfig } from '@/services/api';
import 'github-markdown-css/github-markdown-light.css';
import { BarChart3, Download, FileText, Settings } from 'lucide-react';
import React, { useEffect, useState } from 'react';
import MarkdownView from 'react-showdown';
import ProgressDisplay from './ProgressDisplay';
import StatisticsDisplay from './StatisticsDisplay';


export default function DocumentGeneration() {
  const { currentSession, uploadedMaterials, setError } = useSession();
  
  const [productIdea, setProductIdea] = useState('');
  const [config, setConfig] = useState<DocumentGenerationConfig>({
    enable_safety_check: true,
    max_workflow_steps: 5,
    temperature: 0.05,
    max_tokens: 1000,
  });
  
  const [generating, setGenerating] = useState(false);
  const [generatedDocument, setGeneratedDocument] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [progressMessages, setProgressMessages] = useState<Array<{
    timestamp: string;
    stage: string;
    message: string;
    type: 'info' | 'success' | 'warning' | 'error' | 'in_progress' | 'calling_function' | 'stage_start' | 'stage_complete' | 'step_start' | 'step_complete' | 'custom';
    agent_name?: string;
    step_count?: number;
    total_steps?: number;
  }>>([]);
  const [loadingExisting, setLoadingExisting] = useState(false);
  const [statistics, setStatistics] = useState<{
    session_id: string;
    summary: {
      total_agents_used: number;
      total_execution_time_seconds: number;
      total_tokens: number;
      input_tokens: number;
      output_tokens: number;
    };
    cost_estimate: {
      input_cost_usd: number;
      output_cost_usd: number;
      total_cost_usd: number;
    };
    agents_breakdown: Record<string, {
      total_tokens?: number;
      total_steps?: number;
      total_time_ms?: number;
    }>;
    workflow_diagram?: {
      nodes: Array<{
        id: string;
        label: string;
        type: string;
        description: string;
        role: string;
        expertise: string[];
        level: number;
      }>;
      edges: Array<{
        from: string;
        to: string;
        type: string;
      }>;
      layout: string;
      reasoning: string;
      total_steps: number;
    };
  } | null>(null);
  const [showStatistics, setShowStatistics] = useState(false);

  // Reset state and load existing document data when session changes
  useEffect(() => {
    // Reset all state when session changes
    setProductIdea('');
    setGeneratedDocument(null);
    setProgressMessages([]);
    setStatistics(null);
    setShowStatistics(false);
    setGenerating(false);
    
    const loadExistingDocumentData = async () => {
      if (!currentSession) return;

      setLoadingExisting(true);
      try {
        const documentData = await apiService.getSessionDocumentData(currentSession.session_id);
        
        if (documentData.has_document) {
          setGeneratedDocument(documentData.document);
        }
        
        if (documentData.has_statistics) {
          // Merge statistics with workflow diagram if available
          const stats = { ...documentData.statistics };
          if (documentData.has_workflow && documentData.workflow_diagram) {
            stats.workflow_diagram = documentData.workflow_diagram;
          }
          setStatistics(stats);
        }
        
        // Auto-show statistics if both document and stats exist
        if (documentData.has_document && documentData.has_statistics) {
          setShowStatistics(true);
        }
        
      } catch (error) {
        console.warn('Could not load existing document data:', error);
        // Don't show error to user - this is expected for new sessions
      } finally {
        setLoadingExisting(false);
      }
    };

    loadExistingDocumentData();
  }, [currentSession?.session_id]);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!productIdea.trim() || !currentSession || generating) return;

    setGenerating(true);
    setGeneratedDocument(null);
    setProgressMessages([]);
    setStatistics(null);
    setShowStatistics(false);
    setError(null);

    try {
      // Use the streaming endpoint for real-time progress
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const params = new URLSearchParams({
        user_input: productIdea,
        additional_materials: JSON.stringify(uploadedMaterials),
      });
      const url = `${baseUrl}/documents/generate/stream/${currentSession.session_id}?${params}`;
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader available');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.error) {
                setError(data.error);
                break;
              }
              
              if (data.is_complete) {
                // Final result
                if (data.success && data.document) {
                  setGeneratedDocument(data.document);
                  
                  // Get statistics
                  try {
                    const statsResponse = await apiService.getSessionStatistics(currentSession.session_id);
                    setStatistics(statsResponse);
                  } catch (statsError) {
                    console.warn('Could not load statistics:', statsError);
                  }
                } else {
                  setError(data.message || 'Document generation failed');
                }
                break;
              } else {
                // Progress message
                setProgressMessages(prev => [...prev, {
                  timestamp: data.timestamp,
                  stage: data.stage,
                  message: data.message,
                  type: data.type || 'info',
                  agent_name: data.agent_name,
                  step_count: data.step_count,
                  total_steps: data.total_steps,
                }]);
              }
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', parseError);
            }
          }
        }
      }

    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to generate document');
    } finally {
      setGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!generatedDocument || !currentSession) return;

    const blob = new Blob([generatedDocument], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentSession.project_name}_product_design.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!currentSession) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center p-8">
          <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No Session Selected</h3>
          <p className="text-gray-500">
            Please create or select a session to generate documents.
          </p>
        </div>
      </div>
    );
  }

  if (loadingExisting) {
    return (
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="text-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-500">Loading existing document data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full bg-white overflow-y-auto">
      <div className="max-w-4xl mx-auto p-6">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Generate Product Design Document</h2>
          <p className="text-gray-600">
            Create a comprehensive product design document for your project: {currentSession.project_name}
          </p>
        </div>

        {/* Generation Form */}
        {!generatedDocument && (
          <form onSubmit={handleGenerate} className="space-y-6">
            <div>
              <label htmlFor="productIdea" className="block text-sm font-medium text-gray-700 mb-2">
                Describe Your Product Idea *
              </label>
              <textarea
                id="productIdea"
                value={productIdea}
                onChange={(e) => setProductIdea(e.target.value)}
                placeholder="Example: A mobile fitness app with social features and AI-powered workout recommendations..."
                className="w-full h-32 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                disabled={generating}
                required
              />
            </div>

            {/* Advanced Settings */}
            <div>
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-800 transition-colors"
              >
                <Settings className="h-4 w-4" />
                Advanced Settings
              </button>

              {showAdvanced && (
                <div className="mt-4 p-4 bg-gray-50 rounded-lg space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={config.enable_safety_check}
                          onChange={(e) => setConfig({
                            ...config,
                            enable_safety_check: e.target.checked
                          })}
                          className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-700">Enable Safety Check</span>
                      </label>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Max Workflow Steps: {config.max_workflow_steps}
                      </label>
                      <input
                        type="range"
                        min="3"
                        max="10"
                        value={config.max_workflow_steps}
                        onChange={(e) => setConfig({
                          ...config,
                          max_workflow_steps: parseInt(e.target.value)
                        })}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        AI Creativity: {config.temperature}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={config.temperature}
                        onChange={(e) => setConfig({
                          ...config,
                          temperature: parseFloat(e.target.value)
                        })}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Max Response Length: {config.max_tokens}
                      </label>
                      <input
                        type="range"
                        min="500"
                        max="3000"
                        step="100"
                        value={config.max_tokens}
                        onChange={(e) => setConfig({
                          ...config,
                          max_tokens: parseInt(e.target.value)
                        })}
                        className="w-full"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Uploaded Materials */}
            {Object.keys(uploadedMaterials).length > 0 && (
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h4 className="font-medium text-blue-900 mb-2">Uploaded Materials</h4>
                <div className="space-y-1">
                  {Object.keys(uploadedMaterials).map((filename) => (
                    <div key={filename} className="text-sm text-blue-800">
                      📄 {filename}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={!productIdea.trim() || generating}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {generating ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Generating Document...
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4" />
                  Generate Document
                </>
              )}
            </button>
          </form>
        )}

        {/* Generated Document Display */}
        {generatedDocument && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Generated Document</h3>
              <div className="flex gap-2">
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  <Download className="h-4 w-4" />
                  Download
                </button>
                <button
                  onClick={() => setShowStatistics(!showStatistics)}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <BarChart3 className="h-4 w-4" />
                  {showStatistics ? 'Hide' : 'Show'} Stats
                </button>
                <button
                  onClick={() => {
                    setGeneratedDocument(null);
                    setProductIdea('');
                    setProgressMessages([]);
                    setStatistics(null);
                    setShowStatistics(false);
                  }}
                  className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
                >
                  Generate New
                </button>
              </div>
            </div>

            {/* Statistics Display - top of the page for better visibility */}
            {statistics && showStatistics && (
              <StatisticsDisplay 
                statistics={statistics} 
                onClose={() => setShowStatistics(false)}
              />
            )}

            <div className="prose prose-gray max-w-none p-6 bg-white border rounded-lg">
              <div className="markdown-body text-sm leading-relaxed">
                <MarkdownView
                  markdown={generatedDocument}
                  options={{ tables: true, emoji: true }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Progress Display */}
        <ProgressDisplay 
          messages={progressMessages} 
          isGenerating={generating} 
        />

      </div>
    </div>
  );
}