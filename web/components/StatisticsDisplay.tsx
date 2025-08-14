'use client';

import React, { useState } from 'react';
import { BarChart3, Clock, DollarSign, Cpu, Network, ChevronDown, ChevronUp } from 'lucide-react';

interface WorkflowNode {
  id: string;
  label: string;
  type: string;
  description: string;
  role: string;
  expertise: string[];
  level: number;
}

interface WorkflowEdge {
  from: string;
  to: string;
  type: string;
}

interface WorkflowDiagram {
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  layout: string;
  reasoning: string;
  total_steps: number;
}

interface StatisticsData {
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
  workflow_diagram?: WorkflowDiagram;
}

interface StatisticsDisplayProps {
  statistics: StatisticsData;
  onClose?: () => void;
}

export default function StatisticsDisplay({ statistics, onClose }: StatisticsDisplayProps) {
  const [showWorkflow, setShowWorkflow] = useState(false);
  const { summary, cost_estimate, agents_breakdown, workflow_diagram } = statistics;

  const formatNumber = (num?: number) => {
    if (!num) return '0';
    return num.toLocaleString();
  };

  const formatTime = (seconds?: number) => {
    if (!seconds) return '0s';
    return `${seconds.toFixed(2)}s`;
  };

  const formatTimeMs = (ms?: number) => {
    if (!ms) return '0s';
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatCost = (cost?: number) => {
    if (!cost) return '$0.000000';
    return `$${cost.toFixed(6)}`;
  };

  const renderWorkflowDiagram = (diagram: WorkflowDiagram) => {
    // Group nodes by level for hierarchical display
    const nodesByLevel: Record<number, WorkflowNode[]> = {};
    diagram.nodes.forEach(node => {
      if (!nodesByLevel[node.level]) {
        nodesByLevel[node.level] = [];
      }
      nodesByLevel[node.level].push(node);
    });

    const maxLevel = Math.max(...Object.keys(nodesByLevel).map(Number));

    return (
      <div className="bg-white p-4 rounded-lg border border-blue-200">
        <h4 className="font-medium text-blue-900 mb-3 flex items-center gap-2">
          <Network className="h-4 w-4" />
          Workflow Diagram
        </h4>
        
        {/* Workflow visualization */}
        <div className="space-y-4">
          {Object.keys(nodesByLevel)
            .sort((a, b) => Number(a) - Number(b))
            .map((levelKey) => {
              const level = Number(levelKey);
              const nodes = nodesByLevel[level];
              
              return (
                <div key={level} className="text-center">
                  {/* Level nodes */}
                  <div className="flex flex-wrap justify-center gap-2 mb-2">
                    {nodes.map((node) => (
                      <div 
                        key={node.id}
                        className="px-3 py-2 bg-blue-100 border border-blue-300 rounded-lg text-xs"
                        title={`${node.description} | Role: ${node.role}`}
                      >
                        <div className="font-medium text-blue-900">{node.label}</div>
                        <div className="text-blue-600">{node.type}</div>
                      </div>
                    ))}
                  </div>
                  
                  {/* Arrow to next level */}
                  {level < maxLevel && (
                    <div className="text-blue-400 text-lg">↓</div>
                  )}
                </div>
              );
            })}
        </div>
        
        {/* Workflow reasoning */}
        {diagram.reasoning && (
          <div className="mt-4 p-3 bg-blue-50 rounded border border-blue-200">
            <h5 className="text-sm font-medium text-blue-900 mb-1">Reasoning:</h5>
            <p className="text-xs text-blue-800">{diagram.reasoning}</p>
          </div>
        )}
      </div>
    );
  };

  if (!summary && !cost_estimate && !agents_breakdown) {
    return null;
  }

  return (
    <div className="mb-6 p-4 border border-green-200 rounded-lg bg-green-50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-green-600" />
          <h3 className="text-lg font-semibold text-green-900">Generation Statistics</h3>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-green-600 hover:text-green-800 text-sm"
          >
            ✕
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Overall Summary - Now with comprehensive metrics like CLI */}
        <div className="bg-white p-4 rounded-lg border border-green-200">
          <h4 className="font-medium text-green-900 mb-3 flex items-center gap-2">
            <Cpu className="h-4 w-4" />
            Overall Summary
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Agents Used:</span>
              <span className="font-medium">{summary.total_agents_used || 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Execution Time:</span>
              <span className="font-medium">{formatTime(summary.total_execution_time_seconds)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Total Tokens:</span>
              <span className="font-medium">{formatNumber(summary.total_tokens)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Input Tokens:</span>
              <span className="font-medium">{formatNumber(summary.input_tokens)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Output Tokens:</span>
              <span className="font-medium">{formatNumber(summary.output_tokens)}</span>
            </div>
          </div>
        </div>

        {/* Cost Estimation */}
        <div className="bg-white p-4 rounded-lg border border-green-200">
          <h4 className="font-medium text-green-900 mb-3 flex items-center gap-2">
            <DollarSign className="h-4 w-4" />
            Cost Estimation
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Input Cost:</span>
              <span className="font-medium">{formatCost(cost_estimate.input_cost_usd)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Output Cost:</span>
              <span className="font-medium">{formatCost(cost_estimate.output_cost_usd)}</span>
            </div>
            <div className="flex justify-between border-t pt-2">
              <span className="text-gray-900 font-medium">Total Cost:</span>
              <span className="font-bold text-green-700">
                {formatCost(cost_estimate.total_cost_usd)}
              </span>
            </div>
          </div>
        </div>

        {/* Agent Breakdown */}
        {agents_breakdown && Object.keys(agents_breakdown).length > 0 && (
          <div className="bg-white p-4 rounded-lg border border-green-200">
            <h4 className="font-medium text-green-900 mb-3 flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Agents by Token Usage
            </h4>
            <div className="space-y-2 text-sm max-h-40 overflow-y-auto">
              {Object.entries(agents_breakdown)
                .sort(([, a], [, b]) => (b.total_tokens || 0) - (a.total_tokens || 0))
                .map(([agentName, stats]) => (
                  <div key={agentName} className="border-b border-gray-100 pb-2 last:border-b-0">
                    <div className="font-medium text-gray-900 truncate" title={agentName}>
                      {agentName}
                    </div>
                    <div className="grid grid-cols-3 gap-2 text-xs text-gray-600">
                      <div>
                        <span className="block">Tokens:</span>
                        <span className="font-medium">{formatNumber(stats.total_tokens)}</span>
                      </div>
                      <div>
                        <span className="block">Steps:</span>
                        <span className="font-medium">{stats.total_steps || 0}</span>
                      </div>
                      <div>
                        <span className="block">Time:</span>
                        <span className="font-medium">{formatTimeMs(stats.total_time_ms)}</span>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>

      {/* Workflow Diagram Section */}
      {workflow_diagram && (
        <div className="mt-4">
          <button
            onClick={() => setShowWorkflow(!showWorkflow)}
            className="flex items-center gap-2 px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors text-sm font-medium"
          >
            <Network className="h-4 w-4" />
            {showWorkflow ? 'Hide' : 'Show'} Workflow Diagram
            {showWorkflow ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
          
          {showWorkflow && (
            <div className="mt-3">
              {renderWorkflowDiagram(workflow_diagram)}
            </div>
          )}
        </div>
      )}

      <div className="mt-4 text-xs text-green-600">
        💡 These statistics help you understand the complexity, workflow planning, and cost of your document generation.
      </div>
    </div>
  );
}