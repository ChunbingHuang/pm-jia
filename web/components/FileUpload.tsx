'use client';

import { useSession } from '@/contexts/SessionContext';
import { apiService } from '@/services/api';
import { CheckCircle, File, Upload } from 'lucide-react';
import React, { useRef, useState } from 'react';

export default function FileUpload() {
  const { currentSession, uploadedMaterials, addUploadedMaterials, setError } = useSession();
  
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const supportedTypes = [
    'txt', 'md', 'json', 'csv', 'xlsx', 
    'jpg', 'jpeg', 'png', 'pdf'
  ];

  const handleFileSelect = (files: FileList | null) => {
    if (!files || !currentSession) return;

    const validFiles = Array.from(files).filter(file => {
      const extension = file.name.split('.').pop()?.toLowerCase();
      return extension && supportedTypes.includes(extension);
    });

    if (validFiles.length !== files.length) {
      setError('Some files were skipped due to unsupported file types');
    }

    if (validFiles.length > 0) {
      uploadFiles(validFiles);
    }
  };

  const uploadFiles = async (files: File[]) => {
    if (!currentSession) return;

    setUploading(true);
    setError(null);

    try {
      // Create FileList-like object
      const fileList = Object.assign(files, {
        item: (index: number) => files[index] || null,
        length: files.length,
      }) as unknown as FileList;

      await apiService.uploadFiles(currentSession.session_id, fileList);
      
      // This is a mock - the actual API would return processed content
      const newMaterials: { [key: string]: string } = {};
      files.forEach(file => {
        newMaterials[file.name] = `Processed content from ${file.name}`;
      });
      
      addUploadedMaterials(newMaterials);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Failed to upload files');
    } finally {
      setUploading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(e.target.files);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };


  if (!currentSession) {
    return (
      <div className="p-4 bg-gray-50 border-l border-gray-200">
        <div className="text-center">
          <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
          <p className="text-sm text-gray-500">
            Select a session to upload files
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 bg-gray-50 border-l border-gray-200 w-80 h-full overflow-y-auto">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Materials</h3>

      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
          dragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".txt,.md,.json,.csv,.xlsx,.jpg,.jpeg,.png,.pdf"
          onChange={handleInputChange}
          className="hidden"
          disabled={uploading}
        />

        <Upload className="h-8 w-8 text-gray-400 mx-auto mb-2" />
        
        {uploading ? (
          <div>
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mx-auto mb-2"></div>
            <p className="text-sm text-gray-600">Processing files...</p>
          </div>
        ) : (
          <div>
            <p className="text-sm text-gray-600 mb-2">
              Drag & drop files here, or{' '}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                browse
              </button>
            </p>
            <p className="text-xs text-gray-500">
              Supports: {supportedTypes.join(', ')}
            </p>
          </div>
        )}
      </div>

      {/* Uploaded Files List */}
      {Object.keys(uploadedMaterials).length > 0 && (
        <div className="mt-4">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Uploaded Files</h4>
          <div className="space-y-2">
            {Object.entries(uploadedMaterials).map(([filename]) => (
              <div
                key={filename}
                className="flex items-center gap-2 p-2 bg-white border border-gray-200 rounded-lg"
              >
                <File className="h-4 w-4 text-gray-400 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {filename}
                  </p>
                  <p className="text-xs text-gray-500">
                    Processed successfully
                  </p>
                </div>
                <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* File Type Info */}
      <div className="mt-6 p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <h4 className="text-sm font-medium text-blue-900 mb-2">Supported File Types</h4>
        <div className="space-y-1 text-xs text-blue-800">
          <div><strong>Documents:</strong> TXT, MD, JSON</div>
          <div><strong>Data:</strong> CSV, XLSX</div>
          <div><strong>Images:</strong> JPG, PNG</div>
          <div><strong>PDFs:</strong> PDF</div>
        </div>
        <div className="mt-2 text-xs text-blue-600">
          Files are processed with OCR and content extraction for context.
        </div>
      </div>

      {/* Upload Tips */}
      <div className="mt-4 p-3 bg-gray-100 rounded-lg">
        <h4 className="text-sm font-medium text-gray-900 mb-2">💡 Tips</h4>
        <ul className="text-xs text-gray-600 space-y-1">
          <li>• Upload relevant documents to provide context</li>
          <li>• Images and PDFs will be processed with OCR</li>
          <li>• Multiple files can be uploaded at once</li>
          <li>• Maximum file size: 2MB per file</li>
        </ul>
      </div>
    </div>
  );
}