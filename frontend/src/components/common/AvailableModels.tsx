import { useState } from 'react'

interface ModelItem {
  filename: string;
  created_at: string;
  config?: {
    optimizer?: string;
    loss?: string;
    layers?: Array<{
      type: string;
      units?: number;
      activation?: string;
      dropout?: number;
    }>;
    dropout_rate?: number;
    output_activation?: string;
    metrics?: string[];
  };
}

interface FolderItem {
  folder_name: string;
  created_at: string;
}

interface AvailableModelsProps {
  models: ModelItem[];
  modelsLoading: boolean;
  selectedModel: ModelItem | null;
  selectedModelToUse: string | null;
  selectedFolderToUse: string | null; // Dodaj
  hidsFolders: FolderItem[];
  nidsFolders: FolderItem[];
  currentDataSource: string;
  onRefresh: () => void;
  onSelectModel: (model: ModelItem) => void;
  onSelectFolder: (folder: FolderItem) => void; // Dodaj
  onViewDetails: (model: ModelItem) => void;
}

export default function AvailableModels({
  models,
  modelsLoading,
  selectedModel,
  selectedModelToUse,
  selectedFolderToUse, // Dodaj
  hidsFolders,
  nidsFolders,
  currentDataSource,
  onRefresh,
  onSelectModel,
  onSelectFolder, // Dodaj
  onViewDetails
}: AvailableModelsProps) {
  const [showRawJson, setShowRawJson] = useState(false);
  const [activeTab, setActiveTab] = useState<'models' | 'folders'>('models');
  const [selectedFolder, setSelectedFolder] = useState<FolderItem | null>(null);
  const [folderMetadata, setFolderMetadata] = useState<any>(null);
  const [loadingMetadata, setLoadingMetadata] = useState(false);

  // Get the correct folders based on the selected data source
  const folders = currentDataSource === 'hids' ? hidsFolders : nidsFolders;

  const fetchFolderMetadata = async (folderName: string) => {
  setLoadingMetadata(true);
  try {
    const response = await fetch(`http://localhost:8012/training/folder-metadata/${currentDataSource}/${folderName}`);
    if (response.ok) {
      const data = await response.json();
      setFolderMetadata(data);
    } else {
      console.error('Error fetching folder metadata');
    }
  } catch (error) {
    console.error('Error:', error);
  } finally {
    setLoadingMetadata(false);
  }
  };

  const handleSelectFolder = (folder: FolderItem) => {
    onSelectFolder(folder); // Użyj funkcji przekazanej z rodzica
  };

  const handleViewFolderDetails = async (folder: FolderItem) => {
    setSelectedFolder(folder);
    if (folder) {
      await fetchFolderMetadata(folder.folder_name);
    } else {
      setFolderMetadata(null);
    }
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 md:w-1/2 flex flex-col">
      <div className="flex justify-between items-center mb-3">
        <h2 className="text-xl font-bold">Available Models</h2>
        <button 
          onClick={onRefresh}
          className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 flex items-center"
        >
          <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>
      
      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700 mb-4">
        <nav className="-mb-px flex space-x-4">
          <button
            onClick={() => setActiveTab('models')}
            className={`py-2 px-1 border-b-2 text-sm font-medium ${
              activeTab === 'models'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            Models
          </button>
          <button
            onClick={() => setActiveTab('folders')}
            className={`py-2 px-1 border-b-2 text-sm font-medium ${
              activeTab === 'folders'
                ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
            }`}
          >
            {currentDataSource.toUpperCase()} Folders
          </button>
        </nav>
      </div>
      
      {/* Models Tab Content */}
      {activeTab === 'models' && (
        <>
          <p className="text-gray-600 dark:text-gray-400 mb-3 text-sm">View and inspect trained models from the data/models directory</p>
          
          {modelsLoading ? (
            <div className="flex justify-center items-center h-40">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
              <span className="ml-2 text-gray-600 dark:text-gray-400">Loading models...</span>
            </div>
          ) : models.length === 0 ? (
            <div className="text-center py-8 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-gray-100">No models found</h3>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">No models are available in the data/models directory.</p>
            </div>
          ) : (
            <div className="overflow-y-auto max-h-[700px] pr-1">
              <div className="space-y-3">
                {models.map((model, index) => (
                  <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 transition-all hover:shadow-md">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">{model.filename.replace('.json', '')}</h3>
                        <p className="text-xs text-gray-500 dark:text-gray-400">Created: {model.created_at}</p>
                        {selectedModelToUse === model.filename && (
                          <span className="inline-flex items-center px-2 py-0.5 mt-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">
                            <svg className="mr-1 h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                            </svg>
                            Selected as base
                          </span>
                        )}
                      </div>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => onSelectModel(model)}
                          className={`${
                            selectedModelToUse === model.filename 
                              ? 'bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-300' 
                              : 'bg-gray-100 dark:bg-gray-600 text-gray-600 dark:text-gray-300'
                          } px-3 py-1 rounded-md text-xs font-medium hover:bg-green-200 dark:hover:bg-green-800 transition-colors`}
                        >
                          {selectedModelToUse === model.filename ? 'Selected' : 'Select Model'}
                        </button>
                        <button
                          onClick={() => onViewDetails(model)}
                          className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 px-3 py-1 rounded-md text-xs font-medium hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
                        >
                          View Details
                        </button>
                      </div>
                    </div>
                    
                    {selectedModel && selectedModel.filename === model.filename && (
                      <div className="mt-3 bg-white dark:bg-gray-800 rounded-md p-3 border border-gray-200 dark:border-gray-600">
                        <h4 className="text-xs font-semibold mb-2 text-gray-700 dark:text-gray-300">Model Architecture</h4>
                        {model.config ? (
                          <div className="mb-4">
                            {/* Model visualization - Changed to horizontal layout */}
                            <div className="overflow-x-auto">
                              <div className="flex items-center p-2">
                                <div className="text-center text-xs text-gray-600 dark:text-gray-400 mb-2 mr-4">
                                  <span className="font-medium">Optimizer:</span> {model.config.optimizer || 'N/A'} | 
                                  <span className="font-medium ml-2">Loss Function:</span> {model.config.loss || 'N/A'}
                                </div>
                                
                                {/* Network visualization - Horizontal layout */}
                                <div className="flex items-center">
                                  {/* Input Layer */}
                                  <div className="flex flex-col items-center mx-2">
                                    <div className="w-24 py-2 bg-blue-100 dark:bg-blue-900 text-center rounded-lg text-xs font-medium border border-blue-200 dark:border-blue-800">
                                      Input Layer
                                    </div>
                                  </div>
                                  
                                  {/* Connection Line */}
                                  <div className="w-8 h-1 bg-gray-300 dark:bg-gray-600"></div>
                                  
                                  {/* Layers visualization */}
                                  {model.config.layers && model.config.layers.map((layer, i) => (
                                    <div key={i} className="flex items-center">
                                      <div className={`w-32 py-2 text-center rounded-lg text-xs font-medium border mx-2
                                        ${layer.type === 'dense' ? 'bg-green-100 dark:bg-green-900 border-green-200 dark:border-green-800' : 
                                         layer.type === 'dropout' ? 'bg-red-100 dark:bg-red-900 border-red-200 dark:border-red-800' : 
                                         'bg-purple-100 dark:bg-purple-900 border-purple-200 dark:border-purple-800'}`}>
                                        <div className="font-semibold capitalize">{layer.type}</div>
                                        {layer.type === 'dense' && 
                                          <div className="text-xs opacity-75">
                                            Units: {layer.units} | {layer.activation}
                                          </div>
                                        }
                                        {layer.type === 'dropout' && 
                                          <div className="text-xs opacity-75">
                                            Rate: {layer.dropout || model.config.dropout_rate || 0}
                                          </div>
                                        }
                                      </div>
                                      
                                      {/* Connection Line */}
                                      {i < model.config.layers.length - 1 && (
                                        <div className="w-8 h-1 bg-gray-300 dark:bg-gray-600"></div>
                                      )}
                                    </div>
                                  ))}
                                  
                                  {/* Connection Line to Output */}
                                  <div className="w-8 h-1 bg-gray-300 dark:bg-gray-600"></div>
                                  
                                  {/* Output Layer */}
                                  <div className="flex flex-col items-center mx-2">
                                    <div className="w-24 py-2 bg-yellow-100 dark:bg-yellow-900 text-center rounded-lg text-xs font-medium border border-yellow-200 dark:border-yellow-800">
                                      Output Layer
                                      <div className="text-xs opacity-75">
                                        {model.config.output_activation || 'linear'}
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            {/* Metrics */}
                            {model.config.metrics && model.config.metrics.length > 0 && (
                              <div className="mt-3 w-full">
                                <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Metrics:</div>
                                <div className="flex flex-wrap gap-1">
                                  {model.config.metrics.map((metric, i) => (
                                    <span key={i} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-300">
                                      {metric}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {/* Toggle to show raw JSON */}
                            <div className="mt-2 text-right">
                              <button 
                                type="button"
                                onClick={() => setShowRawJson(prev => !prev)}
                                className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                              >
                                {showRawJson ? 'Hide Raw JSON' : 'Show Raw JSON'}
                              </button>
                            </div>
                            
                            {showRawJson && (
                              <pre className="mt-2 text-xs overflow-x-auto bg-gray-50 dark:bg-gray-900 p-2 rounded max-h-40">
                                {JSON.stringify(model.config, null, 2)}
                              </pre>
                            )}
                          </div>
                        ) : (
                          <p className="text-xs text-gray-500 dark:text-gray-400">No configuration available for this model.</p>
                        )}
                        <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                          <button
                            onClick={() => onViewDetails(null)}
                            className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                          >
                            Close details
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
      
      {/* Folders Tab Content */}
      {activeTab === 'folders' && (
        <>
          <p className="text-gray-600 dark:text-gray-400 mb-3 text-sm">
            Available {currentDataSource.toUpperCase()} folders for model training
          </p>
          
          {folders.length === 0 ? (
            <div className="text-center py-8 bg-gray-50 dark:bg-gray-700 rounded-lg">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
              </svg>
              <h3 className="mt-2 text-sm font-medium text-gray-900 dark:text-gray-100">No folders found</h3>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                No {currentDataSource.toUpperCase()} folders are available for training.
              </p>
            </div>
          ) : (
            <div className="overflow-y-auto max-h-[700px] pr-1">
              <div className="space-y-2">
                {folders.map((folder, index) => (
                  <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 transition-all hover:shadow-md">
                    <div className="flex justify-between items-start">
                      <div className="flex items-center flex-1">
                        <svg className="h-5 w-5 text-yellow-500 mr-2" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                          <path fillRule="evenodd" d="M2 6a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1H8a3 3 0 00-3 3v1.5a1.5 1.5 0 01-3 0V6z" clipRule="evenodd" />
                          <path d="M6 12a2 2 0 012-2h8a2 2 0 012 2v2a2 2 0 01-2 2H2h2a2 2 0 002-2v-2z" />
                        </svg>
                        <div className="flex flex-col">
                          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            {folder.folder_name}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            Created: {folder.created_at}
                          </span>
                          {selectedFolderToUse === folder.folder_name && (
                            <span className="inline-flex items-center px-2 py-0.5 mt-1 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300">
                              <svg className="mr-1 h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                              </svg>
                              Selected for retrain
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleSelectFolder(folder)}
                          className={`${
                            selectedFolderToUse === folder.folder_name 
                              ? 'bg-green-100 dark:bg-green-900 text-green-600 dark:text-green-300' 
                              : 'bg-gray-100 dark:bg-gray-600 text-gray-600 dark:text-gray-300'
                          } px-3 py-1 rounded-md text-xs font-medium hover:bg-green-200 dark:hover:bg-green-800 transition-colors`}
                        >
                          {selectedFolderToUse === folder.folder_name ? 'Selected' : 'Select for Retrain'}
                        </button>
                        <button
                          onClick={() => handleViewFolderDetails(folder)}
                          className="bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300 px-3 py-1 rounded-md text-xs font-medium hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
                        >
                          View Details
                        </button>
                      </div>
                    </div>
                    
                    {selectedFolder && selectedFolder.folder_name === folder.folder_name && (
                      <div className="mt-3 bg-white dark:bg-gray-800 rounded-md p-3 border border-gray-200 dark:border-gray-600">
                        {loadingMetadata ? (
                          <div className="flex justify-center items-center py-4">
                            <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-blue-500"></div>
                            <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">Loading metadata...</span>
                          </div>
                        ) : folderMetadata ? (
                          <div>
                            <h4 className="text-xs font-semibold mb-2 text-gray-700 dark:text-gray-300">Training Metadata</h4>
                            
                            {/* Training Info */}
                            <div className="grid grid-cols-2 gap-2 mb-3 text-xs">
                              <div>
                                <span className="font-medium text-gray-600 dark:text-gray-400">Training Date:</span>
                                <span className="ml-1">{folderMetadata.training_date}</span>
                              </div>
                              <div>
                                <span className="font-medium text-gray-600 dark:text-gray-400">Data Shape:</span>
                                <span className="ml-1">{folderMetadata.data_shape?.join(' x ')}</span>
                              </div>
                              <div>
                                <span className="font-medium text-gray-600 dark:text-gray-400">Epochs:</span>
                                <span className="ml-1">{folderMetadata.epochs}</span>
                              </div>
                              <div>
                                <span className="font-medium text-gray-600 dark:text-gray-400">Batch Size:</span>
                                <span className="ml-1">{folderMetadata.batch_size}</span>
                              </div>
                              <div>
                                <span className="font-medium text-gray-600 dark:text-gray-400">Final Threshold:</span>
                                <span className="ml-1">{folderMetadata.final_threshold?.toFixed(6)}</span>
                              </div>
                              <div>
                                <span className="font-medium text-gray-600 dark:text-gray-400">Base Model:</span>
                                <span className="ml-1">{folderMetadata.base_model_used || 'None'}</span>
                              </div>
                            </div>
                            
                            {/* Columns Used */}
                            {folderMetadata.columns_used && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Columns Used:</div>
                                <div className="flex flex-wrap gap-1">
                                  {folderMetadata.columns_used.map((col, i) => (
                                    <span key={i} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300">
                                      {col}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {/* Anomaly Statistics */}
                            {folderMetadata.anomaly_statistics && (
                              <div className="mb-3">
                                <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">Anomaly Statistics:</div>
                                <div className="bg-gray-50 dark:bg-gray-900 rounded p-2 text-xs">
                                  <div className="grid grid-cols-2 gap-1">
                                    <div>Total Samples: {folderMetadata.anomaly_statistics.total_samples?.toLocaleString()}</div>
                                    <div>Anomalies: {folderMetadata.anomaly_statistics.anomaly_count} ({folderMetadata.anomaly_statistics.anomaly_percentage?.toFixed(2)}%)</div>
                                    <div>Normal: {folderMetadata.anomaly_statistics.normal_count}</div>
                                    <div>Normal %: {folderMetadata.anomaly_statistics.normal_percentage?.toFixed(2)}%</div>
                                  </div>
                                </div>
                              </div>
                            )}
                            
                            {/* Model Architecture (same visualization as for models) */}
                            {folderMetadata.model_config && (
                              <div>
                                <h4 className="text-xs font-semibold mb-2 text-gray-700 dark:text-gray-300">Model Architecture</h4>
                                <div className="overflow-x-auto">
                                  <div className="flex items-center p-2">
                                    <div className="text-center text-xs text-gray-600 dark:text-gray-400 mb-2 mr-4">
                                      <span className="font-medium">Optimizer:</span> {folderMetadata.model_config.optimizer || 'N/A'} | 
                                      <span className="font-medium ml-2">Loss Function:</span> {folderMetadata.model_config.loss || 'N/A'}
                                    </div>
                                  </div>
                                  
                                  {/* Use the same horizontal layout visualization as for models */}
                                  {/* Copy the network visualization code from the models section */}
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <p className="text-xs text-gray-500 dark:text-gray-400">No metadata available for this folder.</p>
                        )}
                        
                        <div className="mt-2 pt-2 border-t border-gray-200 dark:border-gray-700">
                          <button
                            onClick={() => handleViewFolderDetails(null)}
                            className="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                          >
                            Close details
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}