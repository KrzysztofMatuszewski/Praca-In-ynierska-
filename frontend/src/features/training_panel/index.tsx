import { useState, useEffect } from 'react'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import ConfigureModelTraining from '@/components/common/ConfigureModelTraining'
import AvailableModels from '@/components/common/AvailableModels'

// Define column presets for different sources
const columnPresets = {
  hids: 'agent.id, rule.description',
  nids: 'destination.port, source.port, destination.ip, source.ip'
};

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

export default function ModelTraining() {
  const [loading, setLoading] = useState(false);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [notification, setNotification] = useState({ show: false, type: '', message: '' });
  const [isTraining, setIsTraining] = useState(false);
  const [models, setModels] = useState<ModelItem[]>([]);
  const [hidsFolders, setHidsFolders] = useState<FolderItem[]>([]); 
  const [nidsFolders, setNidsFolders] = useState<FolderItem[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelItem | null>(null);
  const [selectedModelToUse, setSelectedModelToUse] = useState<string | null>(null);
  const [selectedFolderToUse, setSelectedFolderToUse] = useState<string | null>(null);
  const [currentDataSource, setCurrentDataSource] = useState('hids'); // Track the current data source
  
  // Function to fetch models from API
  const fetchModels = async () => {
    setModelsLoading(true);
    try {
      const response = await fetch('http://localhost:8012/training/models/list');
      if (response.ok) {
        const data = await response.json();
        setModels(data.models || []);
        setHidsFolders(data.hids_folders || []);
        setNidsFolders(data.nids_folders || []);
      } else {
        console.error('Error fetching models');
        setNotification({
          show: true,
          type: 'error',
          message: 'Failed to fetch models. Please try again.'
        });
      }
    } catch (error) {
      console.error('Error:', error);
      setNotification({
        show: true,
        type: 'error',
        message: 'Network error while fetching models. Please check your connection.'
      });
    } finally {
      setModelsLoading(false);
    }
  };

  // Set default dates and check URL parameters
  useEffect(() => {
    // Fetch available models
    fetchModels();
    
    // Check if there's a source parameter in the URL
    const urlParams = new URLSearchParams(window.location.search);
    const sourceParam = urlParams.get('source');
    
    // Update current data source based on URL parameter
    if (sourceParam === 'nids') {
      setCurrentDataSource('nids');
    }
  }, []);

  // Handle form submission from the ConfigureModelTraining component
  const handleSubmit = async (formData: any) => {
    setLoading(true);
    setIsTraining(true);
    
    try {
      const response = await fetch('http://localhost:8012/training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (response.ok) {
        const result = await response.json();
        setNotification({
          show: true,
          type: 'success',
          message: `Training started successfully! Directory: ${result.message.dir_name}`
        });
        // Refresh models list after successful training
        fetchModels();
      } else {
        setNotification({
          show: true,
          type: 'error',
          message: 'Error starting training. Please try again.'
        });
      }
    } catch (error) {
      console.error('Error:', error);
      setNotification({
        show: true,
        type: 'error',
        message: 'Network error. Please check your connection.'
      });
    } finally {
      setLoading(false);
      setIsTraining(false);
    }
  };

  // Handler for model selection (for use in training)
  const selectModelForTraining = (model) => {
    if (selectedModelToUse === model.filename) {
      setSelectedModelToUse(null);
    } else {
      setSelectedModelToUse(model.filename);
      setSelectedFolderToUse(null); // Resetuj wybór folderu
    }
  };

  const selectFolderForTraining = (folder) => {
    if (selectedFolderToUse === folder.folder_name) {
      setSelectedFolderToUse(null);
    } else {
      setSelectedFolderToUse(folder.folder_name);
      setSelectedModelToUse(null); // Resetuj wybór modelu
    }
  };

  // Handler for viewing model details
  const handleViewModel = (model) => {
    setSelectedModel(model);
  };
  
  // Handler for updating the current data source when changed in ConfigureModelTraining
  const handleDataSourceChange = (source) => {
    setCurrentDataSource(source);
  };

  return (
    <>
      {isTraining && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-xl max-w-md w-full flex flex-col items-center">
            <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500 mb-4"></div>
            <h3 className="text-lg font-semibold mb-2">Training Model</h3>
            <p className="text-gray-600 dark:text-gray-400 text-center">
              Please wait while the model is being trained. This may take several minutes depending on your configuration.
            </p>
            <div className="w-full bg-gray-200 rounded-full h-2.5 mt-4">
              <div className="bg-blue-600 h-2.5 rounded-full animate-pulse w-3/4"></div>
            </div>
          </div>
        </div>
      )}
      <Header>
        <div className="flex items-center space-x-2">
          <div className="bg-blue-500 text-white p-2 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <h1 className="text-xl font-bold">Model Training</h1>
        </div>
      </Header>
      <Main fixed>
        <div className="max-w-7xl mx-auto p-3">
          {notification.show && (
            <div className={`mb-3 p-2 rounded-lg ${notification.type === 'success' ? 'bg-green-50 border border-green-200 text-green-800' : 'bg-red-50 border border-red-200 text-red-800'}`}>
              <div className="flex">
                <div className="flex-shrink-0">
                  {notification.type === 'success' ? (
                    <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
                <div className="ml-2">
                  <p className="text-xs font-medium">{notification.message}</p>
                </div>
              </div>
            </div>
          )}

          <div className="flex flex-col md:flex-row gap-4">
            {/* Training form component */}
            <ConfigureModelTraining 
              columnPresets={columnPresets}
              loading={loading}
              selectedModelToUse={selectedModelToUse}
              selectedFolderToUse={selectedFolderToUse} // Dodaj
              onSubmit={handleSubmit}
              onDataSourceChange={handleDataSourceChange}
              initialDataSource={currentDataSource}
            />

            {/* Models panel component */}
            <AvailableModels 
              models={models}
              modelsLoading={modelsLoading}
              selectedModel={selectedModel}
              selectedModelToUse={selectedModelToUse}
              selectedFolderToUse={selectedFolderToUse} // Dodaj
              hidsFolders={hidsFolders}
              nidsFolders={nidsFolders}
              currentDataSource={currentDataSource}
              onRefresh={fetchModels}
              onSelectModel={selectModelForTraining}
              onSelectFolder={selectFolderForTraining} // Dodaj
              onViewDetails={handleViewModel}
            />
          </div>
        </div>
      </Main>
    </>
  );
}