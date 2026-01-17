import { useState, useEffect } from 'react'

function formatDateForInput(date: Date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}T${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

interface ConfigureModelTrainingProps {
  columnPresets: {
    hids: string;
    nids: string;
  };
  loading: boolean;
  selectedModelToUse: string | null;
  selectedFolderToUse: string | null; // Dodaj
  initialDataSource?: string;
  onSubmit: (formData: any) => void;
  onDataSourceChange?: (source: string) => void;
}

export default function ConfigureModelTraining({ 
  columnPresets, 
  loading, 
  selectedModelToUse,
  selectedFolderToUse, // Dodaj
  initialDataSource = 'hids', 
  onSubmit,
  onDataSourceChange 
}: ConfigureModelTrainingProps) {
  // Define explicit type for form state
  const [formData, setFormData] = useState({
    source: initialDataSource, // Use the initial data source
    batch_size: '50', // Store as string for form input compatibility
    columns_to_use: columnPresets[initialDataSource], // Default for initial source
    relative_from: '',
    relative_to: '',
    epochs: '50', // Store as string for form input compatibility
    max_size: '10000' // Store as string for form input compatibility
  });

  // Set default dates and check URL parameters
  useEffect(() => {
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    // Check if there's a source parameter in the URL
    const urlParams = new URLSearchParams(window.location.search);
    const sourceParam = urlParams.get('source');
    
    const newSource = sourceParam === 'nids' ? 'nids' : initialDataSource;
    
    setFormData(prev => ({
      ...prev,
      source: newSource,
      columns_to_use: columnPresets[newSource],
      relative_from: formatDateForInput(yesterday),
      relative_to: formatDateForInput(today)
    }));
    
    // Notify parent of the initial data source
    if (onDataSourceChange) {
      onDataSourceChange(newSource);
    }
  }, [columnPresets, initialDataSource, onDataSourceChange]);

  const handleInputChange = (e: { target: { name: any; value: any; }; }) => {
    const { name, value } = e.target;
    
    // If changing data source, update both source and columns
    if (name === 'source') {
      // Update the source and columns directly without page reload
      setFormData(prev => ({
        ...prev,
        source: value,
        columns_to_use: columnPresets[value]
      }));
      
      // Notify parent component of the data source change
      if (onDataSourceChange) {
        onDataSourceChange(value);
      }
    } else {
      // For all other inputs, just update the value
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  const handleSubmit = (e: { preventDefault: () => void; }) => {
    e.preventDefault();
    
    // Format columns array from string
    const columnsArray = formData.columns_to_use.split(',').map(col => col.trim());
    
    const dataToSubmit = {
      source: formData.source,
      columns_to_use: columnsArray,
      relative_from: formData.relative_from,
      relative_to: formData.relative_to,
      batch_size: parseInt(formData.batch_size, 10),
      epochs: parseInt(formData.epochs, 10),
      max_size: parseInt(formData.max_size, 10),
      basemodel: selectedModelToUse, // Model config do train
      basefolder: selectedFolderToUse // Folder do retrain
    };
    
    console.log('Submitting:', dataToSubmit);
    onSubmit(dataToSubmit);
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-4 mb-4 md:w-1/2">
      <h2 className="text-xl font-bold mb-1">Configure Model Training</h2>
      <p className="text-gray-600 dark:text-gray-400 mb-3 text-sm">Set up training parameters for anomaly detection models</p>
    
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Data Source Selection Card */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
          <h3 className="text-base font-semibold mb-2">Data Source</h3>
          <div className="flex flex-wrap gap-2">
            <label className={`flex items-center px-3 py-2 rounded-lg border cursor-pointer ${formData.source === 'hids' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30' : 'border-gray-200 dark:border-gray-600'}`}>
              <input
                type="radio"
                name="source"
                value="hids"
                checked={formData.source === 'hids'}
                onChange={handleInputChange}
                className="sr-only" // Hide the actual radio input
              />
              <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center mr-3 ${formData.source === 'hids' ? 'border-blue-500' : 'border-gray-400'}`}>
                {formData.source === 'hids' && <div className="w-3 h-3 rounded-full bg-blue-500"></div>}
              </div>
              <div>
                <span className="font-medium block">HIDS</span>
              </div>
            </label>
            
            <label className={`flex items-center px-4 py-3 rounded-lg border cursor-pointer ${formData.source === 'nids' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30' : 'border-gray-200 dark:border-gray-600'}`}>
              <input
                type="radio"
                name="source"
                value="nids"
                checked={formData.source === 'nids'}
                onChange={handleInputChange}
                className="sr-only" // Hide the actual radio input
              />
              <div className={`w-5 h-5 rounded-full border-2 flex items-center justify-center mr-3 ${formData.source === 'nids' ? 'border-blue-500' : 'border-gray-400'}`}>
                {formData.source === 'nids' && <div className="w-3 h-3 rounded-full bg-blue-500"></div>}
              </div>
              <div>
                <span className="font-medium block">NIDS</span>
              </div>
            </label>
          </div>
        </div>
        
        {/* Time Range Selection Card */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
          <h3 className="text-base font-semibold mb-1">Time Range</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Select the time period from which to collect training data</p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Start Date and Time</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                  <svg className="w-5 h-5 text-gray-500 dark:text-gray-400" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                    <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd"></path>
                  </svg>
                </div>
                <input
                  type="datetime-local"
                  name="relative_from"
                  value={formData.relative_from}
                  onChange={handleInputChange}
                  className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 p-2.5"
                  required
                />
              </div>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">End Date and Time</label>
              <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                  <svg className="w-5 h-5 text-gray-500 dark:text-gray-400" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                    <path fillRule="evenodd" d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z" clipRule="evenodd"></path>
                  </svg>
                </div>
                <input
                  type="datetime-local"
                  name="relative_to"
                  value={formData.relative_to}
                  onChange={handleInputChange}
                  className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 p-2.5"
                  required
                />
              </div>
            </div>
          </div>
        </div>
        
        {/* Model Parameters Card */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
          <h3 className="text-base font-semibold mb-2">Model Parameters</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Epochs</label>
              <div className="relative mt-1 rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                  <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                    <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                  </svg>
                </div>
                <input
                  type="number"
                  name="epochs"
                  value={formData.epochs}
                  onChange={handleInputChange}
                  min="1"
                  max="1000"
                  className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 block w-full rounded-md pl-10 p-2.5 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  required
                />
              </div>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Number of training cycles</p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Batch Size</label>
              <div className="relative mt-1 rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                  <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
                  </svg>
                </div>
                <input
                  type="number"
                  name="batch_size"
                  value={formData.batch_size}
                  onChange={handleInputChange}
                  min="1"
                  max="1000"
                  className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 block w-full rounded-md pl-10 p-2.5 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  required
                />
              </div>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Samples per training update</p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Maximum Records</label>
              <div className="relative mt-1 rounded-md shadow-sm">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                  <svg className="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                  </svg>
                </div>
                <input
                  type="number"
                  name="max_size"
                  value={formData.max_size}
                  onChange={handleInputChange}
                  min="100"
                  className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 block w-full rounded-md pl-10 p-2.5 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  required
                />
              </div>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">Maximum entries to process</p>
            </div>
          </div>
        </div>
        
        {/* Columns Selection Card */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
          <h3 className="text-base font-semibold mb-1">Columns to Use</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">Specify which data columns to include in the training process (comma-separated)</p>
          
          <div className="mt-1">
            <textarea
              name="columns_to_use"
              value={formData.columns_to_use}
              onChange={handleInputChange}
              rows={2}
              className="bg-white dark:bg-gray-800 shadow-sm block w-full focus:ring-blue-500 focus:border-blue-500 sm:text-sm border border-gray-300 dark:border-gray-600 rounded-md p-2"
              placeholder="Enter column names separated by commas"
              required
            />
          </div>
          <div className="mt-3">
            <div className="flex flex-wrap gap-2">
              {formData.columns_to_use.split(',').map((column, index) => (
                column.trim() && (
                  <span key={index} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300">
                    {column.trim()}
                  </span>
                )
              ))}
            </div>
          </div>
        </div>
        
        {/* Submit Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white ${loading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'} focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors duration-200`}
          >
            {loading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Starting Training...
              </>
            ) : (
              <>
                <svg className="mr-2 -ml-1 w-5 h-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-8.707l-3-3a1 1 0 00-1.414 0l-3 3a1 1 0 001.414 1.414L9 9.414V13a1 1 0 102 0V9.414l1.293 1.293a1 1 0 001.414-1.414z" clipRule="evenodd" />
                </svg>
                Start Model Training
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}