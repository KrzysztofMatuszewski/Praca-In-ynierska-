// model_panel/components/NeuralNetworkForm.tsx
import React, { useState } from 'react';

interface Layer {
  type: string;
  units: number;
  activation: string;
  dropout: number;
}

// Dodaj pole name do interfejsu NeuralNetworkConfig
interface NeuralNetworkConfig {
  name: string;  // Nowe pole
  dropout_rate: number;
  activation: string;
  output_activation: string;
  optimizer: string;
  loss: string;
  metrics: string[];
  layers: Layer[];
}

interface NeuralNetworkFormProps {
  onConfigChange?: (config: NeuralNetworkConfig) => void;
}

const NeuralNetworkForm: React.FC<NeuralNetworkFormProps> = ({ onConfigChange }) => {
// Dodaj name do stanu początkowego
const [networkConfig, setNetworkConfig] = useState<NeuralNetworkConfig>({
  name: '',  // Nowe pole z domyślną pustą wartością
  dropout_rate: 0.2,
  activation: 'relu',
  output_activation: 'linear',
  optimizer: 'adam',
  loss: 'mean_squared_error',
  metrics: ['accuracy'],
  layers: [
    { type: 'dense', units: 64, activation: 'relu', dropout: 0.2 },
  ]
});

  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  const activationOptions = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax'];
  const outputActivationOptions = ['linear', 'softmax', 'sigmoid', 'tanh'];
  const optimizerOptions = ['adam', 'sgd', 'rmsprop'];
  // const optimizerOptions = ['adam', 'sgd', 'rmsprop', 'adagrad'];
  const lossOptions = ['mean_squared_error'];
  // const lossOptions = ['mean_squared_error', 'categorical_crossentropy', 'binary_crossentropy', 'sparse_categorical_crossentropy'];
  const metricsOptions = ['accuracy'];
  // const metricsOptions = ['accuracy', 'precision', 'recall', 'f1_score', 'auc'];
  const layerTypeOptions = ['dense', 'conv1d', 'conv2d', 'lstm', 'gru'];

  const handleGeneralChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    const updatedConfig = {
      ...networkConfig,
      [name]: name === 'dropout_rate' ? parseFloat(value) : value
    };
    
    setNetworkConfig(updatedConfig);
    if (onConfigChange) {
      onConfigChange(updatedConfig);
    }
  };

  const handleMetricsChange = (metric: string) => {
    const updatedMetrics = [...networkConfig.metrics];
    const metricIndex = updatedMetrics.indexOf(metric);
    
    if (metricIndex > -1) {
      updatedMetrics.splice(metricIndex, 1);
    } else {
      updatedMetrics.push(metric);
    }
    
    const updatedConfig = {
      ...networkConfig,
      metrics: updatedMetrics
    };
    
    setNetworkConfig(updatedConfig);
    if (onConfigChange) {
      onConfigChange(updatedConfig);
    }
  };

  const addLayer = () => {
    const updatedConfig = {
      ...networkConfig,
      layers: [
        ...networkConfig.layers,
        { type: 'dense', units: 32, activation: 'relu', dropout: 0.2 }
      ]
    };
    
    setNetworkConfig(updatedConfig);
    if (onConfigChange) {
      onConfigChange(updatedConfig);
    }
  };

  const removeLayer = (index: number) => {
    const updatedLayers = [...networkConfig.layers];
    updatedLayers.splice(index, 1);
    
    const updatedConfig = {
      ...networkConfig,
      layers: updatedLayers
    };
    
    setNetworkConfig(updatedConfig);
    if (onConfigChange) {
      onConfigChange(updatedConfig);
    }
  };

  const updateLayer = (index: number, field: keyof Layer, value: string | number) => {
    const updatedLayers = [...networkConfig.layers];
    
    if (field === 'units') {
      updatedLayers[index] = {
        ...updatedLayers[index],
        [field]: parseInt(value as string)
      };
    } else if (field === 'dropout') {
      updatedLayers[index] = {
        ...updatedLayers[index],
        [field]: parseFloat(value as string)
      };
    } else {
      updatedLayers[index] = {
        ...updatedLayers[index],
        [field]: value
      };
    }
    
    const updatedConfig = {
      ...networkConfig,
      layers: updatedLayers
    };
    
    setNetworkConfig(updatedConfig);
    if (onConfigChange) {
      onConfigChange(updatedConfig);
    }
  };

  const moveLayer = (index: number, direction: 'up' | 'down') => {
    if (
      (direction === 'up' && index === 0) ||
      (direction === 'down' && index === networkConfig.layers.length - 1)
    ) {
      return;
    }

    const updatedLayers = [...networkConfig.layers];
    const newIndex = direction === 'up' ? index - 1 : index + 1;
    
    // Swap layers
    [updatedLayers[index], updatedLayers[newIndex]] = [updatedLayers[newIndex], updatedLayers[index]];
    
    const updatedConfig = {
      ...networkConfig,
      layers: updatedLayers
    };
    
    setNetworkConfig(updatedConfig);
    if (onConfigChange) {
      onConfigChange(updatedConfig);
    }
  };

  const duplicateLayer = (index: number) => {
    const layerToDuplicate = networkConfig.layers[index];
    const updatedLayers = [...networkConfig.layers];
    updatedLayers.splice(index + 1, 0, { ...layerToDuplicate });
    
    const updatedConfig = {
      ...networkConfig,
      layers: updatedLayers
    };
    
    setNetworkConfig(updatedConfig);
    if (onConfigChange) {
      onConfigChange(updatedConfig);
    }
  };

  const loadTemplate = (templateName: string) => {
    let template: NeuralNetworkConfig;
    
    switch (templateName) {
      case 'classification':
        template = {
          name: 'autoencoder-' + new Date().toISOString().replace('T', '_').substr(0, 19).replace(/:/g, '-'),
          dropout_rate: 0.2,
          activation: 'relu',
          output_activation: 'linear',
          optimizer: 'adam',
          loss: 'mean_squared_error',
          metrics: ['accuracy'],
          layers: [
            { type: 'dense', units: 128, activation: 'relu', dropout: 0.2 },
            { type: 'dense', units: 64, activation: 'relu', dropout: 0.2 },
            { type: 'dense', units: 32, activation: 'relu', dropout: 0.2 },
            { type: 'dense', units: 16, activation: 'relu', dropout: 0.2 },
            { type: 'dense', units: 32, activation: 'relu', dropout: 0.2 },
            { type: 'dense', units: 64, activation: 'relu', dropout: 0.2 },
            { type: 'dense', units: 128, activation: 'relu', dropout: 0.2 }
          ]
        };
      break;
      default:
        return;
    }
    
    setNetworkConfig(template);
    if (onConfigChange) {
      onConfigChange(template);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onConfigChange) {
      onConfigChange(networkConfig);
    }
  };

  // Style dla wybranej opcji w selectach
  const getOptionStyle = (option: string, selectedOption: string) => {
    const baseClasses = "px-4 py-2 cursor-pointer";
    return selectedOption === option 
      ? `${baseClasses} bg-indigo-100 text-indigo-800` 
      : `${baseClasses} hover:bg-gray-100`;
  };

  return (
    <div className="w-full bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      <form onSubmit={handleSubmit} className="p-5">
        {/* Templates */}
        <div className="mb-6">
          <h2 className="text-xl font-semibold mb-3 text-gray-800">Network template</h2>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => loadTemplate('classification')}
              className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors shadow-sm"
            >
              Autoencoder
            </button>
          </div>
        </div>

        <div className="mb-6 bg-gray-50 p-5 rounded-lg border border-gray-200">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-800">General settings</h2>
            <button
              type="button"
              onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
              className="text-indigo-600 text-sm font-medium hover:text-indigo-800 transition-colors flex items-center"
            >
              {showAdvancedOptions ? (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                  </svg>
                  Hide advanced
                </>
              ) : (
                <>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                  Show advanced
                </>
              )}
            </button>
          </div>

          <div className="relative">
            <label className="block mb-2 font-medium text-gray-700">
              Network Name:
            </label>
            <input
              type="text"
              name="name"
              value={networkConfig.name}
              onChange={handleGeneralChange}
              className="w-full p-2.5 border border-gray-300 rounded-lg text-gray-700 bg-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 appearance-none shadow-sm"
              placeholder="Enter network name"
              required
            />
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="relative">
              <label className="block mb-2 font-medium text-gray-700">
                Default activation:
              </label>
              <div className="relative">
                <select
                  name="activation"
                  value={networkConfig.activation}
                  onChange={handleGeneralChange}
                  className="w-full p-2.5 border border-gray-300 rounded-lg text-gray-700 bg-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 appearance-none shadow-sm"
                >
                  {activationOptions.map(option => (
                    <option key={option} value={option} className={getOptionStyle(option, networkConfig.activation)}>
                      {option}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                  <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                    <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
                  </svg>
                </div>
              </div>
            </div>
            
            <div className="relative">
              <label className="block mb-2 font-medium text-gray-700">
                Activation of the output layer:
              </label>
              <div className="relative">
                <select
                  name="output_activation"
                  value={networkConfig.output_activation}
                  onChange={handleGeneralChange}
                  className="w-full p-2.5 border border-gray-300 rounded-lg text-gray-700 bg-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 appearance-none shadow-sm"
                >
                  {outputActivationOptions.map(option => (
                    <option key={option} value={option} className={getOptionStyle(option, networkConfig.output_activation)}>
                      {option}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                  <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                    <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
                  </svg>
                </div>
              </div>
            </div>
            
            <div className="relative">
              <label className="block mb-2 font-medium text-gray-700">
                Optimizer:
              </label>
              <div className="relative">
                <select
                  name="optimizer"
                  value={networkConfig.optimizer}
                  onChange={handleGeneralChange}
                  className="w-full p-2.5 border border-gray-300 rounded-lg text-gray-700 bg-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 appearance-none shadow-sm"
                >
                  {optimizerOptions.map(option => (
                    <option key={option} value={option} className={getOptionStyle(option, networkConfig.optimizer)}>
                      {option}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                  <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                    <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
                  </svg>
                </div>
              </div>
            </div>
            
            <div className="relative">
              <label className="block mb-2 font-medium text-gray-700">
                Loss function:
              </label>
              <div className="relative">
                <select
                  name="loss"
                  value={networkConfig.loss}
                  onChange={handleGeneralChange}
                  className="w-full p-2.5 border border-gray-300 rounded-lg text-gray-700 bg-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 appearance-none shadow-sm"
                >
                  {lossOptions.map(option => (
                    <option key={option} value={option} className={getOptionStyle(option, networkConfig.loss)}>
                      {option}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                  <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                    <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/>
                  </svg>
                </div>
              </div>
            </div>
            
            {showAdvancedOptions && (
              <div>
                <label className="block mb-2 font-medium text-gray-700">
                  Global Dropout Rate:
                </label>
                <div className="flex items-center">
                  <input
                    type="range"
                    name="dropout_rate"
                    value={networkConfig.dropout_rate}
                    onChange={handleGeneralChange}
                    step="0.1"
                    min="0"
                    max="0.9"
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />
                  <span className="ml-3 bg-indigo-100 text-indigo-800 px-2 py-1 rounded-md min-w-10 text-center">
                    {networkConfig.dropout_rate}
                  </span>
                </div>
              </div>
            )}

            {showAdvancedOptions && (
              <div>
                <label className="block mb-2 font-medium text-gray-700">Metrics:</label>
                <div className="flex flex-wrap gap-2">
                  {metricsOptions.map(metric => (
                    <label key={metric} className="flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={networkConfig.metrics.includes(metric)}
                        onChange={() => handleMetricsChange(metric)}
                        className="form-checkbox h-5 w-5 text-indigo-600 transition duration-150 ease-in-out rounded border border-gray-300 focus:ring-2 focus:ring-indigo-500"
                      />
                      <span className="ml-2 text-gray-700">{metric}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
        
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold text-gray-800">Network architecture</h2>
            <button
              type="button"
              onClick={addLayer}
              className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors flex items-center shadow-sm"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add a layer
            </button>
          </div>
          
          {networkConfig.layers.length === 0 && (
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-md mb-4">
              <div className="flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-500 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <p className="text-yellow-700 font-medium">No layers. Add at least one layer to configure the network.</p>
              </div>
            </div>
          )}
          
          <div className="overflow-auto max-h-96 border border-gray-200 rounded-lg shadow-sm bg-gray-50">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-100 sticky top-0 z-10">
                <tr>
                  <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-12">No.</th>
                  {showAdvancedOptions && (
                    <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                  )}
                  <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Neurons</th>
                  <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Activation</th>
                  <th className="px-2 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Dropout</th>
                  <th className="px-2 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider w-32">Shares</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {networkConfig.layers.map((layer, index) => (
                  <tr key={index} className="hover:bg-gray-50 transition-colors">
                    {/* Numer warstwy */}
                    <td className="px-2 py-3 whitespace-nowrap">
                      <span className="inline-flex items-center justify-center h-6 w-6 rounded-full bg-indigo-100 text-indigo-800 text-xs font-medium">
                        {index + 1}
                      </span>
                    </td>
                    
                    {/* Typ warstwy (jeśli zaawansowane) */}
                    {showAdvancedOptions && (
                      <td className="px-2 py-3 whitespace-nowrap">
                        <select
                          value={layer.type}
                          onChange={(e) => updateLayer(index, 'type', e.target.value)}
                          className="w-full p-1.5 border border-gray-300 rounded text-sm text-gray-700 bg-white focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 appearance-none"
                        >
                          {layerTypeOptions.map(option => (
                            <option key={option} value={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      </td>
                    )}
                    
                    {/* Liczba neuronów */}
                    <td className="px-2 py-3 whitespace-nowrap">
                      <div className="flex items-center">
                        <button
                          type="button"
                          onClick={() => updateLayer(index, 'units', Math.max(1, layer.units - (layer.units >= 100 ? 16 : layer.units >= 32 ? 8 : 4)))}
                          className="h-8 w-8 flex items-center justify-center rounded-l border border-gray-300 bg-gray-50 text-gray-700 hover:bg-gray-100"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                          </svg>
                        </button>
                        <input
                          type="number"
                          value={layer.units}
                          onChange={(e) => updateLayer(index, 'units', e.target.value)}
                          min="1"
                          className="h-8 w-16 text-center border-y border-gray-300 focus:ring-0 focus:outline-none text-sm"
                        />
                        <button
                          type="button"
                          onClick={() => updateLayer(index, 'units', layer.units + (layer.units >= 100 ? 16 : layer.units >= 32 ? 8 : 4))}
                          className="h-8 w-8 flex items-center justify-center rounded-r border border-gray-300 bg-gray-50 text-gray-700 hover:bg-gray-100"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                        </button>
                      </div>
                    </td>
                    
                    {/* Aktywacja */}
                    <td className="px-2 py-3 whitespace-nowrap">
                      <select
                        value={layer.activation}
                        onChange={(e) => updateLayer(index, 'activation', e.target.value)}
                        className="w-full p-1.5 border border-gray-300 rounded text-sm text-gray-700 bg-white focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 appearance-none"
                      >
                        {activationOptions.map(option => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </td>
                    
                    {/* Dropout */}
                    <td className="px-2 py-3 whitespace-nowrap">
                      <div className="flex items-center">
                        <input
                          type="range"
                          value={layer.dropout}
                          onChange={(e) => updateLayer(index, 'dropout', e.target.value)}
                          step="0.1"
                          min="0"
                          max="0.9"
                          className="w-24 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                        />
                        <span className="ml-2 bg-indigo-100 text-indigo-800 px-1.5 py-0.5 rounded text-xs font-medium min-w-8 text-center">
                          {layer.dropout}
                        </span>
                      </div>
                    </td>
                    
                    {/* Akcje */}
                    <td className="px-2 py-3 whitespace-nowrap text-right">
                      <div className="flex justify-end space-x-1">
                        <button
                          type="button"
                          onClick={() => moveLayer(index, 'up')}
                          disabled={index === 0}
                          className={`p-1 rounded ${
                            index === 0 
                              ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                          }`}
                          title="Przesuń w górę"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                          </svg>
                        </button>
                        <button
                          type="button"
                          onClick={() => moveLayer(index, 'down')}
                          disabled={index === networkConfig.layers.length - 1}
                          className={`p-1 rounded ${
                            index === networkConfig.layers.length - 1 
                              ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                          }`}
                          title="Przesuń w dół"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </button>
                        <button
                          type="button"
                          onClick={() => duplicateLayer(index)}
                          className="p-1 rounded bg-green-100 text-green-700 hover:bg-green-200"
                          title="Duplikuj warstwę"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                          </svg>
                        </button>
                        <button
                          type="button"
                          onClick={() => removeLayer(index)}
                          disabled={networkConfig.layers.length <= 1}
                          className={`p-1 rounded ${
                            networkConfig.layers.length <= 1 
                              ? 'bg-red-100 text-red-300 cursor-not-allowed' 
                              : 'bg-red-100 text-red-600 hover:bg-red-200'
                          }`}
                          title="Usuń warstwę"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            
            {networkConfig.layers.length === 0 && (
              <div className="py-6 px-4 text-center">
                <p className="text-gray-500 font-medium">No layers. Add a layer to configure the network.</p>
              </div>
            )}
          </div>
          
          <div className="mt-4 flex justify-center">
            <button
              type="button"
              onClick={addLayer}
              className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors flex items-center shadow-sm"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add a layer
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default NeuralNetworkForm;