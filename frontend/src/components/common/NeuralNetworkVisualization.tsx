import React from 'react';

interface Layer {
  type: string;
  units: number;
  activation: string;
  dropout: number;
}

interface NeuralNetworkConfig {
  dropout_rate: number;
  activation: string;
  output_activation: string;
  optimizer: string;
  loss: string;
  metrics: string[];
  layers: Layer[];
}

interface NeuralNetworkVisualizationProps {
  config: NeuralNetworkConfig | null;
}

const NeuralNetworkVisualization: React.FC<NeuralNetworkVisualizationProps> = ({ config }) => {
  if (!config) {
    return (
      <div className="flex items-center justify-center h-72 bg-gray-100 rounded-lg border border-gray-200 shadow-inner">
        <p className="text-gray-500 font-medium">Configure the network to see the visualization</p>
      </div>
    );
  }

  // Obliczenie maksymalnej liczby neuronów dla skalowania
  const maxNeurons = Math.max(...config.layers.map(layer => layer.units), 10);
  
  // Dodajemy warstwę wejściową i wyjściową do wizualizacji
  const allLayers = [
    { type: 'input', units: 10, activation: 'none', dropout: 0 },
    ...config.layers,
    { type: 'output', units: 1, activation: config.output_activation, dropout: 0 }
  ];

  // Kolor dla każdej funkcji aktywacji z gradientem
  const activationColors: Record<string, { bg: string, border: string, hover: string }> = {
    'relu': { bg: '#FF6B6B', border: '#FF4F4F', hover: '#FF8585' },
    'sigmoid': { bg: '#4ECDC4', border: '#36B5AC', hover: '#66D5CE' },
    'tanh': { bg: '#F9DB6D', border: '#F7CF45', hover: '#FAE38D' },
    'linear': { bg: '#9D84B7', border: '#8C6FA8', hover: '#AE99C6' },
    'softmax': { bg: '#45B7D1', border: '#2FA8C5', hover: '#5DC3D9' },
    'none': { bg: '#A0A0A0', border: '#8A8A8A', hover: '#B6B6B6' }
  };

  // Legenda funkcji aktywacji
  const renderLegend = () => (
    <div className="flex flex-wrap gap-3 mt-3 mb-6 justify-center">
      {Object.keys(activationColors).filter(key => key !== 'none').map(activation => (
        <div key={activation} className="flex items-center">
          <div 
            className="w-4 h-4 rounded-full mr-1"
            style={{ backgroundColor: activationColors[activation].bg }}
          />
          <span className="text-xs">{activation}</span>
        </div>
      ))}
    </div>
  );

  return (
    <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
      <h3 className="text-lg font-semibold mb-2 text-gray-800">Visualization of the network architecture</h3>
      
      {renderLegend()}
      
      <div className="relative min-h-72 flex items-center justify-between p-4 bg-gray-50 rounded-lg overflow-x-auto">
        {/* Linie połączeń między warstwami */}
        <svg className="absolute top-0 left-0 w-full h-full -z-10">
          {allLayers.map((layer, layerIndex) => {
            if (layerIndex === allLayers.length - 1) return null;
            
            const nextLayer = allLayers[layerIndex + 1];
            const maxDisplayNeurons = 10;
            const displayNeurons = Math.min(layer.units, maxDisplayNeurons);
            const nextDisplayNeurons = Math.min(nextLayer.units, maxDisplayNeurons);
            
            // Przybliżona pozycja warstwy w procentach
            const layerPos = (layerIndex / (allLayers.length - 1)) * 100;
            const nextLayerPos = ((layerIndex + 1) / (allLayers.length - 1)) * 100;
            
            const connections = [];
            
            // Rysuj połączenia między warstwami (z gradientem)
            for (let i = 0; i < displayNeurons; i += Math.ceil(displayNeurons / 5)) {
              for (let j = 0; j < nextDisplayNeurons; j += Math.ceil(nextDisplayNeurons / 5)) {
                connections.push(
                  <line
                    key={`${layerIndex}-${i}-${j}`}
                    x1={`${layerPos}%`}
                    y1={`${50 + (i - displayNeurons / 2) * 14}px`}
                    x2={`${nextLayerPos}%`}
                    y2={`${50 + (j - nextDisplayNeurons / 2) * 14}px`}
                    stroke="url(#connection-gradient)"
                    strokeWidth="1.5"
                    opacity="0.4"
                  />
                );
              }
            }
            
            return connections;
          })}
          
          {/* Gradient dla połączeń */}
          <defs>
            <linearGradient id="connection-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#6366F1" stopOpacity="0.7" />
              <stop offset="100%" stopColor="#8B5CF6" stopOpacity="0.7" />
            </linearGradient>
          </defs>
        </svg>
        
        {/* Warstwy i neurony */}
        {allLayers.map((layer, index) => {
          const layerColor = activationColors[layer.activation] || activationColors.none;
          
          // Oblicz maksymalną liczbę neuronów do wyświetlenia (dla czytelności)anomalies_detection/frontend/src/components/tabs
          const maxDisplayNeurons = 10;
          const displayNeurons = Math.min(layer.units, maxDisplayNeurons);
          
          // Określ tytuł warstwy
          let layerTitle = `Layer ${index}`;
          if (index === 0) layerTitle = 'Input';
          else if (index === allLayers.length - 1) layerTitle = 'Output';
          
          return (
            <div key={index} className="flex flex-col items-center relative min-w-24">
              {/* Nagłówek warstwy */}
              <div className="mb-3 text-sm font-medium bg-white px-3 py-1 rounded-full shadow-sm border border-gray-200 text-gray-700">
                {layerTitle}
              </div>
              
              {/* Container dla neuronów z efektem głębi */}
              <div className="flex flex-col items-center relative">
                <div className="absolute left-0 right-0 top-0 bottom-0 rounded-lg bg-gray-200 opacity-20 transform -translate-x-1 translate-y-1"></div>
                <div className="relative bg-white rounded-lg p-3 border border-gray-200 shadow-sm">
                  {/* Neurony */}
                  {Array.from({ length: displayNeurons }).map((_, i) => {
                    // Oblicz wielkość neuronu na podstawie jego pozycji (środkowe większe)
                    const distanceFromCenter = Math.abs(i - (displayNeurons - 1) / 2) / ((displayNeurons - 1) / 2);
                    const sizeMultiplier = 1 - (distanceFromCenter * 0.3);
                    const baseSize = 10; // Bazowy rozmiar neuronu
                    const neuronSize = baseSize * sizeMultiplier;
                    
                    return (
                      <div 
                        key={i}
                        className="rounded-full my-1 transition-all duration-200 hover:scale-110 shadow-sm border"
                        style={{
                          width: `${neuronSize}px`,
                          height: `${neuronSize}px`,
                          backgroundColor: layerColor.bg,
                          borderColor: layerColor.border,
                          opacity: layer.dropout > 0 ? `${0.9 - layer.dropout * 0.7}` : 1
                        }}
                        title={`Neuron ${i+1}\nActivation: ${layer.activation}\nDropout: ${layer.dropout}`}
                      />
                    );
                  })}
                  
                  {/* Wskaźnik dodatkowych neuronów */}
                  {layer.units > maxDisplayNeurons && (
                    <div className="text-xs text-gray-500 mt-2 bg-gray-100 px-2 py-1 rounded-full text-center">
                      +{layer.units - maxDisplayNeurons}
                    </div>
                  )}
                </div>
              </div>
              
              {/* Informacje o warstwie */}
              <div className="mt-3 text-xs text-center bg-white p-2 rounded-md border border-gray-200 shadow-sm">
                <div className="font-medium text-gray-700">{layer.units} {layer.units === 1 ? 'neuron' : 'neurons'}</div>
                {layer.activation !== 'none' && (
                  <div className="text-gray-600 flex items-center justify-center mt-1">
                    <span className="inline-block w-2 h-2 rounded-full mr-1" style={{ backgroundColor: layerColor.bg }}></span>
                    {layer.activation}
                    {layer.dropout > 0 && (
                      <span className="ml-1 bg-gray-200 px-1 rounded text-gray-600 text-xs">
                        drop: {layer.dropout}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Informacje o konfiguracji */}
      <div className="mt-6 pt-4 border-t border-gray-200 text-sm">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 p-3 rounded-lg">
            <span className="font-medium text-gray-700 block mb-1">Optimizer:</span>
            <span className="inline-block bg-indigo-100 text-indigo-800 px-2 py-1 rounded">{config.optimizer}</span>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg">
            <span className="font-medium text-gray-700 block mb-1">Loss function:</span>
            <span className="inline-block bg-red-100 text-red-800 px-2 py-1 rounded">{config.loss}</span>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg">
            <span className="font-medium text-gray-700 block mb-1">Metrics:</span>
            <div>
              {config.metrics.map((metric, idx) => (
                <span key={idx} className="inline-block bg-green-100 text-green-800 px-2 py-1 rounded mr-1 mb-1">{metric}</span>
              ))}
            </div>
          </div>
          <div className="bg-gray-50 p-3 rounded-lg">
            <span className="font-medium text-gray-700 block mb-1">Dropout (global):</span>
            <span className="inline-block bg-purple-100 text-purple-800 px-2 py-1 rounded">{config.dropout_rate}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NeuralNetworkVisualization;