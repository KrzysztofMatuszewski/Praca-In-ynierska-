// model_panel/components/overview.tsx
import React, { useState } from 'react';
import NeuralNetworkForm from '@/components/common/NeuralNetworkForm';
import NeuralNetworkVisualization from '@/components/common/NeuralNetworkVisualization';

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

const Overview: React.FC = () => {
  const [networkConfig, setNetworkConfig] = useState<NeuralNetworkConfig | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submitResult, setSubmitResult] = useState<{ success: boolean; message: string } | null>(null);
  const [activeTab, setActiveTab] = useState<'visualization' | 'json'>('visualization');
  
  const handleConfigChange = (config: NeuralNetworkConfig) => {
    setNetworkConfig(config);
    // Reset submit result when config changes
    setSubmitResult(null);
  };
  
  const handleSubmitToBackend = async () => {
    if (!networkConfig) {
      setSubmitResult({
        success: false,
        message: "No configuration to send. Configure the network first."
      });
      return;
    }
    
    setIsSubmitting(true);
    setSubmitResult(null);
    
    try {
      const response = await fetch('http://localhost:8012/api/neural-network/config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(networkConfig),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      
      const data = await response.json();
      
      setSubmitResult({
        success: true,
        message: "The configuration was successfully sent to the server."
      });
    } catch (error) {
      console.error('Error while sending configuration:', error);
      setSubmitResult({
        success: false,
        message: `The configuration could not be sent: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Neural Network Configurator</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <NeuralNetworkForm onConfigChange={handleConfigChange} />
        </div>
        
        <div>
          {/* Tabs */}
          <div className="flex mb-4 border-b">
            <button
              className={`py-2 px-4 font-medium ${
                activeTab === 'visualization' 
                  ? 'text-blue-600 border-b-2 border-blue-600' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('visualization')}
            >
              Visualization
            </button>
            
            <button
              className={`py-2 px-4 font-medium ${
                activeTab === 'json' 
                  ? 'text-blue-600 border-b-2 border-blue-600' 
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              onClick={() => setActiveTab('json')}
            >
              JSON
            </button>
          </div>
          
          {/* Tab Content */}
          <div className="mb-4">
            {activeTab === 'visualization' ? (
              <NeuralNetworkVisualization config={networkConfig} />
            ) : (
              <div>
                <h2 className="text-xl font-semibold mb-4">JSON configuration</h2>
                <div className="bg-gray-800 text-green-400 p-4 rounded overflow-auto h-72">
                  <pre>{networkConfig ? JSON.stringify(networkConfig, null, 2) : 'Configure the network to see JSON'}</pre>
                </div>
              </div>
            )}
          </div>
          
          <div className="mt-4">
            <button
              onClick={handleSubmitToBackend}
              disabled={isSubmitting || !networkConfig}
              className={`w-full py-3 rounded font-semibold ${
                isSubmitting || !networkConfig 
                  ? 'bg-gray-400 cursor-not-allowed' 
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              {isSubmitting ? 'Sending...' : 'Send to backend'}
            </button>
            
            {submitResult && (
              <div className={`mt-3 p-3 rounded ${
                submitResult.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {submitResult.message}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Overview;