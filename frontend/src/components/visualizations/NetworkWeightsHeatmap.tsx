import React, { useState, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Switch } from '@radix-ui/react-switch';
import { AlertCircle, ZoomIn, ZoomOut } from 'lucide-react';
import { NetworkWeightsHeatmapProps } from '@/types/nids';

const NetworkWeightsHeatmap: React.FC<NetworkWeightsHeatmapProps> = ({ 
  weightsData,
  selectedFolder 
}) => {
  const [selectedLayer, setSelectedLayer] = useState('');
  const [heatmapZoom, setHeatmapZoom] = useState(100);
  const [showValues, setShowValues] = useState(true);
  
  // Check if we have valid data
  if (!weightsData || !selectedFolder || !weightsData[selectedFolder]) {
    return <p className="text-center py-4">No weights data available for this folder</p>;
  }
  
  // Get available layers for this folder (excluding 'features' field)
  const availableLayers = Object.keys(weightsData[selectedFolder])
    .filter(key => key !== 'features');
  
  if (availableLayers.length === 0) {
    return <p className="text-center py-4">No layer weights available for this folder</p>;
  }
  
  // Reset selectedLayer when selectedFolder changes
  useEffect(() => {
    // When folder changes, reset the selected layer to the first available in the new folder
    if (availableLayers.length > 0) {
      setSelectedLayer(availableLayers[0]);
    }
  }, [selectedFolder]); // Only depend on selectedFolder
  
  // Get weights data for the selected layer
  const layerData = selectedLayer ? weightsData[selectedFolder][selectedLayer] : null;
  
  if (!layerData) {
    return <p className="text-center py-4">Please select a layer to visualize</p>;
  }
  
  // Prepare data for the heatmap
  const prepareHeatmapData = () => {
    const weights = layerData.weights;
    
    // If we don't have weights, return empty dataset
    if (!weights || !Array.isArray(weights) || weights.length === 0) {
      return { data: [], xLabels: [], yLabels: [] };
    }
    
    // Get axis labels
    const featuresNames = weightsData[selectedFolder].features || [];
    const xLabels = featuresNames.length >= weights[0].length 
      ? featuresNames.slice(0, weights[0].length) 
      : Array.from({ length: weights[0].length }, (_, i) => `${i+1}`);
    
    const yLabels = featuresNames.length >= weights.length 
      ? featuresNames.slice(0, weights.length) 
      : Array.from({ length: weights.length }, (_, i) => `Feature ${i+1}`);
    
    // Prepare data in format for heatmap
    const data = weights.map((row, rowIndex) => {
      return row.map((value, colIndex) => ({
        x: colIndex,
        y: rowIndex,
        value: value
      }));
    }).flat();
    
    return { data, xLabels, yLabels };
  };
  
  const { data, xLabels, yLabels } = prepareHeatmapData();
  
  // Find min and max weight values for color scale
  const minWeight = Math.min(...data.map(d => d.value));
  const maxWeight = Math.max(...data.map(d => d.value));
  const absMax = Math.max(Math.abs(minWeight), Math.abs(maxWeight));
  
  // Function to determine color based on weight value
  const getColor = (value) => {
    // Normalize value to range [-1, 1]
    const normalizedValue = value / absMax;
    
    // Change colors for better readability and aesthetics
    if (value < 0) {
      // From blue (dark) to light blue
      const blueIntensity = Math.floor((1 - Math.abs(normalizedValue)) * 255);
      return `rgb(${blueIntensity}, ${blueIntensity + Math.floor(Math.abs(normalizedValue) * 100)}, 255)`;
    } else if (value > 0) {
      // From red (dark) to light red
      const redIntensity = Math.floor((1 - normalizedValue) * 255);
      return `rgb(255, ${redIntensity}, ${redIntensity})`;
    } else {
      // Value 0 - grey shade
      return 'rgb(240, 240, 240)';
    }
  };
  
  // Determine text color intensity based on background
  const getTextColor = (value) => {
    const normalizedValue = Math.abs(value) / absMax;
    // For intense background colors use white text
    return normalizedValue > 0.5 ? 'white' : 'black';
  };
  
  // Base cell size, adjusted by zoom
  const baseCellSize = 24;
  const cellSize = Math.floor(baseCellSize * (heatmapZoom / 50));
  
  return (
    <div>
      {/* Control panel */}
      <div className="flex flex-col lg:flex-row gap-4 mb-4">
        {/* Layer selection */}
        <div className="flex-1">
          <label className="text-sm font-medium mb-2 block">Select Layer:</label>
          <Select 
            value={selectedLayer} 
            onValueChange={setSelectedLayer}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a network layer" />
            </SelectTrigger>
            <SelectContent>
              {availableLayers.map(layer => (
                <SelectItem key={layer} value={layer}>
                  {layer} ({weightsData[selectedFolder][layer].shape?.join(' × ') || 'unknown'})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        {/* Zoom control */}
        <div className="flex-1">
          <label className="text-sm font-medium mb-2 block">Zoom: {heatmapZoom}%</label>
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setHeatmapZoom(Math.max(50, heatmapZoom - 10))}
              disabled={heatmapZoom <= 50}
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            <input
              type="range"
              min="50"
              max="200"
              step="10"
              value={heatmapZoom}
              onChange={(e) => setHeatmapZoom(parseInt(e.target.value))}
              className="flex-1"
            />
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setHeatmapZoom(Math.min(200, heatmapZoom + 10))}
              disabled={heatmapZoom >= 200}
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Toggle numeric values */}
        <div className="flex items-end">
          <div className="flex items-center space-x-2">
            <Switch 
              id="show-values"
              checked={showValues}
              onCheckedChange={setShowValues}
            />
            <label htmlFor="show-values" className="text-sm font-medium">Show values</label>
          </div>
        </div>
      </div>
      
      {/* Selected layer information */}
      {layerData && (
        <div className="mb-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-md border">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <h4 className="text-sm font-semibold text-gray-500 dark:text-gray-400">Layer Type</h4>
              <p className="font-medium">{layerData.layer_type || 'Unknown'}</p>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-500 dark:text-gray-400">Shape</h4>
              <p className="font-medium">{layerData.shape?.join(' × ') || 'Unknown'}</p>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-500 dark:text-gray-400">Input Size</h4>
              <p className="font-medium">{layerData.input_size || 'Unknown'}</p>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-gray-500 dark:text-gray-400">Output Size</h4>
              <p className="font-medium">{layerData.output_size || 'Unknown'}</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Heatmap */}
      {data.length > 0 ? (
        <div className="border rounded-md" style={{
          overflow: 'auto',
          maxHeight: '70vh',
          maxWidth: '100%',
          boxShadow: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)'
        }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: `auto ${xLabels.map(() => `${cellSize}px`).join(' ')}`,
            gridTemplateRows: `auto ${yLabels.map(() => `${cellSize}px`).join(' ')}`,
            gap: '1px',
            backgroundColor: '#ddd'
          }}>
            {/* Empty top left corner */}
            <div className="bg-white dark:bg-gray-800" style={{ minWidth: '150px' }}></div>
            
            {/* Column labels (input feature names) */}
            {xLabels.map((label, i) => (
              <div 
                key={`col-${i}`} 
                className="bg-white dark:bg-gray-800 text-xs" 
                style={{
                  padding: '5px 10px',
                  fontSize: '12px',
                  minWidth: '150px',
                  display: 'flex',
                  alignItems: 'center',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                  borderRight: '1px solid #eee'
                }}
                title={label ? String(label) : undefined}
              >
                {label}
              </div>
            ))}
            
            {/* Generate heatmap rows */}
            {yLabels.map((label, rowIndex) => (
              <React.Fragment key={`row-${rowIndex}`}>
                {/* Row label */}
                <div 
                  className="bg-white dark:bg-gray-800 font-medium"
                  style={{
                    padding: '5px 10px',
                    fontSize: '12px',
                    minWidth: '150px',
                    display: 'flex',
                    alignItems: 'center',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    borderRight: '1px solid #eee'
                  }} 
                  title={label ? String(label) : undefined}
                >
                  {label}
                </div>
                
                {/* Heatmap cells */}
                {xLabels.map((_, colIndex) => {
                  const cell = data.find(d => d.x === colIndex && d.y === rowIndex);
                  const value = cell?.value || 0;
                  const cellColor = getColor(value);
                  
                  return (
                    <div 
                      key={`cell-${rowIndex}-${colIndex}`}
                      style={{
                        backgroundColor: cellColor,
                        width: `${cellSize}px`,
                        height: `${cellSize}px`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: cellSize < 20 ? '8px' : '10px',
                        fontWeight: 'medium',
                        color: getTextColor(value),
                        transition: 'background-color 0.2s',
                        cursor: 'pointer'
                      }}
                      title={`Value: ${value.toFixed(6)}`}
                    >
                      {showValues && cellSize >= 18 && (
                        Math.abs(value) < 0.01 
                          ? value.toExponential(1) 
                          : value.toFixed(2)
                      )}
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center p-10 border rounded-md bg-gray-50">
          <div className="text-center text-gray-500">
            <AlertCircle className="h-10 w-10 mx-auto mb-2 text-gray-400" />
            <p>No weights data available for visualization</p>
          </div>
        </div>
      )}
      
      {/* Legend - improved version */}
      <div className="mt-6 p-4 border rounded-md bg-white">
        <h4 className="text-sm font-semibold mb-2">Color Legend</h4>
        <div className="flex flex-col sm:flex-row items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-32 h-4 rounded-md bg-gradient-to-r from-blue-600 via-gray-200 to-red-600"></div>
            <span className="text-xs">Weight values</span>
          </div>
          
          <div className="flex items-center gap-3 ml-4">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-blue-600 rounded-sm"></div>
              <span className="text-xs ml-1">Strong negative</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-gray-200 rounded-sm"></div>
              <span className="text-xs ml-1">Neutral</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-red-600 rounded-sm"></div>
              <span className="text-xs ml-1">Strong positive</span>
            </div>
          </div>
          
          <div className="ml-auto text-xs text-gray-600">
            Range: [{minWeight.toExponential(2)}, {maxWeight.toExponential(2)}]
          </div>
        </div>
        
        <div className="mt-3 text-xs text-gray-500">
          <p>Weights represent the strength of connections between neurons. Positive weights amplify signals, negative weights inhibit them.</p>
        </div>
      </div>
    </div>
  );
};

export default NetworkWeightsHeatmap;