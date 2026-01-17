import React, { useState, useRef, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info, ZoomIn, ZoomOut } from 'lucide-react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis, Cell, Brush } from 'recharts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';

interface TSNEPoint {
  coordinates: number[];
  mse: number;
  original_features: {
    [key: string]: string;
  };
}

interface TSNEData {
  points: TSNEPoint[];
  metadata: {
    n_components: number;
    perplexity: number;
    data_points: number;
    latent_dimensions: number;
    features: string[];
  };
}

interface TSNETabProps {
  tsneData: { [folder: string]: TSNEData };
  selectedFolder: string;
  setSelectedFolder: (folder: string) => void;
  loading: boolean;
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const point = payload[0].payload;
    return (
      <div className="bg-white p-3 border rounded shadow-lg">
        <p className="font-semibold">MSE: {point.mse.toFixed(6)}</p>
        <p className="text-xs text-gray-600">Coordinates: ({point.x.toFixed(4)}, {point.y.toFixed(4)})</p>
        {point.original_features && (
          <div className="mt-2">
            <p className="text-xs font-semibold">Original Features:</p>
            <div className="max-h-40 overflow-y-auto">
              {Object.entries(point.original_features).map(([key, value]) => (
                <p key={key} className="text-xs">
                  <span className="font-medium">{key}:</span> {String(value)}
                </p>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }
  return null;
};

// Custom selection overlay component
const SelectionOverlay = ({ 
  isSelecting, 
  onSelectionComplete, 
  containerRef 
}: { 
  isSelecting: boolean; 
  onSelectionComplete: (selection: { x1: number, y1: number, x2: number, y2: number }) => void; 
  containerRef: React.RefObject<HTMLDivElement | null>;
}) => {
  const [selectionStart, setSelectionStart] = useState<{ x: number, y: number } | null>(null);
  const [selectionCurrent, setSelectionCurrent] = useState<{ x: number, y: number } | null>(null);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isSelecting) {
      setSelectionStart(null);
      setSelectionCurrent(null);
    }
  }, [isSelecting]);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (!isSelecting || !overlayRef.current) return;
    
    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setSelectionStart({ x, y });
    setSelectionCurrent({ x, y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isSelecting || !selectionStart || !overlayRef.current) return;
    
    const rect = overlayRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setSelectionCurrent({ x, y });
  };

  const handleMouseUp = () => {
    if (!isSelecting || !selectionStart || !selectionCurrent || !overlayRef.current || !containerRef.current) return;
    
    // Convert pixel positions to relative positions (0-1)
    const width = overlayRef.current.offsetWidth;
    const height = overlayRef.current.offsetHeight;
    
    // Ensure values are between 0-1
    const x1 = Math.max(0, Math.min(1, selectionStart.x / width));
    const y1 = Math.max(0, Math.min(1, selectionStart.y / height));
    const x2 = Math.max(0, Math.min(1, selectionCurrent.x / width));
    const y2 = Math.max(0, Math.min(1, selectionCurrent.y / height));
    
    // Only trigger if selection area is large enough
    const minSelectionSize = 0.01; // At least 1% of chart area
    if (Math.abs(x2 - x1) > minSelectionSize && Math.abs(y2 - y1) > minSelectionSize) {
      onSelectionComplete({ x1, y1, x2, y2 });
    }
    
    setSelectionStart(null);
    setSelectionCurrent(null);
  };

  // No overlay if not in selection mode
  if (!isSelecting) return null;

  // Calculate selection rectangle dimensions
  const selectionStyle = selectionStart && selectionCurrent ? {
    left: Math.min(selectionStart.x, selectionCurrent.x),
    top: Math.min(selectionStart.y, selectionCurrent.y),
    width: Math.abs(selectionCurrent.x - selectionStart.x),
    height: Math.abs(selectionCurrent.y - selectionStart.y),
  } : undefined;

  return (
    <div 
      ref={overlayRef}
      className="absolute inset-0 cursor-crosshair bg-transparent z-10"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="bg-black bg-opacity-20 text-white px-4 py-2 rounded">
          Click and drag to select an area
        </div>
      </div>
      
      {selectionStyle && (
        <div 
          className="absolute border-2 border-blue-500 bg-blue-500 bg-opacity-20 pointer-events-none"
          style={selectionStyle}
        />
      )}
    </div>
  );
};

const TSNETab: React.FC<TSNETabProps> = ({ 
  tsneData,
  selectedFolder,
  setSelectedFolder,
  loading
}) => {
  const [colorFeature, setColorFeature] = useState<string>('mse');
  const [isSelecting, setIsSelecting] = useState(false);
  const [zoomDomain, setZoomDomain] = useState<{ 
    x: [number, number] | null, 
    y: [number, number] | null 
  }>({ x: null, y: null });
  
  // Reference to the chart container
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // Prepare data for scatter plot
  const prepareScatterData = (data: TSNEData) => {
    if (!data || !data.points) return [];
    
    return data.points.map(point => ({
      x: point.coordinates[0],
      y: point.coordinates[1],
      z: point.coordinates.length > 2 ? point.coordinates[2] : 0,
      mse: point.mse,
      original_features: point.original_features
    }));
  };

  // Get available features for coloring
  const getAvailableFeatures = () => {
    if (!tsneData || !selectedFolder || !tsneData[selectedFolder]) return ['mse'];
    
    const samplePoint = tsneData[selectedFolder].points[0];
    if (!samplePoint || !samplePoint.original_features) return ['mse'];
    
    return ['mse', ...Object.keys(samplePoint.original_features)];
  };

  // Get color value based on selected feature
  const getPointColor = (point: any) => {
    if (colorFeature === 'mse') {
      // Color based on MSE - red for high values, blue for low
      const mseFactor = Math.min(point.mse * 100, 1); // Normalize between 0-1
      return mseFactor > 0.5 ? `rgba(255, ${Math.floor(255 * (1 - mseFactor) * 2)}, 0, 0.7)` : `rgba(0, ${Math.floor(255 * mseFactor * 2)}, 255, 0.7)`;
    } else if (point.original_features && point.original_features[colorFeature]) {
      // Hash string feature value to a color
      const str = point.original_features[colorFeature];
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
      }
      let color = '#';
      for (let i = 0; i < 3; i++) {
        const value = (hash >> (i * 8)) & 0xFF;
        color += ('00' + value.toString(16)).substr(-2);
      }
      return color;
    }
    return '#8884d8'; // Default color
  };

  // Handler for selection completion
  const handleSelectionComplete = (selection: { x1: number, y1: number, x2: number, y2: number }) => {
    const { x1, y1, x2, y2 } = selection;
    
    // Get data domain
    const originalData = scatterData;
    const xValues = originalData.map(d => d.x);
    const yValues = originalData.map(d => d.y);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    // Calculate the actual domain values based on selection percentages
    const xMinZoom = xMin + (xMax - xMin) * Math.min(x1, x2);
    const xMaxZoom = xMin + (xMax - xMin) * Math.max(x1, x2);
    
    // Fix for inverted Y coordinates - use direct mapping instead of inverting
    const yMinZoom = yMin + (yMax - yMin) * Math.min(y1, y2);
    const yMaxZoom = yMin + (yMax - yMin) * Math.max(y1, y2);
    
    // Set zoom domain
    setZoomDomain({
      x: [xMinZoom, xMaxZoom],
      y: [yMinZoom, yMaxZoom]
    });
    
    // Exit selection mode
    setIsSelecting(false);
  };

  // Reset zoom to show all data
  const resetZoom = () => {
    setZoomDomain({ x: null, y: null });
  };

  // Toggle selection mode
  const toggleSelectionMode = () => {
    setIsSelecting(!isSelecting);
  };

  // Reset zoom when changing folder
  useEffect(() => {
    resetZoom();
    setIsSelecting(false);
  }, [selectedFolder]);

  const scatterData = selectedFolder && tsneData[selectedFolder] 
    ? prepareScatterData(tsneData[selectedFolder]) 
    : [];
    
  // Filter data to include only points within current zoom domain
  const getFilteredData = () => {
    if (!zoomDomain.x || !zoomDomain.y || scatterData.length === 0) {
      return scatterData;
    }
    
    const [xMin, xMax] = zoomDomain.x;
    const [yMin, yMax] = zoomDomain.y;
    
    // Add a small buffer around the visible area (5% of domain size)
    const xBuffer = (xMax - xMin) * 0.05;
    const yBuffer = (yMax - yMin) * 0.05;
    
    return scatterData.filter(point => 
      point.x >= xMin - xBuffer && 
      point.x <= xMax + xBuffer && 
      point.y >= yMin - yBuffer && 
      point.y <= yMax + yBuffer
    );
  };
  
  const filteredData = getFilteredData();
  
  // Log stats about filtering for performance monitoring
  useEffect(() => {
    if (zoomDomain.x && zoomDomain.y) {
      console.log(`Rendering ${filteredData.length} points out of ${scatterData.length} total (${Math.round(filteredData.length/scatterData.length*100)}%)`);
    }
  }, [filteredData.length, scatterData.length, zoomDomain]);

  // Get data domain for axes
  const getXDomain = () => {
    if (zoomDomain.x) return zoomDomain.x;
    if (scatterData.length === 0) return [0, 1];
    
    const xValues = scatterData.map(d => d.x);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const xPadding = (xMax - xMin) * 0.05;  // 5% padding
    
    return [xMin - xPadding, xMax + xPadding];
  };

  const getYDomain = () => {
    if (zoomDomain.y) return zoomDomain.y;
    if (scatterData.length === 0) return [0, 1];
    
    const yValues = scatterData.map(d => d.y);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const yPadding = (yMax - yMin) * 0.05;  // 5% padding
    
    return [yMin - yPadding, yMax + yPadding];
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col md:flex-row gap-4 mb-4">
        <div className="md:w-1/3">
          <Label htmlFor="folder-select">Select Model Folder</Label>
          <Select 
            value={selectedFolder} 
            onValueChange={setSelectedFolder}
            disabled={loading || Object.keys(tsneData).length === 0}
          >
            <SelectTrigger id="folder-select" className="w-full">
              <SelectValue placeholder="Select a folder" />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(tsneData).map(folder => (
                <SelectItem key={folder} value={folder}>
                  {folder}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="md:w-1/3">
          <Label htmlFor="color-feature">Color Points By</Label>
          <Select 
            value={colorFeature} 
            onValueChange={setColorFeature}
            disabled={loading || !selectedFolder || !tsneData[selectedFolder]}
          >
            <SelectTrigger id="color-feature" className="w-full">
              <SelectValue placeholder="Select feature for coloring" />
            </SelectTrigger>
            <SelectContent>
              {getAvailableFeatures().map(feature => (
                <SelectItem key={feature} value={feature}>
                  {feature}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="md:w-1/3 flex flex-col">
          <Label className="mb-2">Selection Controls</Label>
          <div className="flex gap-2">
            <Button
              onClick={toggleSelectionMode}
              variant={isSelecting ? "default" : "outline"}
              size="sm"
              className="flex-1"
              disabled={loading || !selectedFolder || !tsneData[selectedFolder]}
            >
              {isSelecting ? (
                <>Cancel Selection</>
              ) : (
                <><ZoomIn className="mr-1 h-4 w-4" /> Select Area</>
              )}
            </Button>
            <Button
              onClick={resetZoom}
              variant="outline"
              size="sm"
              disabled={!zoomDomain.x && !zoomDomain.y}
              className="flex-1"
            >
              <ZoomOut className="mr-1 h-4 w-4" /> Reset Zoom
            </Button>
          </div>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>
            t-SNE Visualization {selectedFolder ? `(${selectedFolder})` : ''}
            {zoomDomain.x && <span className="text-sm font-normal ml-2">(Zoomed)</span>}
          </CardTitle>
        </CardHeader>
        <CardContent className="h-96 relative" ref={chartContainerRef}>
          {scatterData.length > 0 ? (
            <>
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                >
                  <CartesianGrid />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name="t-SNE Dimension 1" 
                    label={{ value: 't-SNE Dimension 1', position: 'insideBottom', offset: -5 }} 
                    domain={getXDomain()}
                    allowDataOverflow
                    tickFormatter={(value) => Math.round(value).toString()}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name="t-SNE Dimension 2" 
                    label={{ value: 't-SNE Dimension 2', angle: -90, position: 'insideLeft' }} 
                    domain={getYDomain()}
                    allowDataOverflow
                    tickFormatter={(value) => Math.round(value).toString()}
                  />
                  <ZAxis 
                    type="number" 
                    dataKey="mse" 
                    range={[20, 200]} 
                    name="MSE" 
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Scatter 
                    name={colorFeature === 'mse' ? 'MSE Value' : `Grouped by ${colorFeature}`}
                    data={filteredData} 
                    fill="#8884d8"
                    shape="circle"
                    fillOpacity={0.7}
                  >
                    {filteredData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getPointColor(entry)} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
              
              {/* Custom selection overlay */}
              <SelectionOverlay 
                isSelecting={isSelecting} 
                onSelectionComplete={handleSelectionComplete} 
                containerRef={chartContainerRef}
              />
            </>
          ) : (
            <p className="text-center py-10">
              {loading ? "Loading data..." : "No t-SNE data available for selected folder"}
            </p>
          )}
        </CardContent>
      </Card>
      
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          The t-SNE visualization shows the 2D projection of the latent space representation from the autoencoder.
          Points are sized by their MSE value (larger points = higher MSE) and can be colored by MSE or feature values.
          <strong className="ml-1">New:</strong> Click "Select Area" and drag on the chart to zoom into specific regions. Use "Reset Zoom" to return to the full view.
          {zoomDomain.x && filteredData.length < scatterData.length && (
            <p className="mt-1 text-xs text-blue-600">
              Showing {filteredData.length} of {scatterData.length} points for better performance ({Math.round(filteredData.length/scatterData.length*100)}%)
            </p>
          )}
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default TSNETab;