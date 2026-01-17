import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info } from 'lucide-react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { WeightsHeatmapData } from '@/types/nids';
import NetworkWeightsHeatmap from '@/components/visualizations/NetworkWeightsHeatmap';

interface WeightsTabProps {
  weightsHeatmapData: { [folder: string]: WeightsHeatmapData };
  selectedNidsFolder: string;
  setSelectedNidsFolder: (folder: string) => void;
  loading: boolean;
}

const WeightsTab: React.FC<WeightsTabProps> = ({ 
  weightsHeatmapData,
  selectedNidsFolder,
  setSelectedNidsFolder,
  loading 
}) => {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Neural Network Weights Heatmap</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Wybór folderu do wizualizacji wag */}
          <div className="mb-4">
            <label className="text-sm font-medium mb-2 block">Select Model Folder:</label>
            <Select 
              value={selectedNidsFolder} 
              onValueChange={setSelectedNidsFolder}
              disabled={Object.keys(weightsHeatmapData).length === 0}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a model folder" />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(weightsHeatmapData).map(folder => (
                  <SelectItem key={folder} value={folder}>{folder}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {/* Renderowanie komponentu mapy ciepła wag */}
          {Object.keys(weightsHeatmapData).length > 0 && selectedNidsFolder ? (
            <NetworkWeightsHeatmap 
              weightsData={weightsHeatmapData}
              selectedFolder={selectedNidsFolder}
            />
          ) : (
            <p className="text-center py-10">
              {loading ? "Loading weights data..." : "No weights data available"}
            </p>
          )}
        </CardContent>
      </Card>
      
      <Alert className="mt-4">
        <Info className="h-4 w-4" />
        <AlertDescription>
          The weights heatmap visualizes the learned connections between neurons in the autoencoder network.
          Darker blue cells represent strong negative weights, darker red cells represent strong positive weights.
          This visualization helps identify which features have the strongest influence on the network's hidden representations.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default WeightsTab;