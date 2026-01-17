import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { RefreshCw, AlertCircle } from 'lucide-react';

// Types
import { 
  ResultItem, 
  LossDataItem, 
  FeatureImportanceItem, 
  FeatureDistributionItem, 
  TrainingMetadata, 
  ApiResponse,
  WeightsHeatmapData,
  TSNEData
} from '@/types/hids';

// Components
import FolderSelector from '@/components/common/FolderSelectorHids';
import MetadataTab from '@/components/tabs/hids/MetadataTab';
import AnomaliesTab from '@/components/tabs/hids/AnomaliesTab';
import TrainingTab from '@/components/tabs/hids/TrainingTab';
import FeaturesTab from '@/components/tabs/hids/FeaturesTab';
import DistributionTab from '@/components/tabs/hids/DistributionTab';
import WeightsTab from '@/components/tabs/hids/WeightsTab';
import TSNETab from '@/components/tabs/hids/TSNETab';

const HIDSDashboard: React.FC = () => {
  // State for HIDS data
  const [hidsData, setHidsData] = useState<{ [folder: string]: ResultItem[] }>({});
  const [hidsLossData, setHidsLossData] = useState<{ [folder: string]: LossDataItem[] }>({});
  const [hidsFeatureImportance, setHidsFeatureImportance] = useState<{ [folder: string]: FeatureImportanceItem[] }>({});
  const [hidsFeatureDistribution, setHidsFeatureDistribution] = useState<{ [folder: string]: { [featureName: string]: FeatureDistributionItem[] } }>({});
  const [weightsHeatmapData, setWeightsHeatmapData] = useState<{ [folder: string]: WeightsHeatmapData }>({});

  // Training metadata
  const [trainingMetadata, setTrainingMetadata] = useState<{ [folder: string]: TrainingMetadata }>({});
  
  // Folder selection states
  const [selectedHidsFolders, setSelectedHidsFolders] = useState<string[]>([]);
  
  // Feature visualization selection states
  const [selectedHidsFolder, setSelectedHidsFolder] = useState<string>('');
  const [selectedHidsFeature, setSelectedHidsFeature] = useState<string>('');
  
  // Thresholds
  const [hidsThresholds, setHidsThresholds] = useState<{[folder: string]: number}>({});
  
  const [tsneData, setTsneData] = useState<{ [folder: string]: TSNEData }>({});

  // Available folders
  const [availableHidsFolders, setAvailableHidsFolders] = useState<string[]>([]);
  
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Construct URL with query parameters for selected folders
      let url = 'http://localhost:8012/model-results';
      const params = new URLSearchParams();
      
      // Join multiple folder selections with commas
      if (selectedHidsFolders.length > 0) {
        params.append('hids_folders', selectedHidsFolders.join(','));
      }
      
      if (params.toString()) {
        url += `?${params.toString()}`;
      }
      
      const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (!response.ok) throw new Error('Failed to fetch data');
      
      const data: ApiResponse = await response.json();

      // W funkcji fetchData, po pobraniu danych z API
      if (data.hids.tsne_data) {
        setTsneData(data.hids.tsne_data);
      }
      
      if (data.hids.weights_heatmap_data) {
        setWeightsHeatmapData(data.hids.weights_heatmap_data);
      }
      
      // Set available folders
      setAvailableHidsFolders(data.available_folders.hids || []);
      
      // Set thresholds by folder
      setHidsThresholds(data.hids.thresholds || {});
      
      // Set selected folders if they're not already set
      if (selectedHidsFolders.length === 0 && data.hids.folders?.length > 0) {
        setSelectedHidsFolders(data.hids.folders);
        setSelectedHidsFolder(data.hids.folders[0]);
      }
      
      // Set results data
      setHidsData(data.hids.results || {});
      
      // Set loss history data
      setHidsLossData(data.hids.loss_history || {});
      
      // Set feature importance data
      setHidsFeatureImportance(data.hids.feature_importance || {});
      
      // Set feature distribution data
      setHidsFeatureDistribution(data.hids.feature_distribution || {});
      
      // Set training metadata
      if (data.hids.training_metadata) {
        setTrainingMetadata(data.hids.training_metadata);
      } else {
        // Fallback do mock data 
        const mockTrainingMetadata: TrainingMetadata = {
          training_date: "2025-03-18 16:03:30",
          data_shape: [74847, 2],
          epochs: 50,
          batch_size: 50,
          final_threshold: 0.0008186792205899516,
          columns_used: ["agent.id", "rule.description"],
        };
        
        const mockMetadataByFolder: { [key: string]: TrainingMetadata } = {};
        data.hids.folders.forEach(folder => {
          mockMetadataByFolder[folder] = {...mockTrainingMetadata};
        });
        
        setTrainingMetadata(mockMetadataByFolder);
      }
      
      // Set default selected folder and feature if not already set
      if (selectedHidsFolder === '' && data.hids.folders?.length > 0) {
        const firstFolder = data.hids.folders[0];
        setSelectedHidsFolder(firstFolder);
        
        // Set first feature from this folder
        if (data.hids.feature_distribution && data.hids.feature_distribution[firstFolder]) {
          const featuresForFolder = Object.keys(data.hids.feature_distribution[firstFolder] || {});
          if (featuresForFolder.length > 0) {
            setSelectedHidsFeature(featuresForFolder[0]);
          }
        }
      }
      
      setError(null);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An error occurred while fetching data';
      setError(errorMessage);
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    
    // Refresh data every minute
    const interval = setInterval(fetchData, 600000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full space-y-4 p-8">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">HIDS Performance Dashboard</h1>
        <Button 
          variant="outline" 
          onClick={fetchData} 
          disabled={loading}
          className="flex items-center gap-2"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh Data
        </Button>
      </div>
      
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
  
      <Tabs defaultValue="metadata" className="w-full">
        <TabsList className="flex mb-4 space-x-1 justify-start ">
          <TabsTrigger 
            value="metadata" 
            className="px-4 py-2 text-sm font-medium"
          >
            Training Metadata
          </TabsTrigger>
          {/* <TabsTrigger 
            value="anomalies" 
            className="px-4 py-2 text-sm font-medium"
          >
            Anomaly Detection
          </TabsTrigger>
          <TabsTrigger 
            value="training" 
            className="px-4 py-2 text-sm font-medium"
          >
            Training History
          </TabsTrigger>
          <TabsTrigger 
            value="features" 
            className="px-4 py-2 text-sm font-medium"
          >
            Feature Importance
          </TabsTrigger>
          <TabsTrigger 
            value="distribution" 
            className="px-4 py-2 text-sm font-medium"
          >
            Feature Distribution
          </TabsTrigger>
          <TabsTrigger 
            value="weights" 
            className="px-4 py-2 text-sm font-medium"
          >
            Network Weights
          </TabsTrigger>
          <TabsTrigger 
            value="tsne" 
            className="px-4 py-2 text-sm font-medium"
          >
            t-SNE Visualization
          </TabsTrigger> */}
        </TabsList>
        
        <TabsContent value="metadata">
          <MetadataTab 
            trainingMetadata={trainingMetadata} 
            selectedFolders={selectedHidsFolders}
            loading={loading}
          />
        </TabsContent>
        
        <TabsContent value="anomalies">
          <AnomaliesTab 
            hidsData={hidsData}
            hidsThresholds={hidsThresholds}
            selectedFolders={selectedHidsFolders}
            loading={loading}
          />
        </TabsContent>

        <TabsContent value="training">
          <TrainingTab 
            hidsLossData={hidsLossData}
            loading={loading}
          />
        </TabsContent>
          
        <TabsContent value="features">
          <FeaturesTab 
            hidsFeatureImportance={hidsFeatureImportance}
            loading={loading}
          />
        </TabsContent>

        <TabsContent value="distribution">
          <DistributionTab 
            hidsFeatureDistribution={hidsFeatureDistribution}
            selectedHidsFeature={selectedHidsFeature}
            setSelectedHidsFeature={setSelectedHidsFeature}
            loading={loading}
          />
        </TabsContent>
        
        <TabsContent value="weights">
          <WeightsTab 
            weightsHeatmapData={weightsHeatmapData}
            selectedHidsFolder={selectedHidsFolder}
            setSelectedHidsFolder={setSelectedHidsFolder}
            loading={loading}
          />
        </TabsContent>
        <TabsContent value="tsne">
          <TSNETab 
            tsneData={tsneData}
            selectedFolder={selectedHidsFolder}
            setSelectedFolder={setSelectedHidsFolder}
            loading={loading}
          />
        </TabsContent>
      </Tabs>
      
      {/* Folder Selection Controls */}
      <Card className="mb-4">
        <CardHeader className="pb-2">
          <CardTitle>Select Model Folders</CardTitle>
        </CardHeader>
        <CardContent>
          <FolderSelector
            availableFolders={availableHidsFolders}
            selectedFolders={selectedHidsFolders}
            onChange={(selected: string[]) => {
              setSelectedHidsFolders(selected);
              if (selected.length > 0 && !selectedHidsFolder) {
                setSelectedHidsFolder(selected[0]);
              }
            }}
            loading={loading}
            label="Model Folders"
            placeholder="Search for model folders..."
          />
          {selectedHidsFolders.length > 0 && (
            <Button 
              variant="outline" 
              size="sm" 
              className="mt-4"
              onClick={fetchData}
              disabled={loading}
            >
              Apply Selection
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default HIDSDashboard;