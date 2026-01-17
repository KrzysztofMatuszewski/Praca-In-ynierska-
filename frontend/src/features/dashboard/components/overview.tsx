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
} from '@/types/nids';

// Components
import FolderSelector from '@/components/common/FolderSelectorNids';
import MetadataTab from '@/components/tabs/nids/MetadataTab';
import AnomaliesTab from '@/components/tabs/nids/AnomaliesTab';
import TrainingTab from '@/components/tabs/nids/TrainingTab';
import FeaturesTab from '@/components/tabs/nids/FeaturesTab';
import DistributionTab from '@/components/tabs/nids/DistributionTab';
import WeightsTab from '@/components/tabs/nids/WeightsTab';
import TSNETab from '@/components/tabs/nids/TSNETab';

const NIDSDashboard: React.FC = () => {
  // State for NIDS data
  const [nidsData, setNidsData] = useState<{ [folder: string]: ResultItem[] }>({});
  const [nidsLossData, setNidsLossData] = useState<{ [folder: string]: LossDataItem[] }>({});
  const [nidsFeatureImportance, setNidsFeatureImportance] = useState<{ [folder: string]: FeatureImportanceItem[] }>({});
  const [nidsFeatureDistribution, setNidsFeatureDistribution] = useState<{ [folder: string]: { [featureName: string]: FeatureDistributionItem[] } }>({});
  const [weightsHeatmapData, setWeightsHeatmapData] = useState<{ [folder: string]: WeightsHeatmapData }>({});

  // Training metadata
  const [trainingMetadata, setTrainingMetadata] = useState<{ [folder: string]: TrainingMetadata }>({});
  
  // Folder selection states
  const [selectedNidsFolders, setSelectedNidsFolders] = useState<string[]>([]);
  
  // Feature visualization selection states
  const [selectedNidsFolder, setSelectedNidsFolder] = useState<string>('');
  const [selectedNidsFeature, setSelectedNidsFeature] = useState<string>('');
  
  // Thresholds
  const [nidsThresholds, setNidsThresholds] = useState<{[folder: string]: number}>({});
  
  const [tsneData, setTsneData] = useState<{ [folder: string]: TSNEData }>({});

  // Available folders
  const [availableNidsFolders, setAvailableNidsFolders] = useState<string[]>([]);
  
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      
      // Construct URL with query parameters for selected folders
      let url = 'http://localhost:8012/model-results';
      const params = new URLSearchParams();
      
      // Join multiple folder selections with commas
      if (selectedNidsFolders.length > 0) {
        params.append('nids_folders', selectedNidsFolders.join(','));
      }
      
      if (params.toString()) {
        url += `?${params.toString()}`;
      }
      
      const response = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if (!response.ok) throw new Error('Failed to fetch data');
      
      const data: ApiResponse = await response.json();
      
      if (data.nids.tsne_data) {
        setTsneData(data.nids.tsne_data);
      }

      if (data.nids.weights_heatmap_data) {
        setWeightsHeatmapData(data.nids.weights_heatmap_data);
      }
      
      // Set available folders
      setAvailableNidsFolders(data.available_folders.nids || []);
      
      // Set thresholds by folder
      setNidsThresholds(data.nids.thresholds || {});
      
      // Set selected folders if they're not already set
      if (selectedNidsFolders.length === 0 && data.nids.folders?.length > 0) {
        setSelectedNidsFolders(data.nids.folders);
        setSelectedNidsFolder(data.nids.folders[0]);
      }
      
      // Set results data
      setNidsData(data.nids.results || {});
      
      // Set loss history data
      setNidsLossData(data.nids.loss_history || {});
      
      // Set feature importance data
      setNidsFeatureImportance(data.nids.feature_importance || {});
      
      // Set feature distribution data
      setNidsFeatureDistribution(data.nids.feature_distribution || {});
      
      // Set training metadata
      if (data.nids.training_metadata) {
        setTrainingMetadata(data.nids.training_metadata);
      } else {
        // Fallback do mock data 
        const mockTrainingMetadata: TrainingMetadata = {
          training_date: "2025-03-18 16:03:30",
          data_shape: [74847, 4],
          epochs: 50,
          batch_size: 50,
          final_threshold: 0.0008186792205899516,
          columns_used: ["destination.port", "source.port", "destination.ip", "source.ip"],
          roc_auc: 0.923 // Przykładowa wartość AUC
        };
        
        const mockMetadataByFolder: { [key: string]: TrainingMetadata } = {};
        data.nids.folders.forEach(folder => {
          mockMetadataByFolder[folder] = {...mockTrainingMetadata};
        });
        
        setTrainingMetadata(mockMetadataByFolder);
      }
      
      // Set default selected folder and feature if not already set
      if (selectedNidsFolder === '' && data.nids.folders?.length > 0) {
        const firstFolder = data.nids.folders[0];
        setSelectedNidsFolder(firstFolder);
        
        // Set first feature from this folder
        if (data.nids.feature_distribution && data.nids.feature_distribution[firstFolder]) {
          const featuresForFolder = Object.keys(data.nids.feature_distribution[firstFolder] || {});
          if (featuresForFolder.length > 0) {
            setSelectedNidsFeature(featuresForFolder[0]);
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
        <h1 className="text-2xl font-bold">NIDS Performance Dashboard</h1>
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
            selectedFolders={selectedNidsFolders}
            loading={loading}
          />
        </TabsContent>
        
        <TabsContent value="anomalies">
          <AnomaliesTab 
            nidsData={nidsData}
            nidsThresholds={nidsThresholds}
            selectedFolders={selectedNidsFolders}
            loading={loading}
          />
        </TabsContent>

        <TabsContent value="training">
          <TrainingTab 
            nidsLossData={nidsLossData}
            loading={loading}
          />
        </TabsContent>
          
        <TabsContent value="features">
          <FeaturesTab 
            nidsFeatureImportance={nidsFeatureImportance}
            loading={loading}
          />
        </TabsContent>

        <TabsContent value="distribution">
          <DistributionTab 
            nidsFeatureDistribution={nidsFeatureDistribution}
            selectedNidsFeature={selectedNidsFeature}
            setSelectedNidsFeature={setSelectedNidsFeature}
            loading={loading}
          />
        </TabsContent>
        
        <TabsContent value="weights">
          <WeightsTab 
            weightsHeatmapData={weightsHeatmapData}
            selectedNidsFolder={selectedNidsFolder}
            setSelectedNidsFolder={setSelectedNidsFolder}
            loading={loading}
          />
        </TabsContent>
        <TabsContent value="tsne">
          <TSNETab 
            tsneData={tsneData}
            selectedFolder={selectedNidsFolder}
            setSelectedFolder={setSelectedNidsFolder}
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
            availableFolders={availableNidsFolders}
            selectedFolders={selectedNidsFolders}
            onChange={(selected: string[]) => {
              setSelectedNidsFolders(selected);
              if (selected.length > 0 && !selectedNidsFolder) {
                setSelectedNidsFolder(selected[0]);
              }
            }}
            loading={loading}
            label="Model Folders"
            placeholder="Search for model folders..."
          />
          {selectedNidsFolders.length > 0 && (
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

export default NIDSDashboard;