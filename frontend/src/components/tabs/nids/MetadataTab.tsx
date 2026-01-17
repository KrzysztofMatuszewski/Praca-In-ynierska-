import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info } from 'lucide-react';
import { TrainingMetadata } from '@/types/nids';
import TrainingMetadataCard from '@/components/training/TrainingMetadataCard';
import MetadataComparisonTable from '@/components/common/MetadataComparisonTable';

interface MetadataTabProps {
  trainingMetadata: { [folder: string]: TrainingMetadata };
  selectedFolders: string[];
  loading: boolean;
}

const MetadataTab: React.FC<MetadataTabProps> = ({ 
  trainingMetadata, 
  selectedFolders, 
  loading 
}) => {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Training Metadata Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          {Object.keys(trainingMetadata).length > 0 ? (
            <MetadataComparisonTable metadataByFolder={trainingMetadata} />
          ) : (
            <p className="text-center py-10">
              {loading ? "Loading training metadata..." : "No training metadata available"}
            </p>
          )}
        </CardContent>
      </Card>
      
      {/* Individual Metadata Cards */}
      {selectedFolders.length > 0 && (
        <div>
          <h3 className="text-lg font-medium mb-3">Individual Model Details</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {selectedFolders.map(folder => (
              <TrainingMetadataCard 
                key={folder} 
                folder={folder} 
                trainingMetadata={trainingMetadata[folder]} 
              />
            ))}
          </div>
        </div>
      )}
      
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          Training metadata shows the configuration and dataset details used to train each model.
          Comparing this information helps understand differences in model performance and behavior.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default MetadataTab;