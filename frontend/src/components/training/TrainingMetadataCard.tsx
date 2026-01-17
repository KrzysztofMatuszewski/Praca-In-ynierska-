import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Calendar, Layers, Hash, Cpu, AlertCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { TrainingMetadataCardProps } from '@/types/nids';

const TrainingMetadataCard: React.FC<TrainingMetadataCardProps> = ({ 
  trainingMetadata,
  folder 
}) => {
  if (!trainingMetadata) {
    return (
      <Card className="mb-4">
        <CardHeader>
          <CardTitle className="text-lg">Training Metadata: {folder}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center py-4 text-muted-foreground">No training metadata available for this folder</p>
        </CardContent>
      </Card>
    );
  }

  // Format date string for better readability
  const formattedDate = new Date(trainingMetadata.training_date).toLocaleString();
  
  return (
    <Card className="mb-4">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">Training Metadata: {folder}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-center space-x-3">
            <Calendar className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium">Training Date</p>
              <p className="text-sm text-muted-foreground">{formattedDate}</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <Layers className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium">Dataset Shape</p>
              <p className="text-sm text-muted-foreground">
                {trainingMetadata.data_shape.join(' × ')} (rows × features)
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <Hash className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium">Epochs</p>
              <p className="text-sm text-muted-foreground">{trainingMetadata.epochs}</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <Cpu className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium">Batch Size</p>
              <p className="text-sm text-muted-foreground">{trainingMetadata.batch_size}</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <AlertCircle className="h-5 w-5 text-blue-500" />
            <div>
              <p className="text-sm font-medium">Final Threshold</p>
              <p className="text-sm text-muted-foreground">
                {trainingMetadata.final_threshold.toExponential(6)}
              </p>
            </div>
          </div>
        </div>
        
        <div className="mt-4">
          <p className="text-sm font-medium mb-2">Columns Used ({trainingMetadata.columns_used.length})</p>
          <div className="flex flex-wrap gap-2">
            {trainingMetadata.columns_used.map(column => (
              <Badge key={column} variant="outline">{column}</Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TrainingMetadataCard;