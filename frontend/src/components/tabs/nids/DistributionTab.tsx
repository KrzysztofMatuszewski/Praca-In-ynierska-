import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { FeatureDistributionItem } from '@/types/nids';
import { prepareFeatureDistributionComparison } from '@/utils/dataTransformers';

interface DistributionTabProps {
  nidsFeatureDistribution: { [folder: string]: { [featureName: string]: FeatureDistributionItem[] } };
  selectedNidsFeature: string;
  setSelectedNidsFeature: (feature: string) => void;
  loading: boolean;
}

const DistributionTab: React.FC<DistributionTabProps> = ({ 
  nidsFeatureDistribution,
  selectedNidsFeature,
  setSelectedNidsFeature,
  loading 
}) => {
  return (
    <div className="space-y-4">
      {/* Feature Distribution Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          {/* Select feature to compare */}
          <div className="mb-4">
            <label className="text-sm font-medium mb-2 block">Select Feature:</label>
            <Select 
              value={selectedNidsFeature} 
              onValueChange={setSelectedNidsFeature}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a feature" />
              </SelectTrigger>
              <SelectContent>
                {/* Get all unique features across all folders */}
                {Array.from(new Set(
                  Object.values(nidsFeatureDistribution)
                    .flatMap(folder => Object.keys(folder))
                )).map(feature => (
                  <SelectItem key={feature} value={feature}>{feature}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {selectedNidsFeature && Object.keys(nidsFeatureDistribution).some(folder => 
            nidsFeatureDistribution[folder] && nidsFeatureDistribution[folder][selectedNidsFeature]) ? (
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={prepareFeatureDistributionComparison(nidsFeatureDistribution, selectedNidsFeature)}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis 
                    type="category" 
                    dataKey="displayValue" 
                    width={100}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip formatter={(value, name) => {
                    // Extract folder name and type (normal/anomaly) from the dataKey
                    const [folder, type] = name.split('_');
                    return [`${value} ${type} events`, folder];
                  }} labelFormatter={(value) => {
                    // Find the original full value from the data
                    const item = prepareFeatureDistributionComparison(nidsFeatureDistribution, selectedNidsFeature)
                      .find(item => item.displayValue === value);
                    return item ? `Feature value: ${item.value}` : value;
                  }} />
                  <Legend />
                  
                  {/* Create normal and anomaly bars for each folder */}
                  {Object.keys(nidsFeatureDistribution).map((folder, index) => [
                    // Normal events (lighter color)
                    <Bar 
                      key={`${folder}_normal`}
                      dataKey={`${folder}_normal`} 
                      name={`${folder}_normal`}
                      stackId={`${folder}`}
                      fill={`hsl(${index * 40}, 40%, 85%)`}
                    />,
                    // Anomaly events (darker color)
                    <Bar 
                      key={`${folder}_anomaly`}
                      dataKey={`${folder}_anomaly`} 
                      name={`${folder}_anomaly`}
                      stackId={`${folder}`}
                      fill={`hsl(${index * 40}, 70%, 50%)`}
                    />
                  ]).flat()}
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-center py-10">
              {loading ? "Loading data..." : 
               !selectedNidsFeature ? "Please select a feature" :
               "No feature distribution data available"}
            </p>
          )}
          
          <div className="mt-4 text-sm text-gray-500">
            <p>For each folder: Lighter colors represent normal events, darker colors represent anomalies.</p>
          </div>
        </CardContent>
      </Card>
      
      <Alert className="mt-4">
        <Info className="h-4 w-4" />
        <AlertDescription>
          Feature distribution shows how values of each feature are distributed between normal and anomalous samples across different model runs.
          This helps identify which feature values consistently trigger anomaly detections in multiple model versions.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default DistributionTab;