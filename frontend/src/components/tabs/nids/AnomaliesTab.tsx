import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { ResultItem } from '@/types/nids';
import { prepareMultiSeriesData } from '@/utils/dataTransformers';
import EnhancedMultiSeriesTooltip from '@/components/visualizations/EnhancedMultiSeriesTooltip';

interface AnomaliesTabProps {
  nidsData: { [folder: string]: ResultItem[] };
  nidsThresholds: { [folder: string]: number };
  selectedFolders: string[];
  loading: boolean;
}

const AnomaliesTab: React.FC<AnomaliesTabProps> = ({ 
  nidsData, 
  nidsThresholds, 
  selectedFolders,
  loading 
}) => {
  return (
    <div className="space-y-4">
      {/* NIDS Chart */}
      <Card>
        <CardHeader>
          <CardTitle>NIDS Results ({selectedFolders.join(', ')})</CardTitle>
        </CardHeader>
        <CardContent className="h-64">
          {Object.keys(nidsData).length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={prepareMultiSeriesData(nidsData)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" label={{ value: 'Anomaly Index', position: 'insideBottom', offset: -5 }} />
                <YAxis />
                <Tooltip content={<EnhancedMultiSeriesTooltip />} />
                <Legend />
                {/* Create a line for each folder */}
                {Object.keys(nidsData).map((folder, index) => (
                  <Line 
                    key={folder}
                    type="monotone" 
                    dataKey={folder} 
                    stroke={`hsl(${index * 40}, 50%, 60%)`} 
                    name={folder} 
                    strokeWidth={2}
                    dot={{ r: 3 }}
                    activeDot={{ r: 5 }}
                  />
                ))}
                {/* Display thresholds */}
                {Object.entries(nidsThresholds).map(([folder, threshold], index) => (
                  <ReferenceLine 
                    key={folder}
                    y={threshold} 
                    stroke={`hsl(${index * 40}, 50%, 60%)`} 
                    strokeDasharray="5 5" 
                    label={`Threshold: ${threshold.toFixed(6)}`} 
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-center py-10">
              {loading ? "Loading data..." : "No anomaly detection data available"}</p>
          )}
        </CardContent>
        {Object.keys(nidsThresholds).length > 0 && (
          <div className="text-center text-sm text-gray-600 p-2">
            {Object.entries(nidsThresholds).map(([folder, threshold]) => (
              <p key={folder}>Threshold for {folder}: {threshold.toFixed(6)}</p>
            ))}
          </div>
        )}
      </Card>
      
      <Alert className="mt-4">
        <Info className="h-4 w-4" />
        <AlertDescription>
          The anomaly detection chart shows the Mean Squared Error (MSE) for each data point. 
          Values above the threshold line are classified as anomalies.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default AnomaliesTab;