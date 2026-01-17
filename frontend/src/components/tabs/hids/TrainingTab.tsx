import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { LossDataItem } from '@/types/hids';
import { prepareMultiSeriesLossData } from '@/utils/dataTransformers';

interface TrainingTabProps {
  hidsLossData: { [folder: string]: LossDataItem[] };
  loading: boolean;
}

const TrainingTab: React.FC<TrainingTabProps> = ({ 
  hidsLossData,
  loading 
}) => {
  return (
    <div className="space-y-4">
      {/* Training History Chart */}
      <Card>
        <CardHeader>
          <CardTitle>HIDS Training History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            {Object.keys(hidsLossData).length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={prepareMultiSeriesLossData(hidsLossData)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" label={{ value: 'Epochs', position: 'insideBottom', offset: -5 }} />
                  <YAxis label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  {/* Create training loss line for each folder */}
                  {Object.keys(hidsLossData).map((folder, index) => (
                    <Line 
                      key={`${folder}_loss`}
                      type="monotone" 
                      dataKey={`${folder}_loss`} 
                      stroke={`hsl(${index * 40}, 50%, 60%)`} 
                      name={`${folder} (Training)`} 
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                    />
                  ))}
                  {/* Create validation loss line for each folder (dashed) */}
                  {Object.keys(hidsLossData).map((folder, index) => (
                    <Line 
                      key={`${folder}_val_loss`}
                      type="monotone" 
                      dataKey={`${folder}_val_loss`} 
                      stroke={`hsl(${index * 40}, 50%, 60%)`} 
                      name={`${folder} (Validation)`} 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={{ r: 3 }}
                      activeDot={{ r: 5 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-center py-10">
                {loading ? "Loading data..." : "No training history available"}
              </p>
            )}
          </div>
        </CardContent>
      </Card>
      
      <Alert className="mt-4">
        <Info className="h-4 w-4" />
        <AlertDescription>
          Training and validation loss history shows model convergence over epochs. Lower values indicate better model fit.
          Solid lines represent training loss and dashed lines represent validation loss.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default TrainingTab;