import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import { FeatureImportanceItem } from '@/types/nids';
import { prepareFeatureImportanceComparison } from '@/utils/dataTransformers';

interface FeaturesTabProps {
  nidsFeatureImportance: { [folder: string]: FeatureImportanceItem[] };
  loading: boolean;
}

const FeaturesTab: React.FC<FeaturesTabProps> = ({ 
  nidsFeatureImportance,
  loading 
}) => {
  return (
    <div className="space-y-4">
      {/* Feature Importance Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Feature Importance Comparison</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            {Object.keys(nidsFeatureImportance).length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={prepareFeatureImportanceComparison(nidsFeatureImportance)}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis 
                    type="category" 
                    dataKey="feature" 
                    width={100}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip 
                    formatter={(value) => [value.toFixed(4), 'Importance']}
                  />
                  <Legend />
                  <ReferenceLine x={0} stroke="#000" />
                  {/* Create a bar for each folder */}
                  {Object.keys(nidsFeatureImportance).map((folder, index) => (
                    <Bar 
                      key={folder}
                      dataKey={folder} 
                      name={folder}
                      fill={`hsl(${index * 40}, 50%, 60%)`}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-center py-10">
                {loading ? "Loading data..." : "No feature importance data available"}
              </p>
            )}
          </div>
        </CardContent>
      </Card>
      
      <Alert className="mt-4">
        <Info className="h-4 w-4" />
        <AlertDescription>
          Feature importance shows the impact of each feature on model predictions across different runs.
          Positive values indicate features that increase prediction values, while negative values decrease them.
          Comparing importance across runs helps identify which features are consistently important to the model.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default FeaturesTab;