import React from 'react';
import { TooltipProps } from '@/types/nids';

// Enhanced tooltip for multi-series charts that shows data from all series at a point
const EnhancedMultiSeriesTooltip: React.FC<TooltipProps> = ({ 
  active, 
  payload, 
  label 
}) => {
  if (active && payload && payload.length) {
    // Group all series by folder name (removing _data suffix from keys)
    const folderData: { [key: string]: { mse: number; originalData: any } } = {};
    
    // Extract all folder names and their data
    payload.forEach(series => {
      const folderName = series.dataKey;
      const value = series.value;
      
      if (value !== undefined && value !== null) {
        // Get original data object if available
        const originalData = series.payload[`${folderName}_data`];
        
        folderData[folderName] = {
          mse: value,
          originalData: originalData || {}
        };
      }
    });
    
    return (
      <div className="custom-tooltip bg-white p-3 border rounded shadow max-w-md overflow-auto" style={{ maxHeight: '300px' }}>
        <p className="font-bold border-b pb-1 mb-2">Anomaly Index: {label}</p>
        
        {/* List each folder's data */}
        {Object.entries(folderData).map(([folder, data], index) => (
          <div key={folder} className="mb-3 last:mb-0">
            <div 
              className="font-semibold pb-1 mb-1 border-b" 
              style={{ borderColor: `hsl(${index * 40}, 50%, 60%)` }}
            >
              {folder}: MSE = {data.mse.toFixed(6)}
              {data.originalData.is_anomaly !== undefined && (
                <span className="ml-2">
                  (Anomaly: {data.originalData.is_anomaly ? 'Yes' : 'No'})
                </span>
              )}
            </div>
            
            {/* Show additional data for this folder if available */}
            {data.originalData && (
              <div className="pl-2 text-sm">
                {Object.entries(data.originalData)
                  .filter(([key]) => !['mse', 'is_anomaly', '_data'].includes(key) && 
                                     !key.startsWith('_'))
                  .map(([key, value]) => (
                    <p key={key}>
                      <span className="font-medium">{key}: </span>
                      {String(value)}
                    </p>
                  ))
                }
              </div>
            )}
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export default EnhancedMultiSeriesTooltip;