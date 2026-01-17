import { ResultItem as NIDSResultItem, LossDataItem as NIDSLossDataItem, FeatureImportanceItem as NIDSFeatureImportanceItem, FeatureDistributionItem as NIDSFeatureDistributionItem, ResultItem, FeatureImportanceItem } from '@/types/nids';
import { ResultItem as HIDSResultItem, LossDataItem as HIDSLossDataItem, FeatureImportanceItem as HIDSFeatureImportanceItem, FeatureDistributionItem as HIDSFeatureDistributionItem, LossDataItem, FeatureDistributionItem } from '@/types/hids';

// Transform data for multi-series line charts
export const prepareMultiSeriesData = (resultsObject: { [folder: string]: ResultItem[] }) => {
  // First create a map of all mse values by folder
  const folderSeriesMap: { [key: string]: { [key: string]: any } } = {};
  
  // Create data points for each folder
  Object.entries(resultsObject).forEach(([folderName, results]) => {
    results.forEach((item, index) => {
      // Create an entry for each data point with a common x-axis index
      if (!folderSeriesMap[index]) {
        folderSeriesMap[index] = { index };
      }
      
      // Add this folder's mse value at this index
      // Use the folder name as the key for this data point
      folderSeriesMap[index][folderName] = item.mse;
      
      // Preserve the original data for the tooltip
      folderSeriesMap[index][`${folderName}_data`] = item;
    });
  });
  
  // Convert map to array
  return Object.values(folderSeriesMap);
};

// Prepare training history data for multi-series visualization
export const prepareMultiSeriesLossData = (lossDataObject: { [folder: string]: LossDataItem[] }) => {
  // First find the maximum number of epochs across all folders
  let maxEpochs = 0;
  Object.values(lossDataObject).forEach(folderData => {
    if (folderData.length > 0) {
      const lastEpoch = folderData[folderData.length - 1].epochs;
      maxEpochs = Math.max(maxEpochs, lastEpoch);
    }
  });
  
  // Create an array with one entry per epoch
  const multiSeriesData = [];
  for (let i = 1; i <= maxEpochs; i++) {
    const dataPoint: { epoch: number; [key: string]: number } = { epoch: i };
    
    // For each folder, find the data for this epoch
    Object.entries(lossDataObject).forEach(([folderName, folderData]) => {
      // Find the entry for this epoch
      const epochData = folderData.find(entry => entry.epochs === i);
      if (epochData) {
        // Add training loss
        dataPoint[`${folderName}_loss`] = epochData.loss;
        // Add validation loss
        dataPoint[`${folderName}_val_loss`] = epochData.val_loss;
      }
    });
    
    multiSeriesData.push(dataPoint);
  }
  
  return multiSeriesData;
};

// Prepare feature importance data for comparison across folders
export const prepareFeatureImportanceComparison = (importanceData: { [folder: string]: FeatureImportanceItem[] }) => {
  // Create a map of all unique features across all folders
  const allFeatures = new Set<string>();
  Object.values(importanceData).forEach(folderData => {
    folderData.forEach(item => {
      allFeatures.add(item.feature);
    });
  });
  
  // Convert to array of feature objects
  const comparisonData = Array.from(allFeatures).map(feature => {
    // Start with the feature name
    const featureData: { feature: string; [key: string]: any } = { feature };
    
    // Add importance values for each folder
    Object.entries(importanceData).forEach(([folderName, folderData]) => {
      // Find this feature in the folder's data
      const featureItem = folderData.find(item => item.feature === feature);
      // Add importance value from this folder (or 0 if not present)
      featureData[folderName] = featureItem ? featureItem.importance : 0;
    });
    
    return featureData;
  });
  
  // Sort by the sum of absolute importance values across all folders (most important first)
  comparisonData.sort((a, b) => {
    const sumA = Object.entries(a)
      .filter(([key]) => key !== 'feature')
      .reduce((sum, [_, value]) => sum + Math.abs(value), 0);
      
    const sumB = Object.entries(b)
      .filter(([key]) => key !== 'feature')
      .reduce((sum, [_, value]) => sum + Math.abs(value), 0);
      
    return sumB - sumA;
  });
  
  // Return top 15 features
  return comparisonData.slice(0, 15);
};

// Prepare feature distribution data for comparison across folders
export const prepareFeatureDistributionComparison = (
  distributionData: { [folder: string]: { [feature: string]: FeatureDistributionItem[] } }, 
  selectedFeature: string
) => {
  if (!selectedFeature) {
    return [];
  }
  
  // Get all folders that have data for this feature
  const relevantFolders = Object.keys(distributionData).filter(
    folder => distributionData[folder] && distributionData[folder][selectedFeature]
  );
  
  if (relevantFolders.length === 0) {
    return [];
  }
  
  // Get all unique feature values across all folders
  const allValues = new Set<string>();
  
  relevantFolders.forEach(folder => {
    distributionData[folder][selectedFeature].forEach(item => {
      // Get the feature value (first property that's not normal, anomaly, or total)
      const featureKey = Object.keys(item).find(key => !['normal', 'anomaly', 'total'].includes(key));
      if (featureKey) {
        allValues.add(String(item[featureKey]));
      }
    });
  });
  
  // Convert to comparison data
  return Array.from(allValues).map(value => {
    const comparisonItem: { 
      value: string | number; 
      displayValue: string; 
      [key: string]: string | number 
    } = { 
      value: value,
      displayValue: typeof value === 'string' && value.length > 20 ? value.substring(0, 20) + '...' : String(value)
    };
    
    // Add data for each folder
    relevantFolders.forEach(folder => {
      const valueData = distributionData[folder][selectedFeature].find(item => {
        const featureKey = Object.keys(item).find(key => !['normal', 'anomaly', 'total'].includes(key));
        return featureKey && String(item[featureKey]) === value;
      });
      
      if (valueData) {
        comparisonItem[`${folder}_normal`] = valueData.normal;
        comparisonItem[`${folder}_anomaly`] = valueData.anomaly;
      } else {
        comparisonItem[`${folder}_normal`] = 0;
        comparisonItem[`${folder}_anomaly`] = 0;
      }
    });
    
    return comparisonItem;
  })
  // Sort by total anomalies across all folders
  .sort((a, b) => {
    const aTotal = Object.keys(a)
      .filter(key => key.endsWith('_anomaly'))
      .reduce((sum, key) => sum + (typeof a[key] === 'number' ? a[key] as number : 0), 0);
      
    const bTotal = Object.keys(b)
      .filter(key => key.endsWith('_anomaly'))
      .reduce((sum, key) => sum + (typeof b[key] === 'number' ? b[key] as number : 0), 0);
      
    return bTotal - aTotal;
  })
  // Limit to top 15 values
  .slice(0, 15);
};