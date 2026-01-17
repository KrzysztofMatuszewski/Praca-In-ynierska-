// Interface definitions for HIDS data

export interface WeightsHeatmapData {
    [layerName: string]: {
      weights: number[][];
      biases: number[];
      shape: number[];
      input_size: number;
      output_size: number;
      layer_type: string;
    };
  }

export interface ResultItem {
    mse: number;
    is_anomaly: boolean;
    [key: string]: any;
  }
  
  export interface LossDataItem {
    epochs: number;
    loss: number;
    val_loss: number;
  }
  
  export interface FeatureImportanceItem {
    feature: string;
    importance: number;
  }
  
  export interface FeatureDistributionItem {
    [key: string]: string | number;
    normal: number;
    anomaly: number;
    total: number;
  }
  
  export interface TrainingMetadata {
    training_date: string;
    data_shape: number[];
    epochs: number;
    batch_size: number;
    final_threshold: number;
    columns_used: string[];
  }
  
  export interface ModelData {
    weights_heatmap_data: any;
    folders: string[];
    results: { [folder: string]: ResultItem[] };
    thresholds: { [folder: string]: number };
    loss_history: { [folder: string]: LossDataItem[] };
    feature_importance: { [folder: string]: FeatureImportanceItem[] };
    feature_distribution: { [folder: string]: { [featureName: string]: FeatureDistributionItem[] } };
    columns_used: string[];
    training_metadata?: { [folder: string]: TrainingMetadata };
    tsne_data?: { [folder: string]: TSNEData };
  }
  
  export interface ApiResponse {
    hids: ModelData;
    available_folders: {
      hids: string[];
    };
  }
  
  export interface TooltipProps {
    active?: boolean;
    payload?: any[];
    label?: string | number;
  }
  
  export interface FolderSelectorProps {
    availableFolders: string[];
    selectedFolders: string[];
    onChange: (selected: string[]) => void;
    loading: boolean;
    label?: string;
    placeholder?: string;
  }
  
  export interface TrainingMetadataCardProps {
    trainingMetadata: TrainingMetadata | undefined;
    folder: string;
  }
  
  export interface MetadataComparisonTableProps {
    metadataByFolder: { [folder: string]: TrainingMetadata };
  }

  export interface TSNEPoint {
    coordinates: number[];
    mse: number;
    original_features: {
      [key: string]: string;
    };
  }
  
  export interface TSNEData {
    points: TSNEPoint[];
    metadata: {
      n_components: number;
      perplexity: number;
      data_points: number;
      latent_dimensions: number;
      features: string[];
    };
  }