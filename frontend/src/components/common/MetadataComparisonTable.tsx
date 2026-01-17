import React from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Button } from '@/components/ui/button';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { MetadataComparisonTableProps } from '@/types/nids';

const MetadataComparisonTable: React.FC<MetadataComparisonTableProps> = ({ 
  metadataByFolder 
}) => {
  const [openModelConfig, setOpenModelConfig] = React.useState<{[key: string]: boolean}>({});
  const [openAnomalyStats, setOpenAnomalyStats] = React.useState<{[key: string]: boolean}>({});

  if (Object.keys(metadataByFolder).length === 0) {
    return (
      <p className="text-center py-4 text-muted-foreground">No training metadata available</p>
    );
  }

  const toggleModelConfig = (folder: string) => {
    setOpenModelConfig(prev => ({
      ...prev,
      [folder]: !prev[folder]
    }));
  };

  const toggleAnomalyStats = (folder: string) => {
    setOpenAnomalyStats(prev => ({
      ...prev,
      [folder]: !prev[folder]
    }));
  };

  // Sprawdź, czy którykolwiek folder ma statystyki anomalii
  const hasAnomalyStats = Object.values(metadataByFolder).some(
    metadata => metadata.anomaly_statistics
  );

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Metric</TableHead>
            {Object.keys(metadataByFolder).map(folder => (
              <TableHead key={folder}>{folder}</TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell className="font-medium">Training Date</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder}>
                {new Date(metadata.training_date || metadata.retraining_date).toLocaleString()}
              </TableCell>
            ))}
          </TableRow>
          
          <TableRow>
            <TableCell className="font-medium">Dataset Size</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder}>
                {metadata.data_shape[0].toLocaleString()} rows
              </TableCell>
            ))}
          </TableRow>
          
          <TableRow>
            <TableCell className="font-medium">Features</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder}>
                {metadata.data_shape[1]}
              </TableCell>
            ))}
          </TableRow>
          
          {/* Nowy wiersz dla statystyk anomalii */}
          {hasAnomalyStats && (
            <TableRow>
              <TableCell className="font-medium">Anomaly Statistics</TableCell>
              {Object.entries(metadataByFolder).map(([folder, metadata]) => (
                <TableCell key={folder}>
                  {metadata.anomaly_statistics ? (
                    <Collapsible
                      open={openAnomalyStats[folder]}
                      onOpenChange={() => toggleAnomalyStats(folder)}
                      className="w-full"
                    >
                      <CollapsibleTrigger asChild>
                        <Button variant="ghost" size="sm" className="p-0 h-6">
                          {openAnomalyStats[folder] ? (
                            <ChevronDown className="h-4 w-4 mr-1" />
                          ) : (
                            <ChevronRight className="h-4 w-4 mr-1" />
                          )}
                          <Badge className="bg-red-100 text-red-800 hover:bg-red-200">
                            {metadata.anomaly_statistics.anomaly_percentage.toFixed(2)}% Anomalies
                          </Badge>
                        </Button>
                      </CollapsibleTrigger>
                      <CollapsibleContent className="mt-2 space-y-2 text-xs">
                        <div className="grid grid-cols-2 gap-1">
                          <span className="font-medium">Total Samples:</span>
                          <span>{metadata.anomaly_statistics.total_samples.toLocaleString()}</span>
                          
                          <span className="font-medium">Anomaly Count:</span>
                          <span>{metadata.anomaly_statistics.anomaly_count.toLocaleString()}</span>
                          
                          <span className="font-medium">Normal Count:</span>
                          <span>{metadata.anomaly_statistics.normal_count.toLocaleString()}</span>
                          
                          <span className="font-medium">Anomaly %:</span>
                          <span>{metadata.anomaly_statistics.anomaly_percentage.toFixed(2)}%</span>
                          
                          <span className="font-medium">Normal %:</span>
                          <span>{metadata.anomaly_statistics.normal_percentage.toFixed(2)}%</span>
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  ) : (
                    <span className="text-muted-foreground">N/A</span>
                  )}
                </TableCell>
              ))}
            </TableRow>
          )}
          
          <TableRow>
            <TableCell className="font-medium">Epochs</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder}>{metadata.epochs}</TableCell>
            ))}
          </TableRow>
          
          <TableRow>
            <TableCell className="font-medium">Batch Size</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder}>{metadata.batch_size}</TableCell>
            ))}
          </TableRow>
          
          <TableRow>
            <TableCell className="font-medium">Final Threshold</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder}>
                {metadata.final_threshold.toExponential(6)}
              </TableCell>
            ))}
          </TableRow>
          
          <TableRow>
            <TableCell className="font-medium">Columns Used</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder} className="max-w-xs">
                <div className="flex flex-wrap gap-1">
                  {metadata.columns_used.map(column => (
                    <Badge key={column} variant="outline" className="text-xs">{column}</Badge>
                  ))}
                </div>
              </TableCell>
            ))}
          </TableRow>

          {/* Row for base model */}
          <TableRow>
            <TableCell className="font-medium">Base Model</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder}>
                {metadata.base_model_used ? (
                  <Badge variant="secondary" className="text-xs">{metadata.base_model_used}</Badge>
                ) : (
                  <span className="text-muted-foreground">N/A</span>
                )}
              </TableCell>
            ))}
          </TableRow>

          {/* Row for model configuration */}
          <TableRow>
            <TableCell className="font-medium">Model Config</TableCell>
            {Object.entries(metadataByFolder).map(([folder, metadata]) => (
              <TableCell key={folder}>
                {metadata.model_config ? (
                  <Collapsible
                    open={openModelConfig[folder]}
                    onOpenChange={() => toggleModelConfig(folder)}
                    className="w-full"
                  >
                    <CollapsibleTrigger asChild>
                      <Button variant="ghost" size="sm" className="p-0 h-6">
                        {openModelConfig[folder] ? (
                          <ChevronDown className="h-4 w-4 mr-1" />
                        ) : (
                          <ChevronRight className="h-4 w-4 mr-1" />
                        )}
                        View Details
                      </Button>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-2 space-y-2 text-xs">
                      <div className="grid grid-cols-2 gap-1">
                        <span className="font-medium">Activation:</span>
                        <span>{metadata.model_config.activation}</span>
                        
                        <span className="font-medium">Output Activation:</span>
                        <span>{metadata.model_config.output_activation}</span>
                        
                        <span className="font-medium">Optimizer:</span>
                        <span>{metadata.model_config.optimizer}</span>
                        
                        <span className="font-medium">Loss:</span>
                        <span>{metadata.model_config.loss}</span>
                        
                        <span className="font-medium">Dropout Rate:</span>
                        <span>{metadata.model_config.dropout_rate}</span>
                      </div>
                      
                      <div>
                        <span className="font-medium">Layers:</span>
                        <div className="mt-1 pl-2 border-l-2 border-gray-200">
                          {metadata.model_config.layers.map((layer, idx) => (
                            <div key={idx} className="text-xs mb-1">
                              {layer.type} ({layer.units}) - {layer.activation}
                              {layer.dropout && `, dropout: ${layer.dropout}`}
                            </div>
                          ))}
                        </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                ) : (
                  <span className="text-muted-foreground">N/A</span>
                )}
              </TableCell>
            ))}
          </TableRow>

          {/* ROC AUC if available */}
          {Object.values(metadataByFolder).some(metadata => 'roc_auc' in metadata) && (
            <TableRow>
              <TableCell className="font-medium">ROC AUC</TableCell>
              {Object.entries(metadataByFolder).map(([folder, metadata]) => (
                <TableCell key={folder}>
                  {metadata.roc_auc !== undefined ? metadata.roc_auc.toFixed(3) : 'N/A'}
                </TableCell>
              ))}
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
};

export default MetadataComparisonTable;