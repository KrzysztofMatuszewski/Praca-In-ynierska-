// model_panel/index.tsx
import React from 'react';
import Overview from './components/overview';

const ModelPanel: React.FC = () => {
  return (
    <div className="container mx-auto">
      <Overview />
    </div>
  );
};

export default ModelPanel;