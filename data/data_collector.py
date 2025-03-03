"""
Data collection module for financial translation system.
Handles acquisition of parallel texts from various sources.
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Optional
import yaml

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects parallel text data from specified sources."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """
        Initialize the data collector.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['data']
        
        self.sources = self.config['sources']
        self.output_dir = "data/processed"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def collect_from_source(self, source_name: str) -> pd.DataFrame:
        """
        Collect data from a specific source.
        
        Args:
            source_name: Name of the source to collect from
            
        Returns:
            DataFrame with parallel texts
        """
        source = next((s for s in self.sources if s['name'] == source_name), None)
        if not source:
            raise ValueError(f"Source {source_name} not found in configuration")
        
        source_path = source['path']
        logger.info(f"Collecting data from {source_name} at {source_path}")
        
        # Implementation depends on source type
        # This is a simplified example for file-based sources
        if source_name == "IMF Reports":
            return self._collect_from_imf(source_path)
        elif source_name == "Central Bank Documents":
            return self._collect_from_central_bank(source_path)
        elif source_name == "Financial News":
            return self._collect_from_news(source_path)
        else:
            raise NotImplementedError(f"Collection from {source_name} not implemented")
    
    def _collect_from_imf(self, path: str) -> pd.DataFrame:
        """Collect data from IMF reports."""
        # Placeholder for actual implementation
        logger.info(f"Collecting IMF data from {path}")
        # In a real implementation, this would parse IMF documents
        # and extract parallel sentences
        return pd.DataFrame({
            'english': ['Sample IMF text 1', 'Sample IMF text 2'],
            'arabic': ['عينة نص صندوق النقد الدولي 1', 'عينة نص صندوق النقد الدولي 2'],
            'source': ['IMF', 'IMF'],
            'document_id': ['imf001', 'imf001']
        })
    
    def _collect_from_central_bank(self, path: str) -> pd.DataFrame:
        """Collect data from central bank documents."""
        # Placeholder for actual implementation
        logger.info(f"Collecting central bank data from {path}")
        return pd.DataFrame({
            'english': ['Sample central bank text'],
            'arabic': ['عينة نص البنك المركزي'],
            'source': ['Central Bank'],
            'document_id': ['cb001']
        })
    
    def _collect_from_news(self, path: str) -> pd.DataFrame:
        """Collect data from financial news sources."""
        # Placeholder for actual implementation
        logger.info(f"Collecting financial news from {path}")
        return pd.DataFrame({
            'english': ['Sample financial news'],
            'arabic': ['عينة أخبار مالية'],
            'source': ['News'],
            'document_id': ['news001']
        })
    
    def collect_all(self) -> pd.DataFrame:
        """
        Collect data from all configured sources.
        
        Returns:
            Combined DataFrame with all parallel texts
        """
        
        en_folder = "data/raw/en"
        ar_folder = "data/raw/ar"
        en_files = sorted(os.listdir(en_folder))
        ar_files = sorted(os.listdir(ar_folder))
        paired_data = []
        
        # Assuming matching sorted order (or adjust with a more robust matching mechanism)
        for en_file, ar_file in zip(en_files, ar_files):
            with open(os.path.join(en_folder, en_file), 'r', encoding='utf-8') as f:
                en_text = f.read()
            with open(os.path.join(ar_folder, ar_file), 'r', encoding='utf-8') as f:
                ar_text = f.read()
            paired_data.append({
                'english': en_text,
                'arabic': ar_text,
                'document_id': en_file.split('.')[0],
                'source': 'manual'
            })
        
        df = pd.DataFrame(paired_data)
        output_path = os.path.join(self.output_dir, "raw_parallel_data.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Raw data saved to {output_path}")
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = DataCollector()
    data = collector.collect_all()
    print(f"Collected {len(data)} parallel text entries") 