"""
Main entry point for the Financial Translation POC.
"""

import os
import logging
import argparse
import yaml
from api.translation_service import TranslationService
from ui.app import TranslationUI

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Financial Translation POC")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", choices=["api", "ui", "both"], default="both", 
                        help="Run mode: api, ui, or both")
    parser.add_argument("--api-host", help="API host")
    parser.add_argument("--api-port", type=int, help="API port")
    parser.add_argument("--ui-host", help="UI host")
    parser.add_argument("--ui-port", type=int, help="UI port")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    api_host = args.api_host or config['api'].get('host', '0.0.0.0')
    api_port = args.api_port or config['api'].get('port', 8000)
    ui_host = args.ui_host or config['ui'].get('host', '0.0.0.0')
    ui_port = args.ui_port or config['ui'].get('port', 5000)
    
    # Run services based on mode
    if args.mode in ["api", "both"]:
        logger.info(f"Starting API service on {api_host}:{api_port}")
        api_service = TranslationService(args.config)
        
        # Start API in a separate process if running both
        if args.mode == "both":
            import multiprocessing
            api_process = multiprocessing.Process(
                target=api_service.run,
                args=(api_host, api_port)
            )
            api_process.start()
        else:
            api_service.run(host=api_host, port=api_port)
    
    if args.mode in ["ui", "both"]:
        logger.info(f"Starting UI service on {ui_host}:{ui_port}")
        ui_service = TranslationUI(args.config)
        ui_service.run(host=ui_host, port=ui_port)

if __name__ == "__main__":
    main() 