#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 Simplified Test: Build Knowledge Graph Based on Domain Configuration
Focus on Core Functions: Clear Neo4j + Build Corresponding Domain Knowledge Graph According to Domain Configuration File
"""

import sys
import os
import argparse
import time
import yaml
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger_utils import get_logger
from src.utils.neo4j_connection import get_neo4j_connection
from src.services.knowledge_graph.knowledge_graph_retriever import KnowledgeGraphRetriever

class SimpleKGBuilder:
    """Simplified Knowledge Graph Builder"""
    
    def __init__(self, domain: str):
        self.domain = domain
        self.logger = get_logger(self.__class__.__name__)
        self.domain_config = None
        self.neo4j_manager = None
        self.kg_service = None
        
    def load_domain_config(self) -> dict:
        """Load domain configuration file directly"""
        config_path = Path(__file__).parent.parent / "configs" / "domains" / f"{self.domain}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Domain configuration file does not exist: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
        
    def initialize(self) -> bool:
        """Initialize components"""
        try:
            self.logger.info("🔧 Initializing components...")
            
            # Load domain configuration directly
            self.domain_config = self.load_domain_config()
            self.logger.info(f"  ✅ {self.domain} domain configuration loaded successfully")
            
            # Initialize Neo4j connection
            self.neo4j_manager = get_neo4j_connection()
            self.logger.info("  ✅ Neo4j connection initialized successfully")
            
            # Read retry configuration
            retry_config = self.domain_config.get('retrieval_params', {}).get('retry_config', {})
            self.logger.info(f"  🔄 Retry configuration: {retry_config}")
            
            # Initialize knowledge graph retrieval service (pass retry configuration)
            self.kg_service = KnowledgeGraphRetriever(self.neo4j_manager, retry_config)
            self.logger.info("  ✅ Knowledge graph retrieval service initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"  ❌ Component initialization failed: {e}")
            return False
    
    def clear_neo4j(self) -> bool:
        """Clear Neo4j database"""
        try:
            self.logger.info("🗑️  Clearing Neo4j database...")
            
            # Use knowledge graph retriever's clear method
            if self.kg_service.clear_graph():
                self.logger.info("  ✅ Neo4j database cleared successfully")
                return True
            else:
                self.logger.error("  ❌ Neo4j database clearing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"  ❌ Failed to clear Neo4j: {e}")
            return False
    
    def build_knowledge_graph(self) -> bool:
        """Build knowledge graph based on domain configuration"""
        try:
            self.logger.info(f"🏗️  Building {self.domain} domain knowledge graph...")
            
            # Get domain seed entities
            seed_ids = self.domain_config['wikidata_seeds']['seeds']
            self.logger.info(f"  📍 Seed entities: {len(seed_ids)} entities")
            
            # Get retrieval parameters
            retrieval_params = self.domain_config['retrieval_params']
            max_depth = retrieval_params.get('max_depth', 3)
            min_sitelinks = retrieval_params.get('min_sitelinks', 50)
            
            self.logger.info(f"  🔍 Retrieval parameters: max_depth={max_depth}, min_sitelinks={min_sitelinks}")
            
            # Use knowledge graph retriever to build graph
            stats = self.kg_service.initialize_graph(
                seed_nodes=seed_ids,
                max_depth=max_depth,
                min_sitelinks=min_sitelinks
            )
            
            self.logger.info(f"  🎉 Knowledge graph construction completed!")
            self.logger.info(f"  📊 Construction results: {stats.node_count} entities, {stats.relationship_count} relationships")
            
            return stats.node_count > 0
            
        except Exception as e:
            self.logger.error(f"  ❌ Knowledge graph construction failed: {e}")
            return False
    
    def verify_results(self) -> dict:
        """Verify construction results"""
        try:
            self.logger.info("🔍 Verifying construction results...")
            
            # Get graph statistics
            stats = self.kg_service.get_graph_stats()
            
            result = {
                'node_count': stats.node_count,
                'relationship_count': stats.relationship_count,
                'node_labels': stats.node_labels,
                'relationship_types': stats.relationship_types
            }
            
            self.logger.info(f"  📊 Final statistics:")
            self.logger.info(f"    Node count: {result['node_count']}")
            self.logger.info(f"    Relationship count: {result['relationship_count']}")
            self.logger.info(f"    Node labels: {result['node_labels']}")
            self.logger.info(f"    Relationship types: {result['relationship_types']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"  ❌ Result verification failed: {e}")
            return {}
    
    def run_test(self) -> bool:
        """Run complete test"""
        self.logger.info(f"\n🚀 Starting Step 1: {self.domain} domain knowledge graph construction")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # 1. Initialize components
        if not self.initialize():
            return False
        
        # 2. Clear Neo4j
        if not self.clear_neo4j():
            return False
        
        # 3. Build knowledge graph
        if not self.build_knowledge_graph():
            return False
        
        # 4. Verify results
        stats = self.verify_results()
        
        # 5. Summary
        end_time = time.time()
        duration = end_time - start_time
        
        self.logger.info("=" * 60)
        if stats.get('node_count', 0) > 0:
            self.logger.info(f"✅ Step 1 successful: {self.domain} domain knowledge graph construction completed")
            self.logger.info(f"⏱️  Duration: {duration:.2f} seconds")
            return True
        else:
            self.logger.error(f"❌ Step 1 failed: Unable to build valid knowledge graph")
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 1: Build knowledge graph based on domain configuration')
    parser.add_argument('--domain', type=str, default='medicine', help='Domain name')
    
    args = parser.parse_args()
    
    # Create and run test
    builder = SimpleKGBuilder(args.domain)
    success = builder.run_test()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()