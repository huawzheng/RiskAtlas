#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 1: Wikidata Knowledge Graph Retrieval and Construction

This module provides functionality for retrieving and constructing knowledge graphs from Wikidata:
- Uses categorized query strategy: generates independent SPARQL queries for four relationship types
- Supports four relationship types: instance_of(P31), subclass_of(P279), part_of(P361), has_part(P527)
- Supports recursive queries with maximum depth of 3 layers (depth>3 automatically falls back to depth 3)
- Supports Wikipedia existence and multi-language version count filtering
- Automatically retrieves English Wikipedia description for each node and saves as node property
- Provides complete data processing pipeline ensuring semantic accuracy
"""

import logging
from typing import Dict, Any, Optional, Union, List
from ..utils.neo4j_client_full import Neo4jClient
from ..utils.file_utils import save_json
from ..utils.common_utils import ensure_dir
from ..utils.wikipedia_api import WikipediaAPI

logger = logging.getLogger(__name__)


class WikidataRetriever:
    """
    Wikidata Knowledge Graph Retriever
    
    Provides complete functionality for retrieving domain-specific knowledge graphs from Wikidata, including:
    - Categorized query strategy: generates independent queries for four relationship types, ensuring semantic accuracy
    - Multiple relationship type support (instance_of, subclass_of, part_of, has_part)
    - Recursive depth queries (supports depth 1-3, depth>3 automatically falls back to depth 3)
    - Wikipedia existence and multi-language version filtering
    - Automatic retrieval and saving of English Wikipedia descriptions as node properties
    - Flexible configuration parameter support
    
    Attributes:
        neo4j_client: Neo4j database client
        
    Examples:
        >>> retriever = WikidataRetriever()
        >>> config = {
        ...     'domain': 'technology',
        ...     'entity_ids': ['Q11032', 'Q11053'],
        ...     'max_depth': 3,
        ...     'min_sitelinks': 5
        ... }
        >>> result = retriever.run_experiments(config)
        
    Node Properties:
        - name: Entity's English label
        - wikipedia_summary: Entity's Wikipedia English summary
    """
    
    
    def __init__(self, neo4j_client=None, wikipedia_api: WikipediaAPI = None, 
                 max_retries: int = 5, retry_delay: float = 2.0):
        """
        Initialize WikidataRetriever instance
        
        Args:
            neo4j_client: Optional Neo4j client instance, creates new instance if not provided
            wikipedia_api: Optional Wikipedia API instance
            max_retries: Maximum retry attempts, default 5
            retry_delay: Retry delay base, default 2 seconds
        """
        if neo4j_client is not None:
            self.neo4j_client = neo4j_client
        else:
            # Create Neo4j client with default configuration
            self.neo4j_client = Neo4jClient(
                uri="bolt://localhost:7687",
                user="neo4j", 
                password="12345678"
            )
        
        if wikipedia_api is not None:
            self.wikipedia_api = wikipedia_api
        else:
            self.wikipedia_api = WikipediaAPI(
                max_retries=max_retries, 
                retry_delay=retry_delay
            )
        
        # Save retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _build_conservative_depth3_queries_by_type(self, wiki_nodes: List[str], min_sitelinks: int) -> List[str]:
        """Build categorized queries for depth 3"""
        # Build CONSTRUCT and WHERE clauses separately for each starting node
        construct_parts = []
        where_parts = []
        
        for i, node in enumerate(wiki_nodes):
            # Create unique variable names for each node
            parent_label_var = f"?parentLabel{i}"
            parent_desc_var = f"?parentDescription{i}"
            
            # CONSTRUCT part
            construct_parts.append(f"""
        wd:{node} a neo:node .
        wd:{node} neo:node {parent_label_var} .
        wd:{node} neo:description {parent_desc_var} .""")
            
            # WHERE part
            where_parts.append(f"""
        wd:{node} rdfs:label {parent_label_var} .
        FILTER(LANG({parent_label_var}) = "en")
        
        OPTIONAL {{
            wd:{node} schema:description {parent_desc_var} .
            FILTER(LANG({parent_desc_var}) = "en")
        }}""")
        
        # Merge all parent node construction parts
        all_parents_construct = "\n        ".join(construct_parts)
        all_parents_where = "\n        ".join(where_parts)
        
        # Build VALUES clause
        wiki_nodes_values = " ".join([f"wd:{node}" for node in wiki_nodes])
        
        queries = []
        
        # Create separate queries for each relationship type, directly using semantic relationship names
        # Assign different LIMITs based on relationship frequency in Wikidata
        relations = [
            ("P31", "neo:instance_of", 3000),    # instance_of most common
            ("P279", "neo:subclass_of", 3000),   # subclass_of very common  
            ("P361", "neo:part_of", 1000),       # part_of less common
            ("P527", "neo:has_part", 1000)       # has_part less common
        ]
        
        for prop, neo_rel, limit in relations:
            # Depth 3 query: add three layers
            query = f"""PREFIX neo: <neo4j://voc#>
PREFIX schema: <http://schema.org/>
    CONSTRUCT {{
        {all_parents_construct}
        
        # Layer 1
        ?child1 a neo:node .
        ?child1 neo:node ?childLabel1 .
        ?child1 neo:description ?childDescription1 .
        ?parent {neo_rel} ?child1 .
        
        # Layer 2
        ?child2 a neo:node .
        ?child2 neo:node ?childLabel2 .
        ?child2 neo:description ?childDescription2 .
        ?child1 {neo_rel} ?child2 .
        
        # Layer 3
        ?child3 a neo:node .
        ?child3 neo:node ?childLabel3 .
        ?child3 neo:description ?childDescription3 .
        ?child2 {neo_rel} ?child3 .
    }}
    WHERE {{
        {all_parents_where}
        
        VALUES ?parent {{ {wiki_nodes_values} }}

        # Layer 1 query
        ?child1 wdt:{prop} ?parent .
        ?child1 rdfs:label ?childLabel1 .
        FILTER(LANG(?childLabel1) = "en")
        
        OPTIONAL {{
            ?child1 schema:description ?childDescription1 .
            FILTER(LANG(?childDescription1) = "en")
        }}
        
        FILTER EXISTS {{
            ?article1 schema:about ?child1 ;
                       schema:inLanguage "en" ;
                       schema:isPartOf <https://en.wikipedia.org/> .
        }}
        
        ?child1 wikibase:sitelinks ?sitelinks1 .
        FILTER(?sitelinks1 >= {min_sitelinks})
        
        # Layer 2 query
        OPTIONAL {{
            ?child2 wdt:{prop} ?child1 .
            ?child2 rdfs:label ?childLabel2 .
            FILTER(LANG(?childLabel2) = "en")
            
            OPTIONAL {{
                ?child2 schema:description ?childDescription2 .
                FILTER(LANG(?childDescription2) = "en")
            }}
            
            FILTER EXISTS {{
                ?article2 schema:about ?child2 ;
                           schema:inLanguage "en" ;
                           schema:isPartOf <https://en.wikipedia.org/> .
            }}
            
            ?child2 wikibase:sitelinks ?sitelinks2 .
            FILTER(?sitelinks2 >= {min_sitelinks})
            
            # Layer 3 query
            OPTIONAL {{
                ?child3 wdt:{prop} ?child2 .
                ?child3 rdfs:label ?childLabel3 .
                FILTER(LANG(?childLabel3) = "en")
                
                OPTIONAL {{
                    ?child3 schema:description ?childDescription3 .
                    FILTER(LANG(?childDescription3) = "en")
                }}
                
                FILTER EXISTS {{
                    ?article3 schema:about ?child3 ;
                               schema:inLanguage "en" ;
                               schema:isPartOf <https://en.wikipedia.org/> .
                }}
                
                ?child3 wikibase:sitelinks ?sitelinks3 .
                FILTER(?sitelinks3 >= {min_sitelinks})
            }}
        }}
    }}
    LIMIT {limit}"""
            queries.append(query)
        
        return queries

    def initialize_graph_mixed_semantic(self, wiki_nodes: Union[str, List[str]], depth: int = 3, min_sitelinks: int = 5) -> Dict[str, Any]:
        """
        Initialize knowledge graph using categorized query method: generate independent SPARQL queries for each relationship type
        
        Args:
            wiki_nodes: Wikidata entity ID or ID list, e.g., 'Q11032' or ['Q11032', 'Q12136']
            depth: Recursive query depth, default 3
            min_sitelinks: Minimum Wikipedia language version count, default 5
            
        Returns:
            Dict[str, Any]: Dictionary containing operation statistics
            
        Notes:
            Core strategy: Categorized queries, generate independent queries for four relationship types (P31, P279, P361, P527)
            Ensure each relationship has correct semantic name, avoid semantic ambiguity from mixed relationship types
            Supports depth 1-3, depth>3 automatically falls back to depth 3 (identical to depth 3 implementation)
        """
        # Ensure wiki_nodes is list format for log display
        if isinstance(wiki_nodes, str):
            display_nodes = [wiki_nodes]
        else:
            display_nodes = wiki_nodes
            
        logger.info(f"Starting knowledge graph initialization with categorized query method, wiki_nodes={display_nodes}, depth={depth}, min_sitelinks={min_sitelinks}")
        logger.info("Strategy: Categorized queries, generate independent queries for each relationship type, ensuring semantic accuracy")
        
        stats = {}
        
        # 1. Clear database
        logger.info("Clearing existing database")
        self.neo4j_client.clear_database()
        stats['cleared_database'] = True
        
        # 2. Initialize n10s configuration
        logger.info("Initializing n10s RDF plugin configuration")
        self.neo4j_client.init_n10s_config()
        stats['n10s_initialized'] = True
        
        # 3. Build categorized queries, generate independent SPARQL queries for each relationship type
        logger.info("Building categorized SPARQL queries")
        
        # Use categorized queries, generate queries for four relationship types separately
        sparql_queries = self._build_comprehensive_mixed_queries(display_nodes, depth, min_sitelinks)
        stats['sparql_queries_count'] = len(sparql_queries)
        
        # Execute each categorized query, ensure complete relationship type coverage
        relation_names = ['P31(instance_of)', 'P279(subclass_of)', 'P361(part_of)', 'P527(has_part)']
        for i, query in enumerate(sparql_queries):
            if i < len(relation_names):
                rel_name = relation_names[i]
                logger.info(f"Executing {rel_name} categorized query...")
                self.neo4j_client.import_rdf_from_sparql(
                    query, 
                    max_retries=self.max_retries, 
                    retry_delay=self.retry_delay
                )
        
        stats['rdf_imported'] = True
        
        # 4. Data post-processing
        logger.info("Renaming node properties")
        self.neo4j_client.rename_node_properties()
        stats['properties_renamed'] = True
        
        logger.info("Removing isolated nodes")
        self.neo4j_client.remove_isolated_nodes()
        stats['isolated_nodes_removed'] = True
        
        # 5. Get current node count, add Wikipedia summaries for all nodes
        node_count = self.neo4j_client.get_node_count()
        logger.info("Adding Wikipedia summaries")
        enrich_stats = self.enrich_nodes_with_wikipedia_summaries(
            batch_size=10, 
            max_nodes=node_count  # Process all nodes
        )
        stats['wikipedia_summary_stats'] = enrich_stats
        
        # 6. Collect statistics
        relationship_count = self.neo4j_client.get_relationship_count()
        
        stats.update({
            'final_node_count': node_count,
            'final_relationship_count': relationship_count,
            'wiki_nodes': display_nodes,
            'depth': depth,
            'min_sitelinks': min_sitelinks,
            'layers_built': depth  # Record built depth
        })
        
        logger.info(f"Categorized knowledge graph initialization complete, nodes: {node_count}, relationships: {relationship_count}")
        return stats
    
    def _build_comprehensive_mixed_queries(self, wiki_nodes: List[str], depth: int, min_sitelinks: int) -> List[str]:
        """
        Build categorized queries: generate independent SPARQL queries for each relationship type
        
        Strategy: Categorized query method, generate independent queries for four relationship types (P31, P279, P361, P527),
        Ensure each relationship has correct semantic name, avoid semantic ambiguity from mixed relationships
        """
        
        # Use categorized queries, select appropriate query method based on depth
        return self._build_enhanced_mixed_type_queries(wiki_nodes, depth, min_sitelinks)
    
    def _build_enhanced_mixed_type_queries(self, wiki_nodes: List[str], depth: int, min_sitelinks: int) -> List[str]:
        """
        Build enhanced categorized queries: route to corresponding categorized query method based on depth
        
        Strategy: Unified categorized query routing, generate independent queries for each relationship type
        Supports depth 1-3, depth>3 automatically falls back to depth 3 (identical to depth 3 implementation)
        """
        
        if depth == 1:
            return self._build_conservative_depth1_queries_by_type(wiki_nodes, min_sitelinks)
        elif depth == 2:
            return self._build_conservative_depth2_queries_by_type(wiki_nodes, min_sitelinks)
        elif depth == 3:
            return self._build_conservative_depth3_queries_by_type(wiki_nodes, min_sitelinks)
        else:
            # For depth>3, fall back to depth 3 directly, identical to depth 3 implementation
            logger.info(f"Requested depth is {depth}, automatically falling back to depth 3")
            return self._build_conservative_depth3_queries_by_type(wiki_nodes, min_sitelinks)
    
    def enrich_nodes_with_wikipedia_summaries(self, batch_size: int = 10, max_nodes: int = None) -> Dict[str, Any]:
        """
        Add Wikipedia summary information to knowledge graph nodes
        
        Args:
            batch_size: Number of nodes to process per batch
            max_nodes: Maximum number of nodes to process, None means process all nodes
            
        Returns:
            Dict[str, Any]: Dictionary containing processing statistics
            
        Notes:
            This method will:
            1. Get nodes without Wikipedia summaries
            2. Construct Wikipedia URL based on node name
            3. Call Wikipedia API to get summary
            4. Save summary as node's wikipedia_summary property
        """
        logger.info("Starting to add Wikipedia summaries to nodes")
        
        stats = {
            'total_processed': 0,
            'successful_updates': 0,
            'failed_requests': 0,
            'not_found': 0,
            'errors': []
        }
        
        # Get nodes that need summaries added
        # If max_nodes is None, get all nodes without summaries
        if max_nodes is None:
            # Get count of all nodes without summaries
            with self.neo4j_client.driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WHERE n.name IS NOT NULL AND n.name <> '' 
                    AND (n.wikipedia_summary IS NULL OR n.wikipedia_summary = '')
                    RETURN count(n) as count
                """)
                max_nodes = result.single()["count"]
        
        nodes_to_process = self.neo4j_client.get_nodes_without_wikipedia_summary(max_nodes)
        
        if not nodes_to_process:
            logger.info("All nodes already have Wikipedia summaries")
            return stats
        
        logger.info(f"Found {len(nodes_to_process)} nodes that need summaries added")
        
        # Process nodes in batches
        for i in range(0, len(nodes_to_process), batch_size):
            batch = nodes_to_process[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}, {len(batch)} nodes")
            
            for node_name in batch:
                try:
                    stats['total_processed'] += 1
                    
                    # Skip empty names
                    if not node_name or not node_name.strip():
                        logger.warning("Skipping empty node name")
                        continue
                    
                    logger.debug(f"Processing node: {node_name}")
                    
                    # Get Wikipedia summary
                    summary = self.wikipedia_api.get_wikipedia_summary(node_name)
                    
                    if summary:
                        # Update node property
                        updated_count = self.neo4j_client.add_wikipedia_summary_to_node(node_name, summary)
                        if updated_count > 0:
                            stats['successful_updates'] += 1
                            logger.info(f"Successfully added Wikipedia summary to node '{node_name}'")
                        else:
                            logger.warning(f"Node '{node_name}' update failed, may not exist")
                    else:
                        stats['not_found'] += 1
                        logger.info(f"Node '{node_name}' Wikipedia page not found")
                        
                except Exception as e:
                    stats['failed_requests'] += 1
                    error_msg = f"Error processing node '{node_name}': {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg)
        
        # Log final statistics
        logger.info(f"Wikipedia summary addition complete:")
        logger.info(f"  Total processed: {stats['total_processed']}")
        logger.info(f"  Successful updates: {stats['successful_updates']}")
        logger.info(f"  Pages not found: {stats['not_found']}")
        logger.info(f"  Failed requests: {stats['failed_requests']}")
        
        return stats
    
    def _build_conservative_depth1_queries_by_type(self, wiki_nodes: List[str], min_sitelinks: int) -> List[str]:
        """Build categorized queries for depth 1"""
        # Build CONSTRUCT and WHERE clauses separately for each starting node
        construct_parts = []
        where_parts = []
        
        for i, node in enumerate(wiki_nodes):
            # Create unique variable names for each node
            parent_label_var = f"?parentLabel{i}"
            parent_desc_var = f"?parentDescription{i}"
            
            # CONSTRUCT part
            construct_parts.append(f"""
        wd:{node} a neo:node .
        wd:{node} neo:node {parent_label_var} .
        wd:{node} neo:description {parent_desc_var} .""")
            
            # WHERE part
            where_parts.append(f"""
        wd:{node} rdfs:label {parent_label_var} .
        FILTER(LANG({parent_label_var}) = "en")
        
        OPTIONAL {{
            wd:{node} schema:description {parent_desc_var} .
            FILTER(LANG({parent_desc_var}) = "en")
        }}""")
        
        # Merge all parent node construction parts
        all_parents_construct = "\n        ".join(construct_parts)
        all_parents_where = "\n        ".join(where_parts)
        
        # Build VALUES clause
        wiki_nodes_values = " ".join([f"wd:{node}" for node in wiki_nodes])
        
        queries = []
        
        # Create separate queries for each relationship type
        relations = [
            ("P31", "neo:instance_of", 2000),    # instance_of most common
            ("P279", "neo:subclass_of", 2000),   # subclass_of very common  
            ("P361", "neo:part_of", 1000),       # part_of less common
            ("P527", "neo:has_part", 1000)       # has_part less common
        ]
        
        for prop, neo_rel, limit in relations:
            # Depth 1 query: add only one layer
            query = f"""PREFIX neo: <neo4j://voc#>
PREFIX schema: <http://schema.org/>
    CONSTRUCT {{
        {all_parents_construct}
        
        # Layer 1
        ?child1 a neo:node .
        ?child1 neo:node ?childLabel1 .
        ?child1 neo:description ?childDescription1 .
        ?parent {neo_rel} ?child1 .
    }}
    WHERE {{
        {all_parents_where}
        
        VALUES ?parent {{ {wiki_nodes_values} }}

        # Layer 1 query
        ?child1 wdt:{prop} ?parent .
        ?child1 rdfs:label ?childLabel1 .
        FILTER(LANG(?childLabel1) = "en")
        
        OPTIONAL {{
            ?child1 schema:description ?childDescription1 .
            FILTER(LANG(?childDescription1) = "en")
        }}
        
        FILTER EXISTS {{
            ?article1 schema:about ?child1 ;
                       schema:inLanguage "en" ;
                       schema:isPartOf <https://en.wikipedia.org/> .
        }}
        
        ?child1 wikibase:sitelinks ?sitelinks1 .
        FILTER(?sitelinks1 >= {min_sitelinks})
    }}
    LIMIT {limit}"""
            queries.append(query)
        
        return queries
    
    def _build_conservative_depth2_queries_by_type(self, wiki_nodes: List[str], min_sitelinks: int) -> List[str]:
        """Build categorized queries for depth 2"""
        # Build CONSTRUCT and WHERE clauses separately for each starting node
        construct_parts = []
        where_parts = []
        
        for i, node in enumerate(wiki_nodes):
            # Create unique variable names for each node
            parent_label_var = f"?parentLabel{i}"
            parent_desc_var = f"?parentDescription{i}"
            
            # CONSTRUCT part
            construct_parts.append(f"""
        wd:{node} a neo:node .
        wd:{node} neo:node {parent_label_var} .
        wd:{node} neo:description {parent_desc_var} .""")
            
            # WHERE part
            where_parts.append(f"""
        wd:{node} rdfs:label {parent_label_var} .
        FILTER(LANG({parent_label_var}) = "en")
        
        OPTIONAL {{
            wd:{node} schema:description {parent_desc_var} .
            FILTER(LANG({parent_desc_var}) = "en")
        }}""")
        
        # Merge all parent node construction parts
        all_parents_construct = "\n        ".join(construct_parts)
        all_parents_where = "\n        ".join(where_parts)
        
        # Build VALUES clause
        wiki_nodes_values = " ".join([f"wd:{node}" for node in wiki_nodes])
        
        queries = []
        
        # Create separate queries for each relationship type
        relations = [
            ("P31", "neo:instance_of", 2500),    # instance_of most common
            ("P279", "neo:subclass_of", 2500),   # subclass_of very common  
            ("P361", "neo:part_of", 1000),       # part_of less common
            ("P527", "neo:has_part", 1000)       # has_part less common
        ]
        
        for prop, neo_rel, limit in relations:
            # Depth 2 query: add two layers
            query = f"""PREFIX neo: <neo4j://voc#>
PREFIX schema: <http://schema.org/>
    CONSTRUCT {{
        {all_parents_construct}
        
        # Layer 1
        ?child1 a neo:node .
        ?child1 neo:node ?childLabel1 .
        ?child1 neo:description ?childDescription1 .
        ?parent {neo_rel} ?child1 .
        
        # Layer 2
        ?child2 a neo:node .
        ?child2 neo:node ?childLabel2 .
        ?child2 neo:description ?childDescription2 .
        ?child1 {neo_rel} ?child2 .
    }}
    WHERE {{
        {all_parents_where}
        
        VALUES ?parent {{ {wiki_nodes_values} }}

        # Layer 1 query
        ?child1 wdt:{prop} ?parent .
        ?child1 rdfs:label ?childLabel1 .
        FILTER(LANG(?childLabel1) = "en")
        
        OPTIONAL {{
            ?child1 schema:description ?childDescription1 .
            FILTER(LANG(?childDescription1) = "en")
        }}
        
        FILTER EXISTS {{
            ?article1 schema:about ?child1 ;
                       schema:inLanguage "en" ;
                       schema:isPartOf <https://en.wikipedia.org/> .
        }}
        
        ?child1 wikibase:sitelinks ?sitelinks1 .
        FILTER(?sitelinks1 >= {min_sitelinks})
        
        # Layer 2 query
        OPTIONAL {{
            ?child2 wdt:{prop} ?child1 .
            ?child2 rdfs:label ?childLabel2 .
            FILTER(LANG(?childLabel2) = "en")
            
            OPTIONAL {{
                ?child2 schema:description ?childDescription2 .
                FILTER(LANG(?childDescription2) = "en")
            }}
            
            FILTER EXISTS {{
                ?article2 schema:about ?child2 ;
                           schema:inLanguage "en" ;
                           schema:isPartOf <https://en.wikipedia.org/> .
            }}
            
            ?child2 wikibase:sitelinks ?sitelinks2 .
            FILTER(?sitelinks2 >= {min_sitelinks})
        }}
    }}
    LIMIT {limit}"""
            queries.append(query)
        
        return queries