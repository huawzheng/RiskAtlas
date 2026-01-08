#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j Database Operation Utility Class
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import pandas as pd


class Neo4jClient:
    """Neo4j Database Client"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def clear_database(self):
        """Clear database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def init_n10s_config(self):
        """Initialize n10s RDF plugin configuration"""
        with self.driver.session() as session:
            session.run("""
                CALL n10s.graphconfig.init({
                    handleVocabUris: 'MAP',
                    handleMultival: 'OVERWRITE',
                    keepLangTag: false,
                    keepCustomDataTypes: false,
                    applyNeo4jNaming: false
                })
            """)
    
    def import_rdf_from_sparql(self, sparql_query: str, max_retries: int = 5, retry_delay: float = 2.0):
        """Import RDF data from SPARQL query with retry mechanism"""
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        for attempt in range(max_retries):
            try:
                with self.driver.session() as session:
                    logger.debug(f"🌐 Starting n10s.rdf.import.fetch (attempt {attempt + 1}/{max_retries})")
                    logger.debug(f"📝 Query length: {len(sparql_query)} characters")
                    
                    result = session.run(
                        """
                        CALL n10s.rdf.import.fetch(
                            'https://query.wikidata.org/sparql?query=' + apoc.text.urlencode($sparql),
                            'Turtle',
                            { headerParams: { Accept: 'application/x-turtle' } }
                        )
                        """,
                        sparql=sparql_query
                    )
                    
                    # Get import results
                    records = list(result)
                    logger.debug(f"📊 n10s import result record count: {len(records)}")
                    
                    for record in records:
                        logger.debug(f"📋 Import record: {dict(record)}")
                    
                    # Successfully executed, return
                    logger.info(f"✅ SPARQL import succeeded (attempt {attempt + 1}/{max_retries})")
                    return
                        
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"❌ SPARQL import failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                
                # Check if it's a retryable error
                if self._is_retryable_error(error_msg):
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"⏳ Waiting {wait_time:.1f} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                else:
                    # Non-retryable error, raise directly
                    logger.error(f"❌ Detected non-retryable error, stopping retry: {error_msg}")
                    raise
        
        # All retries failed
        logger.error(f"❌ SPARQL import finally failed after {max_retries} retries")
        raise Exception(f"SPARQL import failed after {max_retries} retries")
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """Determine if error is retryable"""
        # Define retryable error patterns
        retryable_patterns = [
            'timeout',
            'connection',
            'network',
            'temporary',
            'unavailable',
            'busy',
            'overload',
            '503',  # Service Unavailable
            '502',  # Bad Gateway  
            '504',  # Gateway Timeout
            'read timeout',
            'connect timeout'
        ]
        
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def rename_node_properties(self):
        """Rename node properties, including name and description"""
        with self.driver.session() as session:
            # Rename name property
            session.run("""
                MATCH (n:node)
                SET n.name = n.node
                REMOVE n.node
            """)
            
            # Rename description property
            session.run("""
                MATCH (n:node)
                WHERE n.description IS NOT NULL
                SET n.wikipedia_description = n.description
                REMOVE n.description
            """)
    
    def clean_duplicate_relationships(self):
        """Clean duplicate relationships after import, keep only one relationship type per node pair"""
        # This method should be called after RDF import and before relationship conversion
        # Because at this time relationship types are still P31, P279, P361, P527
        with self.driver.session() as session:
            # Check actual relationship types
            result = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type LIMIT 5")
            current_types = [record['rel_type'] for record in result]
            print(f"Current relationship types: {current_types}")
            
            if 'P31' in current_types:
                # Relationships not yet converted, delete duplicates of original relationship types
                print("Cleaning duplicates of original relationship types...")
                
                # Delete duplicate P527 relationships
                result = session.run("""
                    MATCH (s:node)-[r:P527]->(e:node)
                    WHERE EXISTS { (s)-[:P31|P279|P361]->(e) }
                    DELETE r
                    RETURN count(r) as deleted
                """)
                deleted_p527 = result.single()["deleted"]
                
                # Delete duplicate P361 relationships
                result = session.run("""
                    MATCH (s:node)-[r:P361]->(e:node)
                    WHERE EXISTS { (s)-[:P31|P279]->(e) }
                    DELETE r
                    RETURN count(r) as deleted
                """)
                deleted_p361 = result.single()["deleted"]
                
                # Delete duplicate P279 relationships
                result = session.run("""
                    MATCH (s:node)-[r:P279]->(e:node)
                    WHERE EXISTS { (s)-[:P31]->(e) }
                    DELETE r
                    RETURN count(r) as deleted
                """)
                deleted_p279 = result.single()["deleted"]
                
                total_deleted = deleted_p527 + deleted_p361 + deleted_p279
                if total_deleted > 0:
                    print(f"Cleaned {total_deleted} duplicate original relationships")
                    
            elif 'instance_of' in current_types:
                # Relationships already converted, only delete duplicate relationships between same node pairs
                print("Cleaning duplicates of converted relationship types (only deleting multiple relationships between same node pairs)...")
                
                # Delete duplicate has_part relationships between same node pairs if higher priority relationships exist
                result = session.run("""
                    MATCH (s:node)-[r:has_part]->(e:node)
                    WHERE EXISTS { 
                        MATCH (s)-[:instance_of|subclass_of|part_of]->(e)
                    }
                    DELETE r
                    RETURN count(r) as deleted
                """)
                deleted_has_part = result.single()["deleted"]
                
                # Delete duplicate part_of relationships between same node pairs if higher priority relationships exist
                result = session.run("""
                    MATCH (s:node)-[r:part_of]->(e:node)
                    WHERE EXISTS { 
                        MATCH (s)-[:instance_of|subclass_of]->(e)
                    }
                    DELETE r
                    RETURN count(r) as deleted
                """)
                deleted_part_of = result.single()["deleted"]
                
                # Delete duplicate subclass_of relationships between same node pairs if instance_of relationship exists
                result = session.run("""
                    MATCH (s:node)-[r:subclass_of]->(e:node)
                    WHERE EXISTS { 
                        MATCH (s)-[:instance_of]->(e)
                    }
                    DELETE r
                    RETURN count(r) as deleted
                """)
                deleted_subclass_of = result.single()["deleted"]
                
                total_deleted = deleted_has_part + deleted_part_of + deleted_subclass_of
                if total_deleted > 0:
                    print(f"Cleaned {total_deleted} duplicate relationships between same node pairs")
                    
                # Display relationship type distribution after cleanup
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as relation_type, count(r) as count
                    ORDER BY count DESC
                """)
                print("Relationship type distribution after cleanup:")
                for record in result:
                    print(f"  - {record['relation_type']}: {record['count']}")

    def reverse_relationships(self):
        """Reverse relationships and create correct relationship names based on original relationship types"""
        with self.driver.session() as session:
            # First query the actually existing relationship types in the database
            existing_relations = session.run("""
                MATCH ()-[r]->()
                RETURN DISTINCT type(r) as relation_type
            """).values()
            
            print(f"Discovered relationship types: {[r[0] for r in existing_relations]}")
            
            # Process P31 (instance_of) relationships
            result = session.run("""
                MATCH (s:node)-[r:P31]->(e:node)
                CREATE (e)-[r2:instance_of]->(s)
                SET r2 = r
                DELETE r
                RETURN count(r2) as processed
            """)
            p31_count = result.single()["processed"]
            if p31_count > 0:
                print(f"Processed {p31_count} P31 (instance_of) relationships")
            
            # Process P279 (subclass_of) relationships
            result = session.run("""
                MATCH (s:node)-[r:P279]->(e:node)
                CREATE (e)-[r2:subclass_of]->(s)
                SET r2 = r
                DELETE r
                RETURN count(r2) as processed
            """)
            p279_count = result.single()["processed"]
            if p279_count > 0:
                print(f"Processed {p279_count} P279 (subclass_of) relationships")
            
            # Process P361 (part_of) relationships
            result = session.run("""
                MATCH (s:node)-[r:P361]->(e:node)
                CREATE (e)-[r2:part_of]->(s)
                SET r2 = r
                DELETE r
                RETURN count(r2) as processed
            """)
            p361_count = result.single()["processed"]
            if p361_count > 0:
                print(f"Processed {p361_count} P361 (part_of) relationships")
            
            # Process P527 (has_part) relationships  
            result = session.run("""
                MATCH (s:node)-[r:P527]->(e:node)
                CREATE (e)-[r2:has_part]->(s)
                SET r2 = r
                DELETE r
                RETURN count(r2) as processed
            """)
            p527_count = result.single()["processed"]
            if p527_count > 0:
                print(f"Processed {p527_count} P527 (has_part) relationships")
            
            # Check if there are other remaining relationship types
            remaining_relations = session.run("""
                MATCH (s:node)-[r]->(e:node)
                WHERE NOT (type(r) IN ['instance_of', 'subclass_of', 'part_of', 'has_part'])
                RETURN DISTINCT type(r) as relation_type
                LIMIT 5
            """).values()
            
            if remaining_relations:
                print(f"Discovered remaining relationship types: {[r[0] for r in remaining_relations]}")
                # Process remaining relationships uniformly as subclass_of (keep original behavior)
                result = session.run("""
                    MATCH (s:node)-[r]->(e:node)
                    WHERE NOT (type(r) IN ['instance_of', 'subclass_of', 'part_of', 'has_part'])
                    CREATE (e)-[r2:subclass_of]->(s)
                    SET r2 = r
                    DELETE r
                    RETURN count(r2) as processed
                """)
                remaining_count = result.single()["processed"]
                if remaining_count > 0:
                    print(f"Processed {remaining_count} remaining relationships, uniformly named as subclass_of")
            
            # Display final relationship type distribution
            final_relations = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relation_type, count(r) as count
                ORDER BY count DESC
            """).data()
            
            print("Final relationship type distribution:")
            for rel in final_relations:
                print(f"  - {rel['relation_type']}: {rel['count']}")
    
    def remove_isolated_nodes(self):
        """Remove isolated nodes"""
        with self.driver.session() as session:
            session.run("MATCH (n) WHERE NOT EXISTS {(n)--()} DETACH DELETE n")
    
    def fetch_all_nodes(self) -> List[Dict[str, Any]]:
        """Fetch all nodes"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE NOT '_masked' IN labels(n) AND NOT '_GraphConfig' IN labels(n)
                RETURN toString(ID(n)) AS id, properties(n) AS properties
                """
            )
            return [record.data() for record in result]
    
    def fetch_nodes_with_names(self) -> List[Dict[str, Any]]:
        """Fetch nodes with names"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n) WHERE NOT '_masked' IN labels(n) AND NOT '_GraphConfig' IN labels(n)
                RETURN toString(ID(n)) AS id, n.name AS name
                """
            )
            return [{"nodeId": int(r["id"]), "node_name": r["name"] or ""} for r in result]
    
    def update_node_property(self, node_id: int, property_name: str, value: Any):
        """Update node property"""
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (n) WHERE ID(n) = $nid
                SET n.{property_name} = $value
                """,
                nid=node_id,
                value=value
            )
    
    def batch_update_node_properties(self, df: pd.DataFrame, 
                                   id_col: str, property_name: str, value_col: str):
        """Batch update node properties"""
        with self.driver.session() as session:
            for _, row in df.iterrows():
                session.run(
                    f"""
                    MATCH (n) WHERE ID(n) = $nid
                    SET n.{property_name} = $value
                    """,
                    nid=int(row[id_col]),
                    value=float(row[value_col])
                )
    
    def delete_nodes_not_in_list(self, keep_node_ids: List[int]):
        """Delete nodes not in the specified list"""
        with self.driver.session() as session:
            session.run(
                "MATCH (n) WHERE NOT ID(n) IN $ids DETACH DELETE n",
                ids=keep_node_ids
            )
    
    def get_node_count(self) -> int:
        """Get total node count"""
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            return result.single()["count"]
    
    def get_relationship_count(self) -> int:
        """Get total relationship count"""
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            return result.single()["count"]
    
    def get_relationship_type_distribution(self) -> Dict[str, int]:
        """
        Get relationship type distribution statistics
        
        Returns:
            Dict[str, int]: Mapping from relationship type name to count
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as relation_type, count(r) as count
                ORDER BY count DESC
            """)
            return {record["relation_type"]: record["count"] for record in result}
    
    def get_node_names(self, limit: int = 10) -> List[str]:
        """
        Get node name list
        
        Args:
            limit: Limit on number of nodes to return
            
        Returns:
            List of node names
        """
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (n)
                WHERE n.name IS NOT NULL AND n.name <> ''
                RETURN DISTINCT n.name as name
                LIMIT {limit}
            """)
            return [record["name"] for record in result]
    
    def get_nodes_without_wikipedia_summary(self, limit: int = 10) -> List[str]:
        """
        Get node names without Wikipedia summary
        
        Args:
            limit: Limit on number of nodes to return
            
        Returns:
            List of node names
        """
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (n)
                WHERE n.name IS NOT NULL AND n.name <> '' 
                AND (n.wikipedia_summary IS NULL OR n.wikipedia_summary = '')
                RETURN DISTINCT n.name as name
                LIMIT {limit}
            """)
            return [record["name"] for record in result]
    
    def add_wikipedia_summary_to_node(self, node_name: str, wikipedia_summary: str):
        """
        Add Wikipedia summary to specified node
        
        Args:
            node_name: Node name
            wikipedia_summary: Wikipedia summary
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n {name: $node_name})
                SET n.wikipedia_summary = $wikipedia_summary
                RETURN count(n) as updated_count
            """, node_name=node_name, wikipedia_summary=wikipedia_summary)
            return result.single()["updated_count"]
    
    def update_node_toxicity(self, 
                            node_id: int, 
                            average_toxicity: float, 
                            max_toxicity: float, 
                            harmful_ratio: Optional[float] = None,
                            total_prompts: Optional[int] = None):
        """
        Update node toxicity scores
        
        Args:
            node_id: Node ID
            average_toxicity: Average toxicity score
            max_toxicity: Maximum toxicity score
            harmful_ratio: Harmful prompt ratio
            total_prompts: Total prompt count
        """
        with self.driver.session() as session:
            # Build SET clause
            set_properties = [
                "n.average_toxicity = $avg_toxicity",
                "n.max_toxicity = $max_toxicity",
                "n.toxicity_analyzed = true"
            ]
            
            params = {
                "node_id": node_id,
                "avg_toxicity": average_toxicity,
                "max_toxicity": max_toxicity
            }
            
            if harmful_ratio is not None:
                set_properties.append("n.harmful_ratio = $harmful_ratio")
                params["harmful_ratio"] = harmful_ratio
            
            if total_prompts is not None:
                set_properties.append("n.total_prompts = $total_prompts")
                params["total_prompts"] = total_prompts
            
            set_clause = ", ".join(set_properties)
            query = f"""
                MATCH (n) WHERE ID(n) = $node_id
                SET {set_clause}
            """
            
            session.run(query, **params)
    
    def get_node_by_id(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Get node information by ID"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n) WHERE ID(n) = $node_id
                RETURN n, ID(n) as node_id
            """, node_id=node_id)
            
            record = result.single()
            if record:
                node = record["n"]
                return {
                    "id": record["node_id"],
                    "labels": list(node.labels),
                    "properties": dict(node)
                }
            return None
    
    def fix_relationship_types(self):
        """
        Fix relationship types: Convert Wikidata property relationships to semantic relationship names
        Convert from original wdt:P31, wdt:P279 etc. to instance_of, subclass_of etc.
        
        This method solves the "all relationships are called subclass_of" problem by checking original relationship properties to determine correct semantic types
        """
        with self.driver.session() as session:
            # Check current relationship distribution
            current_rels = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(*) as count")
            print("Relationship type distribution before fix:")
            for record in current_rels:
                print(f"  {record['rel_type']}: {record['count']}")
            
            # If the graph already has original wdt property relationships, we need to convert them
            # But typically n10s import creates semantic relationship names, so we may need a different strategy
            
            # Strategy 1: If original wdt relationships exist, convert them
            wdt_relations = [
                ("P31", "instance_of"),
                ("P279", "subclass_of"), 
                ("P361", "part_of"),
                ("P527", "has_part")
            ]
            
            for wdt_prop, semantic_name in wdt_relations:
                # Check if this wdt relationship exists
                check_query = f"MATCH ()-[r:`wdt:{wdt_prop}`]->() RETURN count(r) as count"
                result = session.run(check_query)
                count = result.single()['count']
                
                if count > 0:
                    print(f"Converting wdt:{wdt_prop} -> {semantic_name} ({count} relationships)")
                    
                    # Create new semantic relationship and delete original
                    convert_query = f"""
                    MATCH (a)-[old:`wdt:{wdt_prop}`]->(b)
                    CREATE (a)-[new:{semantic_name}]->(b)
                    DELETE old
                    """
                    session.run(convert_query)
            
            # Strategy 2: If all relationships are subclass_of, rebuild correct relationships based on node properties
            # Check if all relationships are subclass_of
            subclass_count = session.run("MATCH ()-[r:subclass_of]->() RETURN count(r) as count").single()['count']
            total_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            
            if subclass_count > 0 and subclass_count == total_count:
                print(f"Found all {total_count} relationships are subclass_of, need to rebuild correct relationship types based on original data")
                
                # In this case, we need to go back to the original query to get correct relationship types
                # But this requires re-querying Wikidata, which is more complex
                print("Warning: Cannot rebuild relationship types from current data, recommend re-importing with categorized queries")
            
            # Check relationship distribution after fix
            final_rels = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(*) as count ORDER BY count DESC")
            print("Relationship type distribution after fix:")
            for record in final_rels:
                print(f"  {record['rel_type']}: {record['count']}")
    
    def clean_duplicate_relationships(self):
        """
        Clean duplicate relationships: If a pair of nodes has multiple relationship types, keep only one
        
        Priority: instance_of > subclass_of > part_of > has_part
        """
        with self.driver.session() as session:
            # Check duplicate relationships
            duplicate_check = session.run("""
                MATCH (a)-[r]->(b)
                WITH a, b, collect(type(r)) as rel_types, count(r) as rel_count
                WHERE rel_count > 1
                RETURN count(*) as duplicate_pairs
            """).single()['duplicate_pairs']
            
            if duplicate_pairs > 0:
                print(f"Found {duplicate_pairs} node pairs with duplicate relationships, starting cleanup...")
                
                # Delete duplicate relationships, maintain priority
                session.run("""
                    MATCH (a)-[r]->(b)
                    WITH a, b, collect(r) as rels
                    WHERE size(rels) > 1
                    
                    // Sort by priority: instance_of > subclass_of > part_of > has_part
                    WITH a, b, rels,
                         [rel IN rels WHERE type(rel) = 'instance_of'] as instance_rels,
                         [rel IN rels WHERE type(rel) = 'subclass_of'] as subclass_rels,
                         [rel IN rels WHERE type(rel) = 'part_of'] as part_rels,
                         [rel IN rels WHERE type(rel) = 'has_part'] as has_part_rels
                    
                    WITH a, b, rels,
                         CASE 
                           WHEN size(instance_rels) > 0 THEN instance_rels[0]
                           WHEN size(subclass_rels) > 0 THEN subclass_rels[0] 
                           WHEN size(part_rels) > 0 THEN part_rels[0]
                           ELSE has_part_rels[0]
                         END as keep_rel
                    
                    WITH rels, keep_rel, [rel IN rels WHERE rel <> keep_rel] as delete_rels
                    
                    FOREACH (rel IN delete_rels | DELETE rel)
                """)
                
                print("Duplicate relationship cleanup completed")
            else:
                print("No duplicate relationships found")
    
    def get_relationship_type_distribution(self) -> Dict[str, int]:
        """
        Get relationship type distribution statistics
        
        Returns:
            Dict[str, int]: Mapping from relationship type to count
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
            """)
            
            return {record["rel_type"]: record["count"] for record in result}
