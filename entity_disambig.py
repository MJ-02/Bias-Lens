#!/usr/bin/env python3

import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm.auto import tqdm
import time
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import math
import os
import argparse

def get_top_5_entities(row):
    """
    Keep only the top 5 entities for each document based on score.
    
    Args:
        row: DataFrame row containing ID and entities list
        
    Returns:
        Dictionary with ID and filtered entities list
    """
    # Sort entities by score in descending order
    sorted_entities = sorted(row['entities'], key=lambda x: x['score'], reverse=True)
    # Keep top 5
    top_5 = sorted_entities[:5]
    
    return {
        'ID': row['ID'],
        'entities': top_5
    }

def query_wikidata(entity):
    """
    Query Wikidata for entity disambiguation.
    
    Args:
        entity: Dictionary containing entity information
        
    Returns:
        Updated entity dictionary with Wikidata information
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # Set User-Agent to avoid 403 errors
    sparql.addCustomHttpHeader('User-Agent', 'BiasLensCooking/1.0 (https://github.com/MJ-02/BiasLensCooking; majaliabdullah@live.com)')
    
    # Construct query based on entity label
    if entity['label'] == 'person':
        type_filter = 'wdt:P31 wd:Q5'  # instance of human
    elif entity['label'] == 'location':
        type_filter = 'wdt:P31/wdt:P279* wd:Q17334923'  # instance of location
    else:
        type_filter = 'wdt:P31 ?type'
    
    # Improved query with better entity matching
    query = f"""
    SELECT DISTINCT ?item ?itemLabel ?description WHERE {{
      ?item rdfs:label "{entity['text']}"@en.
      ?item {type_filter}.
      ?item schema:description ?description.
      FILTER (LANG(?description) = "en")
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 1
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        if results["results"]["bindings"]:
            result = results["results"]["bindings"][0]
            entity.update({
                'wikidata_id': result["item"]["value"].split("/")[-1],
                'wikidata_label': result["itemLabel"]["value"],
                'wikidata_description': result["description"]["value"]
            })
    except Exception as e:
        print(f"Error querying Wikidata for {entity['text']}: {str(e)}")
    
    time.sleep(0.1)  # Rate limiting per request
    return entity

def process_chunk(args):
    """
    Process a chunk of documents in parallel.
    
    Args:
        args: Tuple of (chunk, chunk_index)
        
    Returns:
        List of processed documents with disambiguated entities
    """
    chunk, chunk_index = args
    results = []
    
    # Create progress bar for entities within this chunk
    with tqdm(total=len(chunk), 
              desc=f"Chunk {chunk_index}", 
              position=chunk_index+1,
              leave=False) as pbar:
        
        for doc in chunk:
            disambiguated = []
            for entity in doc['entities']:
                updated_entity = query_wikidata(entity.copy())
                disambiguated.append(updated_entity)
            
            results.append({
                'ID': doc['ID'],
                'entities': disambiguated
            })
            pbar.update(1)
    
    return results

def parallelize_disambiguation(df, num_processes=None):
    """
    Parallelize entity disambiguation across multiple processes.
    
    Args:
        df: DataFrame containing documents and entities
        num_processes: Number of processes to use (defaults to CPU count - 1)
        
    Returns:
        DataFrame with disambiguated entities
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    # Convert DataFrame to list of dictionaries
    docs = df.to_dict('records')
    
    # Split documents into chunks
    chunk_size = math.ceil(len(docs) / num_processes)
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
    
    # Create list of (chunk, index) tuples
    chunk_args = [(chunk, i) for i, chunk in enumerate(chunks)]
    
    print(f"Processing {len(docs)} documents using {num_processes} processes")
    
    try:
        # Process chunks in parallel
        with mp.Pool(num_processes) as pool:
            # Create main progress bar for chunks
            with tqdm(total=len(chunks), 
                     desc="Overall Progress", 
                     position=0) as main_pbar:
                
                # Process chunks
                results = []
                for result in pool.imap(process_chunk, chunk_args):
                    results.append(result)
                    main_pbar.update(1)
        
        # Flatten results
        flattened_results = [doc for chunk_result in results for doc in chunk_result]
        
        return pd.DataFrame(flattened_results)
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        # Fallback to sequential processing if parallel fails
        print("Falling back to sequential processing...")
        results = []
        for chunk_arg in tqdm(chunk_args, desc="Processing chunks"):
            results.extend(process_chunk(chunk_arg))
        return pd.DataFrame(results)

def analyze_results(enriched_df, output_dir):
    """
    Analyze disambiguation results and save analysis.
    
    Args:
        enriched_df: DataFrame with disambiguated entities
        output_dir: Directory to save analysis results
    """
    total_entities = sum(len(row['entities']) for _, row in enriched_df.iterrows())
    linked_entities = sum(1 for _, row in enriched_df.iterrows() 
                         for entity in row['entities'] 
                         if 'wikidata_id' in entity)

    print(f"Total entities: {total_entities}")
    print(f"Successfully linked entities: {linked_entities}")
    print(f"Linking success rate: {linked_entities/total_entities*100:.2f}%")

    # Analyze entity types
    entity_labels = defaultdict(int)
    for _, row in enriched_df.iterrows():
        for entity in row['entities']:
            entity_labels[entity['label']] += 1

    print("\nEntity label distribution:")
    for label, count in sorted(entity_labels.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {count}")

    # Save analysis results
    analysis_file = os.path.join(output_dir, 'entity_analysis.json')
    analysis_results = {
        'total_entities': total_entities,
        'linked_entities': linked_entities,
        'linking_success_rate': linked_entities/total_entities*100,
        'entity_label_distribution': dict(entity_labels)
    }
    try:
        pd.DataFrame([analysis_results]).to_json(analysis_file)
        print(f"\nSaved analysis results to {analysis_file}")
    except Exception as e:
        print(f"Error saving analysis results: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Entity disambiguation using Wikidata')
    parser.add_argument('input_file', help='Path to input JSON file containing entities')
    parser.add_argument('--output-dir', default=os.getcwd(),
                      help='Directory to save output files (default: current directory)')
    parser.add_argument('--processes', type=int, default=None,
                      help='Number of processes to use (default: CPU count - 1)')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load entities
    print(f"Loading entities from {args.input_file}")
    df = pd.read_json(args.input_file)

    # Filter top 5 entities
    print("Filtering top 5 entities per document...")
    filtered_df = pd.DataFrame([get_top_5_entities(row) for _, row in df.iterrows()])

    # Process documents in parallel
    print("Disambiguating entities...")
    enriched_df = parallelize_disambiguation(filtered_df, args.processes)
    
    # Save processed entities
    output_file = os.path.join(args.output_dir, 'enriched_entities.json')
    try:
        enriched_df.to_json(output_file)
        print(f"\nSaved enriched entities to {output_file}")
    except Exception as e:
        print(f"Error saving enriched entities: {str(e)}")
    
    # Analyze and save results
    analyze_results(enriched_df, args.output_dir)

if __name__ == '__main__':
    main()
