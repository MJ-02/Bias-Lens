import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re
import json
from rdflib import Graph, Literal, Namespace, RDF, XSD, RDFS

def clean_author_names(names):
    """
    Cleans the author names provided by both the news API and in the dataset. Will split authors into a list. Will also handle missing names

    :param names: The author names to be cleaned
    :type names: str

    :return: list of cleaned author names
    :rtype: list
    """
    if names == None:
        return ["anonymous"]
    pattern = r"[#.<>\]\[\\]"

    cleaned_names = []
    for name in names:
        if name == "":
            name = "anonymous"
        name = name.lower()
        name = name.strip()
        name = re.sub(pattern, "", name)
        name = re.sub(r"\s", "", name)
        name = re.sub(r"\"", "", name)
        cleaned_names.append(name)
    return cleaned_names

def clean_outlet_names(name: str) -> str:
    """
    Cleans the outlet names by removing special characters and converting to lower case.
    
    :param name: The outlet name to be cleaned
    :type name: str

    :return: cleaned outlet name
    :rtype: str
    """
    pattern = r"[()\[\]\!<>-]|CDATA|\.com|(online news)"
    name = name.lower()
    name = re.sub(pattern, "", name)
    name = re.sub(r"\s", "", name)
    name = re.sub(r"\"", "", name)
    return name

def process_entities(entity_data: pd.DataFrame) -> dict:
    """
    Process entities from enriched_entities.json and create a dictionary of unique entities and a dictionary of pairs of
    article ids and the entities mentioned in the article holding that ID.
    
    :param entity_data: DataFrame containing entity data for articles
    :type entity_data: pd.DataFrame
    
    :return: Dictionary of unique entities with their metadata and dictionary of pairs of article IDs and entities
    """
    unique_entities = {}
    entity_id_pairs = {}
    for _, row in entity_data.iterrows():
        article_id = row["ID"]
        entities_list = row["entities"]

        entity_id_pairs[article_id] = []
        for entity in entities_list:
            # print(f"entity {entity}")
            # if it has no wikidata_id, skip it
            if "wikidata_id" not in entity:
                continue
            wikidata_id = entity['wikidata_id']
            # print(f"wikidata_id {wikidata_id}")
            if wikidata_id not in unique_entities:
                unique_entities[wikidata_id] = {
                    'name': entity["wikidata_label"],
                    'type': entity["label"],
                    'description': entity["wikidata_description"],
                    'start': entity["start"],
                    'end': entity["end"]
                }
            entity_id_pairs[article_id].append(wikidata_id)

    return unique_entities, entity_id_pairs

# print(process_entities(df))
def add_entity_nodes(g, ns, entities):
    """
    Add entity nodes to the graph with their metadata.
    
    :param g: RDF graph
    :param ns: Namespace
    :param entities: Dictionary of entities with metadata
    """
    for wikidata_id, entity_data in entities.items():
        entity = ns[f"entity_{wikidata_id}"]
        
        # Add entity type
        g.add((entity, RDF.type, ns.MentionedEntity))
        g.add((entity, ns.entityType, Literal(entity_data['type'], datatype=XSD.string)))
        
        # Add metadata
        g.add((entity, ns.wikidataId, Literal(wikidata_id, datatype=XSD.string)))
        g.add((entity, RDFS.label, Literal(entity_data['name'], datatype=XSD.string)))
        g.add((entity, ns.description, Literal(entity_data['description'], datatype=XSD.string)))
        g.add((entity, ns.start, Literal(entity_data['start'], datatype=XSD.integer)))
        g.add((entity, ns.end, Literal(entity_data['end'], datatype=XSD.integer)))

def add_topic_nodes(g, ns, topics):
    """
    Add topic nodes to the graph.
    
    :param g: RDF graph
    :param ns: Namespace
    :param topics: Set of unique topics
    """
    for topic in topics:
        topic_node = ns[f"topic_{topic.lower().replace(' ', '_')}"]
        g.add((topic_node, RDF.type, ns.Topic))
        g.add((topic_node, RDFS.label, Literal(topic, datatype=XSD.string)))

def generate_graph(df: pd.DataFrame, entities_data: pd.DataFrame, output_path: str):
    """
    Generate knowledge graph from articles data and entities.
    
    :param df: DataFrame containing article data
    :param entities_data: DataFrame containing entity data by article ID
    :param output_path: Path to save the output files
    """
    g = Graph()
    ns = Namespace("http://biaslens.io/")
    # ns_topic = Namespace("http://biaslens.io/topic")
    # ns_author = Namespace("http://biaslens.io/author")
    # ns_entity = Namespace("http://biaslens.io/entity")
    # ns_article= Namespace("http://biaslens.io/article")
    g.bind("bl", ns)
    
    # Need URI label pairs to split the graph into test and training data
    uri_label_pairs = {}

    # Process entities
    unique_entities, entity_id_pairs = process_entities(entities_data)
    
    # Get unique topics
    unique_topics = set(df['topic'].unique())
    
    # First collect all entities that are actually mentioned in our articles
    mentioned_entities = {}
    for _, row in df.iterrows():
        article_id = row['ID']
        if article_id in entity_id_pairs:
            for wikidata_id in entity_id_pairs[article_id]:
                if wikidata_id in unique_entities:
                    mentioned_entities[wikidata_id] = unique_entities[wikidata_id]
    
    # Add only the entities that are actually mentioned
    print(f"Adding {len(mentioned_entities)} entity nodes (out of {len(unique_entities)} total entities)...")
    add_entity_nodes(g, ns, mentioned_entities)
    
    # Add topic nodes
    print("Adding topic nodes...")
    add_topic_nodes(g, ns, unique_topics)
    
    # Process articles and create relationships
    print("Processing articles and creating relationships...")
    for _, row in tqdm(df.iterrows()):
        article_id = row['ID']
        article = ns[f"article_{article_id}"]
        uri_label_pairs[article] = str(row['bias'])
        
        # Article node and basic properties
        g.add((article, RDF.type, ns.Article))
        g.add((article, ns.id, Literal(article_id, datatype=XSD.string)))
        g.add((article, ns.title, Literal(row['title'], datatype=XSD.string)))
        # g.add((article, ns.date, Literal(row['date'], datatype=XSD.dateTime)))
        g.add((article, ns.bias, Literal(row['bias'], datatype=XSD.integer)))
        
        # Topic relationship
        topic = ns[f"topic_{row['topic'].lower().replace(' ', '_')}"]
        g.add((article, ns.hasTopic, topic))
        
        # Outlet relationship
        outlet = ns[clean_outlet_names(row['source'])]
        g.add((outlet, RDF.type, ns.Outlet))
        g.add((outlet, RDFS.label, Literal(row['source'], datatype=XSD.string)))
        g.add((outlet, ns.url, Literal(row['source_url'], datatype=XSD.string)))
        g.add((article, ns.publishedBy, outlet))
        
        # Author relationships
        authors = clean_author_names(row['authors'].split(','))
        for author_name in authors:
            author = ns[f"author_{author_name}"]
            g.add((author, RDF.type, ns.Author))
            g.add((author, RDFS.label, Literal(author_name, datatype=XSD.string)))
            g.add((article, ns.writtenBy, author))
        
        # Entity relationships
        if article_id in entity_id_pairs:
            for wikidata_id in entity_id_pairs[article_id]:
                entity_node = ns[f"entity_{wikidata_id}"]
                g.add((article, ns.mentions, entity_node))
                # print(f"Added mention relationship: {article_id} -> {wikidata_id}")
    
    # Save the graph in different formats
    print("Saving graph...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as Turtle
    turtle_output = g.serialize(format="turtle")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(turtle_output)
    print(f"Graph saved as Turtle at {output_path}")
    # Save URI label pairs
    labels_path = output_path.replace('.ttl', '_labels.csv')
    uri_label_pairs = pd.DataFrame.from_dict(uri_label_pairs, orient='index')
    uri_label_pairs.to_csv(labels_path, header=False)
    print(f"URI label pairs saved at {labels_path}")
    print("Graph generation complete!")

def create_balanced_dataset(df, samples_per_class):
    """
    Create a balanced dataset with specified number of samples per class using a while loop
    
    :param df: DataFrame containing article data
    :param samples_per_class: Number of samples to include for each class
    :return: Balanced DataFrame
    """
    if samples_per_class == None:
        return df
    balanced_df = pd.DataFrame(columns=df.columns)
    for label in [0, 1, 2]:
        label_data = df[df['bias'] == label].copy()
        sampled_indices = []
        while len(sampled_indices) < samples_per_class:
            remaining = samples_per_class - len(sampled_indices)
            new_indices = label_data.index[~label_data.index.isin(sampled_indices)].to_numpy()
            if len(new_indices) == 0:
                break
            n_sample = min(remaining, len(new_indices))
            sampled_indices.extend(np.random.choice(new_indices, size=n_sample, replace=False))
        
        balanced_df = pd.concat([balanced_df, df.loc[sampled_indices]], ignore_index=True)
    
    return balanced_df

def main():
    """
    Main function to build the knowledge graph
    """
    # Configuration
    SAMPLES_PER_CLASS = None  # Set to None for full dataset
    OUTPUT_PATH = f"results/{'kg_balanced.ttl' if SAMPLES_PER_CLASS else 'knowledge_graph.ttl'}"
    # Load article data
    print("Loading article data...")
    df = pd.read_parquet('articles.parquet')
    
    # Create balanced dataset
    print(f"\nCreating balanced dataset with {SAMPLES_PER_CLASS} samples per class...")
    balanced_df = create_balanced_dataset(df, SAMPLES_PER_CLASS)
    print(f"Original dataset size: {len(df)}")
    print(f"Balanced dataset size: {len(balanced_df)}")
    print("\nClass distribution in balanced dataset:")
    print(balanced_df['bias'].value_counts().sort_index())
    
    # Load entity data
    print("\nLoading entity data...")
    df_entities = pd.read_json("results/enriched_entities.json")
    
    # Filter entity data to match balanced dataset
    balanced_ids = set(balanced_df['ID'])
    balanced_entities = df_entities[df_entities['ID'].isin(balanced_ids)].copy()
    print(f"Filtered entities from {len(df_entities)} to {len(balanced_entities)} entries")
    
    # Generate the knowledge graph
    print("\nGenerating knowledge graph...")
    generate_graph(balanced_df, balanced_entities, OUTPUT_PATH)

if __name__ == "__main__":
    main()
