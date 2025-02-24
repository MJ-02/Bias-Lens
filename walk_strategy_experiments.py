import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec, FastText
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import (
    UniformSampler,
    PageRankSampler,
    WideSampler,
    ObjFreqSampler,
    PredFreqSampler,
)
from pyrdf2vec.walkers import HALKWalker, NGramWalker, CommunityWalker, RandomWalker, WalkletWalker, Walker, WLWalker

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os
from datetime import datetime

# Configuration
# KG_PATH = "results/kg_balanced.ttl"  # Path to the input data file
# DATA_PATH = "results/kg_balanced_labels.csv"  # Path to the knowledge graph file
KG_PATH = "results/knowledge_graph.ttl"  # Path to the input data file
DATA_PATH = "results/URI_label_pairs.csv"  # Path to the knowledge graph file
OUTPUT_DIR = "results/experiments"  # Directory to save results
TRAIN_SPLIT = 0.2  # Training data split ratio
RANDOM_STATE = 22  # Random state for reproducibility
PAGERANK_ALPHA = 0.85  # Alpha parameter for PageRank sampler

def load_data(path, train_split=0.2):
    """Load and prepare the dataset"""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(lambda x: x.strip('<>'))  # Clean URI format
    print(df.head())
    train_size = int(len(df) * train_split)
    test_data = df.iloc[:train_size]
    train_data = df.iloc[train_size:]

    train_entities = [entity for entity in train_data.index]
    train_labels = list(train_data['1'])  # Column name is '1' in the CSV
    test_entities = [entity for entity in test_data.index]
    test_labels = list(test_data['1'])  # Column name is '1' in the CSV

    entities = train_entities + test_entities
    labels = train_labels + test_labels

    return train_entities, train_labels, test_entities, test_labels, entities, labels

def get_scores(X_train, X_test, y_train, y_test):
    """Evaluate multiple classifiers and return their scores"""
    arr = []
    clfs = ["XGBClassifier", "LogisticRegression", "RandomForestClassifier", "SVC", "KNeighborsClassifier"]
    for clf in clfs:
        model = eval(clf + "()")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        arr.append((clf, accuracy_score(y_test, preds), f1_score(y_test, preds, average="macro")))
    return pd.DataFrame(arr, columns=["Model", "Accuracy", "F1 Macro"])

def evaluate_sampler(walker_class, sampler_class, kg, entities, train_entities, train_labels, test_labels, random_state=22, **sampler_params):
    """Evaluate a specific walker-sampler combination"""
    rdf2vec = RDF2VecTransformer(
        Word2Vec(workers=8, epochs=20),
        walkers=[
            walker_class(
                2, None,
                n_jobs=8,
                sampler=sampler_class(**sampler_params),
                random_state=random_state,
                md5_bytes=None
            )
        ],
        verbose=1
    )
    embeddings, literals = rdf2vec.fit_transform(kg, np.array(entities))
    train_embeddings = np.array(embeddings[:len(train_entities)])
    test_embeddings = np.array(embeddings[len(train_entities):])
    return get_scores(train_embeddings, test_embeddings, train_labels, test_labels)

def save_result(walker_name, sampler_name, scores_df, output_dir):
    """Save results for a single walker-sampler combination"""
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert scores to a format suitable for saving
    result_rows = []
    for _, row in scores_df.iterrows():
        result = {
            'Walker': walker_name,
            'Sampler': sampler_name,
            'Model': row['Model'],
            'Accuracy': row['Accuracy'],
            'F1_Macro': row['F1 Macro'],
            'Timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        result_rows.append(result)
    
    # Save to CSV
    result_df = pd.DataFrame(result_rows)
    output_file = os.path.join(output_dir, f'{walker_name}_{sampler_name}_results.csv')
    
    # If file exists, append to it, otherwise create new
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return result_df

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Set pandas display format
    pd.options.display.float_format = '{:,.4f}'.format

    # Load data
    print("Loading data...")
    train_entities, train_labels, test_entities, test_labels, entities, labels = load_data(DATA_PATH, TRAIN_SPLIT)

    # Initialize knowledge graph
    print("Loading knowledge graph...")
    kg = KG(
        location=KG_PATH,
        skip_predicates={"http://biaslens.io/bias"},
        skip_verify=True
    )

    # Define walker classes to evaluate
    walkers = {
        'HALK': HALKWalker,
        'NGram': NGramWalker,
        'Random': RandomWalker,
        'Walklet': WalkletWalker
    }

    # Define samplers to evaluate
    samplers = {
        'Uniform': (UniformSampler, {}),
        'PageRank': (PageRankSampler, {'alpha': PAGERANK_ALPHA}),
        'Wide': (WideSampler, {}),
        'ObjFreq': (ObjFreqSampler, {}),
        'PredFreq': (PredFreqSampler, {})
    }

    # Evaluate each combination and save results immediately
    for walker_name, walker_class in walkers.items():
        for sampler_name, (sampler_class, params) in samplers.items():
            print(f"\nEvaluating {walker_name} with {sampler_name} sampler")
            try:
                scores = evaluate_sampler(
                    walker_class, sampler_class, kg, entities, 
                    train_entities, train_labels, test_labels,
                    random_state=RANDOM_STATE, **params
                )
                print(f"Results for {walker_name} with {sampler_name}:")
                print(scores)
                
                # Save results immediately
                save_result(walker_name, sampler_name, scores, OUTPUT_DIR)
                
            except Exception as e:
                print(f"Error evaluating {walker_name} with {sampler_name}: {str(e)}")
                continue

    print("\nExperiments complete. Results saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
