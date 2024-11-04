
import argparse
import csv
import json
import random
import uuid

import chromadb
import numpy as np
import pandas as pd
from numpy.linalg import norm
from transformers import AutoModel


class EmbeddingsModel:
    """
    A class to handle embeddings model initialization, encoding of texts, and normalization of vectors.
    """

    def __init__(self, model_name='jinaai/jina-embeddings-v2-base-es'):
        """
        Initializes the model using the provided model name.
        
        Args:
            model_name (str): The name of the model to load. Defaults to Jina embeddings for Spanish.
        """
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def encode(self, texts):
        """
        Encodes a list of texts into embeddings using the model.
        
        Args:
            texts (list): List of texts to encode.
        
        Returns:
            np.ndarray: Embeddings generated for the input texts.
        """
        embeddings = self.model.encode(texts)
        return embeddings

    def normalize(self, embeddings):
        """
        Normalizes the embeddings using L2 normalization.
        
        Args:
            embeddings (np.ndarray): Embeddings to be normalized.
        
        Returns:
            np.ndarray: Normalized embeddings.
        """
        l2_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / l2_norms
        return normalized_embeddings


class VectorDatabase:
    """
    A class to manage vector storage and queries using Chroma database.
    """

    def __init__(self, collection_name: str):
        """
        Initializes a Chroma vector database and creates a collection for vectors.
        
        Args:
            collection_name (str): The name of the collection where vectors will be stored.
        """
        self.client = chromadb.Client()  # Initialize the Chroma client
        self.collection = self.client.create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created.")

    def add_vectors_from_tsv(self, tsv_file_path: str):
        """
        Adds vectors from a TSV file into the Chroma collection.
        
        Args:
            tsv_file_path (str): Path to the TSV file containing vectors.
        """
        with open(tsv_file_path, mode='r', newline='', encoding='utf-8') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            for i, row in enumerate(reader):
                # Convert space-separated values in the TSV to a list of floats
                vector = [float(num) for num in row]
                # Add vector to the Chroma collection
                self.collection.add(
                    embeddings=[vector],
                    ids=[str(i)]
                )

    def query_vectors(self, query_vector, top_k=5):
        """
        Queries the Chroma collection for the closest vectors to the provided query vector.
        
        Args:
            query_vector (list): The vector to query.
            top_k (int): Number of closest vectors to return. Default is 5.
        
        Returns:
            dict: A dictionary of the closest vectors' metadata and distances.
        """
        results = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=top_k
        )
        return results


class CoemDB:
    """
    A class that integrates vector databases for stories and poems and handles querying of vectors.
    """

    def __init__(self, path_to_stories, path_to_poems, metadata_stories, metadata_poems, model_name='jinaai/jina-embeddings-v2-base-es',fragmented=True):
        """
        Initializes the CoemDB with paths to TSV files and metadata, and sets up vector databases.
        
        Args:
            path_to_stories (str): Path to the TSV file containing stories vectors.
            path_to_poems (str): Path to the TSV file containing poems vectors.
            metadata_stories (str): Path to the TSV file containing stories metadata.
            metadata_poems (str): Path to the file containing poems metadata.
            model_name (str): Name of the embeddings model. Default is Jina embeddings for Spanish.
        """
        self.uuid = uuid.uuid4()
        self.fragmented=fragmented
        self.model = EmbeddingsModel(model_name)
        self.path_to_stories = path_to_stories
        self.path_to_poems = path_to_poems
        self.vector_db_stories = VectorDatabase(collection_name=f"stories_{self.uuid}")
        self.vector_db_stories.add_vectors_from_tsv(path_to_stories)
        self.stories_metadata = pd.read_csv(metadata_stories, delimiter='\t')
        if fragmented:
          metadata_poems = metadata_poems.split(".")[0]
          path_to_poems = path_to_poems.split(".")[0]
          self.vector_db_poems1 = VectorDatabase(collection_name=f"poems_1_{self.uuid}")
          self.vector_db_poems2 = VectorDatabase(collection_name=f"poems_2_{self.uuid}")
          self.vector_db_poems3 = VectorDatabase(collection_name=f"poems_3_{self.uuid}")
          self.vector_db_poems4 = VectorDatabase(collection_name=f"poems_4_{self.uuid}")
          self.vector_db_poems1.add_vectors_from_tsv(f"{path_to_poems}_1.tsv")
          self.vector_db_poems2.add_vectors_from_tsv(f"{path_to_poems}_2.tsv")
          self.vector_db_poems3.add_vectors_from_tsv(f"{path_to_poems}_3.tsv")
          self.vector_db_poems4.add_vectors_from_tsv(f"{path_to_poems}_4.tsv")
          self.poems_metadata1 = pd.read_csv(f"{metadata_poems}_1.tsv", delimiter='\t')
          self.poems_metadata2 = pd.read_csv(f"{metadata_poems}_2.tsv", delimiter='\t')
          self.poems_metadata3 = pd.read_csv(f"{metadata_poems}_3.tsv", delimiter='\t')
          self.poems_metadata4 = pd.read_csv(f"{metadata_poems}_4.tsv", delimiter='\t')
        else:
          self.vector_db_poems = VectorDatabase(collection_name="poems")
          self.vector_db_poems.add_vectors_from_tsv(metadata_poems)
          self.poems_metadata = pd.read_csv(metadata_poems, delimiter='\t')

    
    def query(self, texts, type_mode="stories", top_k=3):
        """
        Queries the database for stories or poems based on the input texts.
        
        Args:
            texts (list): List of texts to encode and query.
            type_mode (str): Type of search - either "stories", "poems", or "both". Default is "stories".
            top_k (int): Number of results to return. Default is 3.
        
        Returns:
            str: JSON string with the closest stories or poems based on the query.
        """
        embeddings = self.model.encode(texts)
        norm_embed = self.model.normalize(embeddings)

        if type_mode == "stories":
            results = self.vector_db_stories.query_vectors(query_vector=norm_embed, top_k=top_k)
            rows_data = [self.stories_metadata.iloc[i].to_dict() for i in results['ids']]
            rows_json = json.dumps({"stories": rows_data, "poems": None}, indent=4, ensure_ascii=False)
            return rows_json

        elif type_mode == "poems":
            if self.fragmented:
              results = self.vector_db_poems1.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data1 = [self.poems_metadata1.iloc[i].to_dict() for i in results['ids']]
              results = self.vector_db_poems2.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data2 = [self.poems_metadata2.iloc[i].to_dict() for i in results['ids']]
              results = self.vector_db_poems3.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data3 = [self.poems_metadata3.iloc[i].to_dict() for i in results['ids']]
              results = self.vector_db_poems4.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data4 = [self.poems_metadata4.iloc[i].to_dict() for i in results['ids']]
              rows_data_tot=[*rows_data1,*rows_data2,*rows_data3,*rows_data4]
              rows_data = random.sample(rows_data_tot,1)
              rows_data = rows_data[0]
            else:
              results = self.vector_db_poems.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data = [self.poems_metadata.iloc[i].to_dict() for i in results['ids']]




            rows_json = json.dumps({"stories": None, "poems": rows_data}, indent=4, ensure_ascii=False)
            return rows_json

        else:  # For "both" mode
            results_stories = self.vector_db_stories.query_vectors(query_vector=norm_embed, top_k=top_k)
            if self.fragmented:
              results = self.vector_db_poems1.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data1 = [self.poems_metadata1.iloc[i].to_dict() for i in results['ids']]
              results = self.vector_db_poems2.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data2 = [self.poems_metadata2.iloc[i].to_dict() for i in results['ids']]
              results = self.vector_db_poems3.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data3 = [self.poems_metadata3.iloc[i].to_dict() for i in results['ids']]
              results = self.vector_db_poems4.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data4 = [self.poems_metadata4.iloc[i].to_dict() for i in results['ids']]
              rows_data_tot=[rows_data1,rows_data2,rows_data3,rows_data4]
              # Get one sample of the databases at random
              # TODO: mix them all and get three random or the best score
              rows_data_poems = random.sample(rows_data_tot,1)
            else:
              results_poems = self.vector_db_poems.query_vectors(query_vector=norm_embed, top_k=top_k)
              rows_data_poems = [self.poems_metadata.iloc[i].to_dict() for i in results_poems['ids']]
            rows_data_stories = [self.stories_metadata.iloc[i].to_dict() for i in results_stories['ids']]
            rows_json = json.dumps({"stories": rows_data_stories, "poems": rows_data_poems}, indent=4, ensure_ascii=False)
            return rows_json



def main():
  """
  Main function to demonstrate the usage of the CoemDB class for querying vectors.
  It accepts command-line arguments for modelname, paths, type of search, and top_k.
  """
  # Set up argument parsing
  parser = argparse.ArgumentParser(description="Query the CoemDB for stories or poems based on text input.")
  
  parser.add_argument('--modelname', type=str, default='jinaai/jina-embeddings-v2-base-es', 
                      help="Name of the model to use for embeddings. Default is 'jinaai/jina-embeddings-v2-base-es'.")
  parser.add_argument('--path_to_stories', type=str, default='stories_tensors.tsv', 
                      help="Path to the TSV file containing story vectors. Default is 'stories_tensors.tsv'.")
  parser.add_argument('--path_to_poems', type=str, default='poems_tensors.tsv', 
                      help="Path to the TSV file containing poem vectors. Default is 'poems_tensors.tsv'.")
  parser.add_argument('--metadata_stories', type=str, default='stories_metadata.tsv', 
                      help="Path to the TSV file containing story metadata. Default is 'stories_metadata.tsv'.")
  parser.add_argument('--metadata_poems', type=str, default='poems_metadata.tsv', 
                      help="Path to the file containing poem metadata. Default is 'poems_metadata.tsv'.")
  parser.add_argument('--type_of_search', type=str, default='both', choices=['stories', 'poems', 'both'], 
                      help="Specify the type of search: 'stories', 'poems', or 'both'. Default is 'both'.")
  parser.add_argument('--top_k', type=int, default=3, 
                      help="Number of top results to return. Default is 3.")
  parser.add_argument('--texts', type=str,  default='Una ballena que se introduce en el mar y ocurren misterios incomprendidos', 
                      help="Text(s) to query the database. Provide one or more strings.")

  # Parse the arguments
  args = parser.parse_args()

  # Initialize the CoemDB instance
  if 'CoemDB_instance' not in globals():
      CoemDB_instance = CoemDB(args.path_to_stories, args.path_to_poems, args.metadata_stories, args.metadata_poems, args.modelname)
      print("CoemDB_instance instantiated.")
  else:
      print("CoemDB_instance is already instantiated.")
  # Query the database
  result = CoemDB_instance.query(texts=[args.texts], type_mode=args.type_of_search, top_k=int(args.top_k))

  # Print the result
  print(result)


if __name__ == "__main__":
  main()  