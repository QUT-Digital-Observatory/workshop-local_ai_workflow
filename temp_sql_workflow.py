#temp sql workflow

import numpy as np
import pandas as pd
import sqlite3
import os
import json
from tqdm import tqdm
import spacy
from openai import OpenAI
import yaml
from bertopic import BERTopic
import torch
from sentence_transformers import SentenceTransformer
import ollama
from typing import List, Union

os.environ["OPENAI_API_KEY"] = "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"

client = OpenAI()

nlp = spacy.load("en_core_web_trf")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_config(config_path: str) -> dict:
    """
    Load the configuration file from the specified path.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_path = 'rob_config.yaml'

config = load_config(config_path)
hardware = config.get('hardware', 'GPU')

if hardware == 'GPU':
    from cuml.manifold import UMAP
    from cuml.cluster import HDBSCAN
    print("Using GPU for UMAP and HDBSCAN.")
else:
    from umap import UMAP
    from hdbscan import HDBSCAN
    print("Using CPU for UMAP and HDBSCAN.")

class DataImporter:
    def __init__(self, table_name, embedding_model = embedding_model, db_path = 'temp.db'):
        self.db_path = db_path
        self.table_name = table_name
        self.embedding_model = embedding_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_column = None
      
    def _set_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def import_data(self, file_path, file_type='csv', text_column=None):
        if text_column is None:
            raise ValueError("text_column must be specified")
        self.text_column = text_column

        if file_type.lower() == 'csv':
            self._import_csv(file_path)
        elif file_type.lower() == 'parquet':
            self._import_parquet(file_path)
        elif file_type.lower() == 'excel':
            self._import_excel(file_path)    
        else:
            raise ValueError("Unsupported file type. Use 'csv', 'parquet', or 'excel'.")
        
        self._add_embeddings()
        
    def _import_csv(self, file_path):
        df = pd.read_csv(file_path)
        self._validate_and_write_to_db(df)
        
    def _import_parquet(self, file_path):
        df = pd.read_parquet(file_path)
        self._validate_and_write_to_db(df)

    def _import_excel(self, file_path):
        df = pd.read_excel(file_path)
        self._validate_and_write_to_db(df)

    def _validate_and_write_to_db(self, df):
        if self.text_column not in df.columns:
            raise ValueError(f"Specified text column '{self.text_column}' not found in the data")
        
        df[self.text_column] = df[self.text_column].fillna('')

        conn = sqlite3.connect(self.db_path)
        df.to_sql(self.table_name, conn, if_exists='replace', index=False)
        conn.close()

    def _add_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)

        df[self.text_column] = df[self.text_column].fillna('')
    
        # Generate embeddings
        embeddings = self.embedding_model.encode(
                df[self.text_column].tolist(), 
                device=self.device
            )
        
        # Ensure embeddings are always a list of floats
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.tolist()
        elif isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
    
        # Convert embeddings to a JSON string format suitable for SQLite
        df['embedding'] = [json.dumps(emb.tolist() if isinstance(emb, np.ndarray) else emb) for emb in embeddings]
    
        # Write back to SQLite
        df.to_sql(self.table_name, conn, if_exists='replace', index=False)
        conn.close()

    def get_embedding(self, row_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT embedding FROM {self.table_name} WHERE rowid = ?", (row_id,))
        result = cursor.fetchone()
        conn.close()
    
        if result:
            return np.array(json.loads(result[0]))
        else:
            return None

    def combine_text_columns(self, columns_to_combine, new_column_name='combined_text', separator=' '):
        """
        Combines specified text columns into a single column and updates the database.
    
        :param columns_to_combine: List of column names to combine
        :param new_column_name: Name of the new combined column (default: 'combined_text')
        :param separator: String to use as separator between column contents (default: ' ')
        """
    # Validate input
        if not isinstance(columns_to_combine, list) or len(columns_to_combine) < 2:
            raise ValueError("columns_to_combine must be a list with at least two column names")

    # Connect to the database
        conn = sqlite3.connect(self.db_path)
    
    # Read the current data
        df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
    
    # Check if all specified columns exist
        missing_columns = [col for col in columns_to_combine if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following columns are not in the database: {', '.join(missing_columns)}")
    
    # Combine the specified columns
        df[new_column_name] = df[columns_to_combine].fillna('').astype(str).agg(separator.join, axis=1)
    
    # Update the database
        df.to_sql(self.table_name, conn, if_exists='replace', index=False)
    
    # Close the connection
        conn.close()
    
    # Update the text_column attribute if it's one of the combined columns
        if self.text_column in columns_to_combine:
            self.text_column = new_column_name
    
        print(f"Columns {', '.join(columns_to_combine)} have been combined into {new_column_name}")
        print(f"Database has been updated with the new column")

    def check_table(self, table_name, n=5):
        """
        Retrieve and display the first n rows of the table.
        
        Args:
        n (int): Number of rows to retrieve. Default is 5.
        
        Returns:
        pandas.DataFrame: The first n rows of the table.
        """
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM {table_name} LIMIT {n}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # If 'embedding' column exists, truncate its content for display
        if 'embedding' in df.columns:
            df['embedding'] = df['embedding'].apply(lambda x: json.loads(x)[:5] + ['...'])
        
        return df

    def get_table_info(self, table_name):
        """
        Retrieve and display information about the table structure.
        
        Returns:
        list of tuples: Information about each column in the table.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        table_info = cursor.fetchall()
        conn.close()
        
        return table_info    
    
    def export_table_to_csv(self, table_name: str, output_file: str) -> None:
        conn = sqlite3.connect(self.db_path)

        # Query to select all data from the table
        query = f"SELECT * FROM {table_name}"

        # Load the data into a pandas DataFrame
        df = pd.read_sql_query(query, conn)

        # Close the database connection
        conn.close()

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)    

    def clean_up(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            print(f"Database file {self.db_path} has been removed.")
        else:
            print("The database file does not exist.")

#make llama ccp implementation, investigate onnx, make yaml switch for starting local llm as required.
class AIProcessor:
    def __init__(self, table_name, db_path = 'temp.db'):
        self.db_path = db_path
        self.table_name = table_name
        self.client = OpenAI()
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)  
       
    def process_table_llm(self, table_name, input_column, output_column, key, batch_size, system_prompt, user_prompt_template, rows_to_process='all'):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

    # Add new column to the original table if it doesn't exist
        try:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {output_column} TEXT")
        except sqlite3.OperationalError:
        # Ignore if the column already exists
            pass

    # Determine the total number of rows to process
        total_rows = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    
        if rows_to_process == 'all':
            rows_to_process = total_rows
        else:
            rows_to_process = min(int(rows_to_process), total_rows)

    # Create a tqdm progress bar
        with tqdm(total=rows_to_process, desc="Processing rows", unit="row") as pbar:
        # Process in batches
            for offset in range(0, rows_to_process, batch_size):
                current_batch_size = min(batch_size, rows_to_process - offset)
                cur.execute(f"SELECT {key}, {input_column} FROM {table_name} LIMIT {current_batch_size} OFFSET {offset}")
                rows = cur.fetchall()

                for row in rows:
                    row_id, input_text = row  # Unpack the row
                
                # Use the text from the input column in the user prompt
                    user_prompt = user_prompt_template.format(input_text=input_text)
                
                    output = self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0,
                    )

                    if output.choices:
                        completion = output.choices[0].message.content
                    else:
                        completion = 'unknown'

                    cur.execute(f"UPDATE {table_name} SET {output_column} = ? WHERE {key} = ?", 
                            (completion, row_id))
                
                # Update the progress bar
                    pbar.update(1)

                conn.commit()

                if offset + current_batch_size >= rows_to_process:
                    break

        conn.close()

    def read_from_sqlite(self, table_name):
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def process_df_bertopic(self, df, input_column):
        """
        Fit a BERTopic model to the input data.
        """
        umap_params = self.config['umap']
        hdbscan_params = self.config['hdbscan']
        bertopic_params = self.config['bertopic']
        # Initialize models with parameters from config
        if hardware == 'GPU':
            umap_model = UMAP(
            n_components=umap_params['n_components'],
            n_neighbors=umap_params['n_neighbors'],
            min_dist=umap_params['min_dist'],
            random_state=umap_params['random_state']
        )
            hdbscan_model = HDBSCAN(
            min_samples=hdbscan_params['min_samples'],
            gen_min_span_tree=hdbscan_params['gen_min_span_tree'],
            prediction_data=hdbscan_params['prediction_data']
        )
        else:
            umap_model = UMAP()
            hdbscan_model = HDBSCAN()    

        all_model = BERTopic(umap_model=umap_model, 
                         hdbscan_model=hdbscan_model,
                         calculate_probabilities=bertopic_params['calculate_probabilities'], 
                        verbose=bertopic_params['verbose'], 
                        min_topic_size=bertopic_params['min_topic_size'])
        all_model.fit_transform (df[input_column])
        results = all_model.get_document_info(df[input_column], df)

        self.update_database_with_results(results)
        return all_model
    
    def update_database_with_results(self, results):
        """
        Update the database with BERTopic results, handling list-type columns.
        """
    # Function to convert lists to JSON strings
        def convert_list_to_json(val):
            if isinstance(val, list):
                return json.dumps(val)
            return val

    # Apply the conversion to all columns
        results_processed = results.applymap(convert_list_to_json)

        conn = sqlite3.connect(self.db_path)
    
    # Write the processed results to the database
        results_processed.to_sql(self.table_name, conn, if_exists='replace', index=False)
    
        conn.close()

    def retrieve_results_from_db(self):
        """
        Retrieve results from the database, converting JSON strings back to lists where applicable.
        """
        conn = sqlite3.connect(self.db_path)
    
    # Read the data from the database
        df = pd.read_sql(f"SELECT * FROM {self.table_name}", conn)
    
    # Function to convert JSON strings back to lists
        def convert_json_to_list(val):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return val

    # Apply the conversion to all columns
        df_processed = df.applymap(convert_json_to_list)
    
        conn.close()
    
        return df_processed
    
    #update db with topics
    
    def topics_from_model(self, model):
        topics = model.get_topics()
        rep_docs = model.get_representative_docs()
    
        topics_out = []
    
        for topic_id, words in tqdm(topics.items(), desc="Extracting topics", unit="topic"):
            topic_words = ', '.join([word for word, _ in words[:10]])  # Get top 10 words
            topic_docs = rep_docs.get(topic_id, [''])[:3]  # Get up to 3 representative docs
            topics_out.append({
                'Topic': topic_id,
                'Representative Words': topic_words,
                'Representative Docs': ' | '.join(topic_docs)
            })
        return pd.DataFrame(topics_out)
    
   

    def label_topic_clusters(self, topics_out):
    # Group by Topic and get the first occurrence of Representative Words and Docs
        rep_data = topics_out.groupby('Topic').agg({
            'Representative Words': 'first',
            'Representative Docs': 'first'
        }).reset_index()

        for index, row in tqdm(rep_data.iterrows(), total=len(rep_data), desc="Labeling topics", unit="topic"):
            prompt = "Consider the following representative words and representative documents and label the topic cluster."
            text = f"Representative Words: {row['Representative Words']}\nRepresentative Docs: {row['Representative Docs']}"
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=0,
                )
                completion_text = completion.choices[0].message.content
            except Exception as e:
                print(f"Error generating completion for Topic {row['Topic']}: {e}")
                completion_text = "Error: Completion could not be generated."

        # Store the completion message or error notice back in the DataFrame
            rep_data.loc[index, 'Topic Label'] = completion_text  

    # Merge the labels back into the original DataFrame
        topics_out = topics_out.merge(rep_data[['Topic', 'Topic Label']], on='Topic', how='left')

        return topics_out

    def process_and_update_ollama(self, instruction, table_name, key_column, text_column, output_column, batch_size, rows_to_process, model_name, db_path='temp.db'):
    # Open a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

    # Add new column to the original table if it doesn't exist
        try:
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {output_column} TEXT")
        except sqlite3.OperationalError:
        # Ignore if the column already exists
            pass

    # Determine the total number of rows to process
        total_rows = min(rows_to_process, cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])

    # Process in batches
        for offset in range(0, total_rows, batch_size):
        # Calculate the number of rows in the current batch
            current_batch_size = min(batch_size, total_rows - offset)

        # Read a batch of data from the specified column
            cur.execute(f"SELECT {key_column}, {text_column} FROM {table_name} LIMIT {current_batch_size} OFFSET {offset}")
            rows = cur.fetchall()

        # Iterate through each row in the batch and apply the Ollama generate function
            for row in tqdm(rows):
                text = row[1]
                prompt = f"{instruction}\n\nText: {text}"
            
                try:
                    response = ollama.generate(model=model_name, prompt=prompt)
                    generated_text = response['response'].strip()
                except Exception as e:
                    print(f"Error processing row: {e}")
                    generated_text = f"Error: {str(e)}"

                row_id = row[0]

            # Update the original table with the new data
                cur.execute(f"UPDATE {table_name} SET {output_column} = ? WHERE {key_column} = ?", 
                            (generated_text, row_id))

        # Commit changes after each batch
            conn.commit()

        # Break if we have processed the desired number of rows
            if offset + current_batch_size >= rows_to_process:
                break

    # Close connection
        conn.close()        

