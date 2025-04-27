
NAME = "Francisco Guerra Piloto"
GROUP = "311"
CAREER = "Ciencia de la Computación"
MODEL = "Modelo Aleatorio de ejemplo"

"""
INFORMACIÓN EXTRA:

Fuente bibliográfica:
...

Mejora implementada:
...

Definición del modelo:
Q: ... 
D: ...
F: ...
R: ...

¿Dependencia entre los términos?
...

Correspondencia parcial documento-consulta:
...

Ranking:
...

"""


import ir_datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import random
from typing import Dict, List, Tuple

class InformationRetrievalModel:
    def __init__(self):
        """
        Inicializa el modelo de recuperación de información.
        """
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.documents = []
        self.doc_ids = []
        self.dataset = None
        self.queries = {}
    
    def fit(self, dataset_name: str):
        """
        Carga y procesa un dataset de ir_datasets, incluyendo todas sus queries.
        
        Args:
            dataset_name (str): Nombre del dataset en ir_datasets (ej: 'cranfield')
        """
        # Cargar dataset
        self.dataset = ir_datasets.load(dataset_name)
        
        if not hasattr(self.dataset, 'queries_iter'):
            raise ValueError("Este dataset no tiene queries definidas")
        
        self.documents = []
        self.doc_ids = []
        
        for doc in self.dataset.docs_iter():
            self.doc_ids.append(doc.doc_id)
            self.documents.append(doc.text)
        
        self.queries = {q.query_id: q.text for q in self.dataset.queries_iter()}
    
    def predict(self, top_k: int) -> Dict[str, Dict[str, List[str]]]:
        """
        Realiza búsquedas para TODAS las queries del dataset automáticamente.
        
        Args:
            top_k (int): Número máximo de documentos a devolver por query.
            threshold (float): Umbral de similitud mínimo para considerar un match.
            
        Returns:
            dict: Diccionario con estructura {
                query_id: {
                    'text': query_text,
                    'results': [(doc_id, score), ...]
                }
            }
        """
        results = {}
        
        for qid, query_text in self.queries.items():
            random_docs = random.sample(self.doc_ids, min(top_k, len(self.doc_ids)))
            results[qid] = {
                'text': query_text,
                'results': random_docs
            }
        
        return results
    
    def evaluate(self, top_k: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Evalúa los resultados para TODAS las queries comparando con los qrels oficiales.
        
        Args:
            top_k (int): Número máximo de documentos a considerar por query.
            
        Returns:
            dict: Métricas de evaluación por query y métricas agregadas.
        """
        if not hasattr(self.dataset, 'qrels_iter'):
            raise ValueError("Este dataset no tiene relevancias definidas (qrels)")
        
        predictions = self.predict(top_k=top_k)
        
        qrels = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        result = {}
        
        for qid, data in predictions.items():
            if qid not in qrels:
                continue
                
            relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
            retrieved_docs = set(data['results'])
            relevant_retrieved = relevant_docs & retrieved_docs
            
            result[qid] = {
                'all_relevant': relevant_docs,
                'all_retrieved': retrieved_docs,
                'relevant_retrieved': relevant_retrieved
            }
        
        return result
