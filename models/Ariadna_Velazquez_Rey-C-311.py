
NAME = "Ariadna Velázquez Rey"
GROUP = "311"
CAREER = "Ciencia de la Computación"
MODEL = "Modelo de Semántica Latente (LSI)"

"""
INFORMACIÓN EXTRA:

Fuente bibliográfica: Deerwester, S., et al. (1990). 
"Indexing by Latent Semantic Analysis". Journal of the American Society for Information Science.
...

Mejora implementada:
...

Definición del modelo:
Q: Consultas representadas como vectores en espacio latente (SVD(query))
D: Documentos proyectados en espacio latente (SVD(docs))
F: Función de similitud coseno entre vectores latentes
R: Ranking por orden descendente de similitud

¿Dependencia entre los términos?
No (pero captura relaciones semánticas latentes)
...

Correspondencia parcial documento-consulta:
Sí (similitud de cosenos en espacio latente)
...

Ranking:
Sí (ordenamiento por score de similitud)
...

"""


import ir_datasets
import random


import numpy as np
import spacy
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from typing import Dict, List

def _load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("Ejecuta 'uv pip install en_core_web_sm' primero!")


class InformationRetrievalModel:
    def __init__(self, n_components=200):
        """
        Inicializa el modelo de recuperación de información.
        """
        self.pipeline = None          # TF-IDF + SVD + Normalizer
        self.doc_vectors = None       # Matriz de documentos en espacio latente
        self.doc_ids = []              # IDs originales de documentos
        self.queries = {}              # {query_id: texto_pre-procesado}
        self.n_components = n_components  # Dimensiones del espacio latente
        self.nlp = _load_spacy_model()  # Carga diferida

    def _preprocess(self, text: str) -> str:
        """
        Procesamiento lingüístico completo
        """
        doc = self.nlp(text)  # 1. Tokenización con spaCy
        tokens = [
            token.lemma_.lower()       # 2. Lematización (raíz de palabra)
            for token in doc
            if not token.is_stop and    # 3. Filtra stopwords (el, la, y...)
            token.is_alpha and          # 4. Elimina números/puntuación
            len(token.lemma_) > 2       # 5. Descarta palabras muy cortas
        ]
        return " ".join(tokens)  # 6. Une tokens en string limpio


    def fit(self, dataset_name: str):
        """
        Carga y procesa un dataset de ir_datasets, incluyendo todas sus queries.
        
        Args:
            dataset_name (str): Nombre del dataset en ir_datasets (ej: 'cranfield')
        """
        # 1. Carga dataset
        self.dataset = ir_datasets.load(dataset_name)
        
        # 2. Extrae y pre-procesa documentos
        self.doc_ids = [doc.doc_id for doc in self.dataset.docs_iter()]
        processed_docs = [self._preprocess(doc.text) for doc in self.dataset.docs_iter()]
        
        # 3. Construye pipeline de transformación
        self.pipeline = make_pipeline(
            TfidfVectorizer(
                max_df=0.8,    # Ignora términos en >80% docs
                min_df=2       # Ignora términos en <2 docs
            ),
            TruncatedSVD(      # Reducción dimensional
                n_components=self.n_components,
                algorithm='arpack'  # Algoritmo estable para matrices sparse
            ),
            Normalizer(norm='l2')   # Normalización de vectores
        )
        
        # 4. Entrena modelo y transforma documentos
        self.doc_vectors = self.pipeline.fit_transform(processed_docs)
        
        # 5. Pre-procesa y almacena queries
        self.queries = {
            q.query_id: self._preprocess(q.text)
            for q in self.dataset.queries_iter()
        }
    
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
            # # Aplicar expansión de consulta
            # expanded_query = self._expand_query(query_text)
            
            # # 1. Transforma query al espacio latente
            # query_vec = self.pipeline.transform([expanded_query])
            
            # 1. Transforma query al espacio latente
            query_vec = self.pipeline.transform([query_text])
            
            # 2. Calcula similitud coseno con documentos
            similarities = cosine_similarity(query_vec, self.doc_vectors)
            
            # 3. Ordena documentos por relevancia
            top_indices = np.argsort(similarities[0])[::-1][:top_k]
            results[qid] = {
                'text': query_text,  
                'results': [self.doc_ids[i] for i in top_indices] 
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

    def _expand_query(self, original_query: str, top_docs: int = 3, top_terms: int = 5) -> str:
        """
        Mejora: Expande la consulta con términos relevantes de los primeros documentos recuperados
        Técnica: Pseudo-Relevance Feedback (PRF)
        """
        # 1. Recuperación inicial de documentos
        query_vec = self.pipeline.transform([original_query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)
        top_indices = np.argsort(similarities[0])[::-1][:top_docs]
        
        # 2. Extracción de términos relevantes
        vectorizer = self.pipeline.named_steps['tfidfvectorizer']
        feature_names = vectorizer.get_feature_names_out()
        
        # 3. Cálculo de pesos de términos en documentos relevantes
        doc_weights = np.sum(self.doc_vectors[top_indices], axis=0)
        top_terms_indices = np.argsort(doc_weights)[-top_terms:]
        
        # 4. Construcción de consulta expandida
        expanded_terms = [feature_names[i] for i in top_terms_indices]
        return f"{original_query} {' '.join(expanded_terms)}"
