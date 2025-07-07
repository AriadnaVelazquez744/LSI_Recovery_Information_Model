
NAME = "Ariadna Velázquez Rey"
GROUP = "311"
CAREER = "Ciencia de la Computación"
MODEL = "Modelo de Semántica Latente (LSI)"

"""
INFORMACIÓN EXTRA:

Fuente bibliográfica: 
Deerwester, S., et al. (1990). "Indexing by Latent Semantic Analysis". Journal of the American Society for Information Science.
...

Mejora implementada:
    1. Preprocesamiento mejorado con bigramas y corrección léxica:
    - Detección automática de pares de términos frecuentes (ej: "machine_learning")
    - Corrección de errores tipográficos específicos ("teh" → "the", "adn" → "and", "th e" → "the")
    -. Beneficio: Mejora la cobertura de términos relevantes

    2. Expansión semiautomática de consultas (Pseudo-Relevance Feedback):
    - Recupera documentos iniciales usando consulta original
    - Extrae términos relevantes de los top-3 documentos
    -. Beneficio: Mitiga problemas de vocabulario escaso

    3. Ponderación adaptativa de términos en consultas:
    - Asigna pesos usando fórmula (1 + log(tf)) * idf
    - Formato de consulta mejorado (ej: "información^2.3 retrieval^1.8")
    -. Beneficio: Prioriza términos discriminativos

    4. Filtrado dinámico por umbral de similitud:
    - Descarta documentos con score < 65% del máximo
    - Justificación: El umbral se calcula como el 65% del score máximo de cada consulta específica, no es un valor fijo predefinido. Esto significa que se adapta automáticamente a la distribución de similitudes de cada consulta individual.
    -. Beneficio: Reduce ruido en resultados

...

Definición del modelo:
Q: Consultas representadas como vectores en espacio latente (SVD(query)).
D: Documentos proyectados en espacio latente (SVD(docs)).
F: Framework scikit-learn con TruncatedSVD, TfidfVectorizer y cosine_similarity.
R: sim(dj, q) en el rango [0,1], donde mayores valores indican mayor similitud semántica.

¿Dependencia entre los términos?
Sí, SVD captura co-ocurrencias latentes en espacio reducido.
...

Correspondencia parcial documento-consulta:
Sí, se realiza matching semántico vía similitud coseno en el espacio latente.
...

Ranking:
Sí, ordenamiento por sim(Q,D) con filtrado de documentos marginalmente relevantes.
...
"""

import ir_datasets
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

    def fit(self, dataset_name: str):
        """
        Carga y procesa un dataset de ir_datasets, incluyendo todas sus queries.
        
        Args:
            dataset_name (str): Nombre del dataset en ir_datasets (ej: 'cranfield')
        """
        # Carga dataset
        self.dataset = ir_datasets.load(dataset_name)
        
        # Extrae y pre-procesa documentos
        self.doc_ids = [doc.doc_id for doc in self.dataset.docs_iter()]
        processed_docs = [self._text_preprocess(doc.text) for doc in self.dataset.docs_iter()]
        
        # Construye pipeline de transformación
        self.pipeline = make_pipeline(
            TfidfVectorizer(
                max_df=0.8,    # Ignora términos en >80% docs
                min_df=2       # Ignora términos en <2 docs
            ),
            TruncatedSVD(      # Reducción dimensional
                n_components=self.n_components,
                algorithm='randomized',  # Algoritmo estable para matrices densas
            ),
            Normalizer(norm='l2')   # Normalización de vectores
        )
        
        # Entrena modelo y transforma documentos
        self.doc_vectors = self.pipeline.fit_transform(processed_docs)
        
        # Pre-procesa y almacena queries
        self.queries = {
            q.query_id: self._text_preprocess(q.text)
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
            # Aplicar expansión y peso de la consulta 
            expanded_query = self._expand_query(query_text)
            weighted_query = self._weight_terms(expanded_query)         
            
            # Transforma query al espacio latente
            query_vec = self.pipeline.transform([weighted_query])
            
            # Calcula similitud coseno con documentos
            similarities = cosine_similarity(query_vec, self.doc_vectors)
            
            # Ordena documentos por relevancia, filtrando la relevancia para solo los score mayores incluso si no completa el top_k
            similarities_array = similarities[0]
            threshold = np.max(similarities_array) * 0.65  # 65% del score máximo
            mask = similarities_array >= threshold
            filtered_indices = np.where(mask)[0]
            sorted_filtered_indices = filtered_indices[np.argsort(-similarities_array[filtered_indices])]
            top_indices = sorted_filtered_indices[:top_k]

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

    def _text_preprocess(self, text: str) -> str:
        """
        Procesamiento lingüístico completo.
        Detección de bigramas y correcciones básicas
        """
        doc = self.nlp(text)     # Tokenización con spaCy
        
        # Detección de bigramas frecuentes
        bigrams = []
        for token in doc[:-1]:
            if not token.is_stop and not doc[token.i+1].is_stop:
                bigrams.append(f"{token.lemma_}_{doc[token.i+1].lemma_}")
        
        # Corrección de errores comunes
        corrections = {
            'teh': 'the', 'adn': 'and', 'th e': 'the'
        }
        
        tokens = [
            corrections.get(token.lemma_.lower(), token.lemma_.lower())
            for token in doc
            if not token.is_stop and 
            token.is_alpha and 
            len(token.lemma_) > 2
        ] + bigrams
        
        return " ".join(tokens)


    def _expand_query(self, original_query: str, top_docs: int = 3, top_terms: int = 5) -> str:
        """
        Mejora: Expande la consulta con términos relevantes de los primeros documentos recuperados
        Técnica: Pseudo-Relevance Feedback (PRF)
        """
        # Recuperación inicial de documentos
        query_vec = self.pipeline.transform([original_query])
        similarities = cosine_similarity(query_vec, self.doc_vectors)
        top_indices = np.argsort(similarities[0])[::-1][:top_docs]
        
        # Extracción de términos relevantes
        vectorizer = self.pipeline.named_steps['tfidfvectorizer']
        feature_names = vectorizer.get_feature_names_out()
        
        # Cálculo de pesos de términos en documentos relevantes
        doc_weights = np.sum(self.doc_vectors[top_indices], axis=0)
        top_terms_indices = np.argsort(doc_weights)[-top_terms:]
        
        # Construcción de consulta expandida
        expanded_terms = [feature_names[i] for i in top_terms_indices]
        return f"{original_query} {' '.join(expanded_terms)}"


    def _weight_terms(self, query: str) -> str:
        """
        Asigna pesos a términos usando: (1 + log(tf)) * idf
        Formato: "term^2.5 other_term^0.8"
        """
        vectorizer = self.pipeline.named_steps['tfidfvectorizer']
        tf = vectorizer.transform([query]).sum(axis=0).A1
        idf = vectorizer.idf_
        
        weighted_terms = []
        for term in query.split():
            if term in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[term]
                weight = (1 + np.log1p(tf[idx])) * idf[idx]
                weighted_terms.append(f"{term}^{round(weight, 1)}")
            else:
                weighted_terms.append(term)
                
        return ' '.join(weighted_terms)