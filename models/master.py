import importlib
import os
from metrics import calculate_all_metrics

def evaluate_model(model_module, corpus_path, metrics):
    try:
        model = model_module.InformationRetrievalModel()
        # Cargar el corpus
        # Aquí iría la lógica de evaluación real con el corpus
        model.fit(corpus_path)
        test_results = model.evaluate()  # Resultados simulados para el ejemplo
        scores = calculate_all_metrics(test_results, metrics)
        return {
            'model_name': model_module.__name__,
            'scores': scores,
            'error': None
        }
    except Exception as e:
        return {
            'model_name': model_module.__name__,
            'scores': {m: 0 for m in metrics},
            'error': str(e)
        }

def evaluate_all_models(corpus_path, metrics):
    results = []
    for filename in os.listdir(os.path.dirname(__file__)):
        if filename.endswith('.py') and filename not in ['__init__.py', 'master.py', 'template.py']:
            module_name = f"models.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                results.append(evaluate_model(module, corpus_path, metrics))
            except ImportError as e:
                results.append({
                    'model_name': filename[:-3],
                    'scores': {m: 0 for m in metrics},
                    'error': f"Import error: {str(e)}"
                })
    return results
