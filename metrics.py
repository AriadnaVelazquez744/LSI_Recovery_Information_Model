import numpy as np
def precision(relevant_retrieved, all_retrieved):
    return len(relevant_retrieved) / len(all_retrieved) if all_retrieved else 0

def recall(relevant_retrieved, all_relevant):
    return len(relevant_retrieved) / len(all_relevant) if all_relevant else 0

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

def calculate_metrics(test_results, requested_metrics):
    """Calcula las métricas solicitadas basándose en los resultados de prueba"""
    metrics = {}
    if 'precision' in requested_metrics:
        metrics['precision'] = precision(test_results['relevant_retrieved'], test_results['all_retrieved'])
    if 'recall' in requested_metrics:
        metrics['recall'] = recall(test_results['relevant_retrieved'], test_results['all_relevant'])
    if 'f1' in requested_metrics and 'precision' in metrics and 'recall' in metrics:
        metrics['f1'] = f1_score(metrics['precision'], metrics['recall'])
    return metrics


def calculate_all_metrics(test_results,requested_metrics):
    all_results = {metrics: [] for metrics in requested_metrics}
    for test in test_results.values():
        result = calculate_metrics(test,requested_metrics)
        for key,value in result.items():
            all_results[key].append(value)
    
    score = {key:np.mean(values) for key,values in all_results.items()}
    
    return score