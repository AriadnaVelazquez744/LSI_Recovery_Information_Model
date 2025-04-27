from tabulate import tabulate

def generate_ranking_table(results):
    # Preparar datos para la tabla
    table_data = []
    for result in sorted(results, key=lambda x: sum(x['scores'].values()), reverse=True):
        row = [result['model_name']]
        row.extend([f"{result['scores'].get(m, 0):.3f}" for m in sorted(result['scores'])])
        row.append(result['error'] or 'OK')
        table_data.append(row)
    
    # Encabezados
    headers = ['Model'] + sorted(results[0]['scores'].keys()) + ['Status']
    
    return tabulate(table_data, headers=headers, tablefmt='grid')
