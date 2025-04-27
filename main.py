import argparse
from models.master import evaluate_all_models
from ranking import generate_ranking_table

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', help='Path to corpus directory', default='cranfield')
    parser.add_argument('--metrics', nargs='+', help='Metrics to compute', default=['precision', 'recall'])
    args = parser.parse_args()
    
    results = evaluate_all_models(args.corpus, args.metrics)
    print(generate_ranking_table(results))

if __name__ == "__main__":
    main()
