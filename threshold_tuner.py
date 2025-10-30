
import argparse
from threshold_tuner_impl import tune_thresholds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV with y_true_<L>, y_pred_<L> columns')
    ap.add_argument('--mode', default='fbeta', choices=['fbeta', 'min_precision'])
    ap.add_argument('--beta', type=float, default=0.5, help='F-beta (precision emphasis if <1)')
    ap.add_argument('--min_macro_precision', type=float, default=0.60, help='Target precision for min_precision mode')
    ap.add_argument('--out_json', default='thresholds.json')
    ap.add_argument('--out_metrics', default='thresholds_summary.csv')
    args = ap.parse_args()

    result = tune_thresholds(csv_path=args.csv,
                             out_json=args.out_json,
                             out_metrics=args.out_metrics,
                             mode=args.mode,
                             beta=args.beta,
                             min_macro_precision=args.min_macro_precision)
    print(result)

if __name__ == '__main__':
    main()
