
import argparse
import sys
from pathlib import Path

# Import from same directory (works when run as script)
thresholds_dir = Path(__file__).parent
if str(thresholds_dir) not in sys.path:
    sys.path.insert(0, str(thresholds_dir))
from threshold_tuner_impl import tune_thresholds, CHEXPERT14

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='CSV with y_true_<L>, y_pred_<L> columns')
    ap.add_argument('--mode', default='fbeta', choices=['fbeta', 'min_precision'])
    ap.add_argument('--beta', type=float, default=0.5, help='F-beta (precision emphasis if <1)')
    ap.add_argument('--min_macro_precision', type=float, default=0.60, help='Target precision for min_precision mode')
    ap.add_argument('--out_json', default='thresholds.json')
    ap.add_argument('--out_metrics', default='thresholds_summary.csv')
    ap.add_argument('--labels', nargs='*', default=CHEXPERT14, help='Labels to tune (default: all CHEXPERT14)')
    args = ap.parse_args()

    result = tune_thresholds(csv_path=args.csv,
                             out_json=args.out_json,
                             out_metrics=args.out_metrics,
                             mode=args.mode,
                             beta=args.beta,
                             min_macro_precision=args.min_macro_precision,
                             labels=args.labels)
    print(result)

if __name__ == '__main__':
    main()
