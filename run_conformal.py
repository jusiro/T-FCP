
import argparse

# Pats and configs
from data.configs.tasks_batch import *
from local_data.constants import *

# Conformal procedure runners
from conformal.procedures import icp, scp, icpt, fcp

def main():
    parser = argparse.ArgumentParser()

    # Tasks
    parser.add_argument('--tasks', default=TASKS_FG)

    # Model to employ
    parser.add_argument('--backbone', default='CLIP-ViTB16',
     help='"CLIP-ViTB32", "CLIP-ViTB16", "CLIP-ViTL14", "MetaCLIP-ViTB16", "MetaCLIP-ViTL14", "MetaCLIP-ViTH14"'
          )
          
    # Number of seeds.
    parser.add_argument('--seeds', default=50, type=int, help='Number of experiments repetitions')
    
    # Calibration sample size.
    parser.add_argument('--K', default=16, type=int, help='Number of samples per category, such that N=KxC')
    parser.add_argument('--split', default="test", help='train-test / test')
    parser.add_argument('--limit_test', default=True, type=lambda x: (str(x).lower() == 'true'))

    # Linear probing solver.
    parser.add_argument('--solver', default='gd', type=str, help='gd - lda - ot - tim')

    # Target alpha.
    parser.add_argument('--alpha_icp', default=0.100, type=float, help='Miscoverage level desired for inductive conformal prediction.')
    parser.add_argument('--alpha_fcp', default=0.000, type=float, help='Miscoverage level desired for full conformal prediction.')

    # Procedure
    parser.add_argument('--procedure', default="scp", help='icp - icpt - scp - fcp - tfcp')

    args, unknown = parser.parse_known_args()
    
    if args.procedure == "icp":
        icp.process(args=args)
    elif args.procedure == "icpt":
        icpt.process(args=args)
    elif args.procedure == "scp":
        scp.process(args=args)
    elif args.procedure == "fcp":
        fcp.process(args=args)
    elif args.procedure == "tfcp":
        fcp.process(args=args)
    else:
        print("Conformal procedure not implemented.")
        return


if __name__ == "__main__":
    main()