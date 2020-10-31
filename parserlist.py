import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--gamma", type=float, default=1000000)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--K", type=int, default=1000) # for n-Smectic transition with large K
parser.add_argument("--solver-type", choices=["lu", "allu", "almg-star", "mgvanka", "fasvanka", "faspardecomp", "almg-pbj", "almg-j"], default="almg-star")
parser.add_argument("--stab-type", choices=["discrete", "continuous"], default="continuous")
parser.add_argument("--prolong-type", choices=["auto", "none"], default="none")
parser.add_argument("--pressure-element", choices=["DG0", "CG1"], default="CG1")
parser.add_argument("--nonlinear-iteration", choices=["newton", "picard"], default="picard")
parser.add_argument("--improv-constraint", choices=["True", "False"], default="False")
