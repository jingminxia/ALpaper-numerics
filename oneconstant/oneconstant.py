from firedrake import *
from datetime import datetime
import argparse
import sys
sys.path.append("./")
import solveroptions, parserlist
from mpi4py import MPI

args, _ = parserlist.parser.parse_known_args()

def mdiv(n):
    return n[0].dx(0) + n[1].dx(1)

def mcurl(n):
    return as_vector([n[2].dx(1), -n[2].dx(0), n[1].dx(0) - n[0].dx(1)])

base = Mesh("./mesh/coarsetri.msh", distribution_parameters=solveroptions.distribution_parameters)

def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)

mh = MeshHierarchy(base, args.nref, callbacks=(before, after), distribution_parameters=solveroptions.distribution_parameters)
mesh = mh[-1]

Vele = VectorElement("CG", mesh.ufl_cell(), args.k, dim=3)

if args.pressure_element == "DG0":
    Qele = FiniteElement("DG", mesh.ufl_cell(), 0)
elif args.pressure_element == "CG1":
    Qele = FiniteElement("CG", mesh.ufl_cell(), 1)
else:
    raise ValueError

Zele = MixedElement([Vele, Qele])

V = FunctionSpace(mesh, Vele)
Z = FunctionSpace(mesh, Zele)

K1 = Constant(1)
K2 = Constant(1)
K3 = Constant(1)
q0 = Constant(0)
gamma = Constant(args.gamma)

(x, y) = SpatialCoordinate(mesh)
#n0 = as_vector([sin(x)*cos(y), cos(x)*sin(y), 0.5])
n0 = Constant((0, 0, 0.8))
bcs = [DirichletBC(Z.sub(0), Constant((0, 0, 1)), "on_boundary")]

z = Function(Z)
z.split()[0].interpolate(n0)
dofs = z.function_space().dim()
if MPI.COMM_WORLD.size == 1:
    print(GREEN % ("dofs: %s" % dofs))

(n, lmbda) = split(z)

E = 0.5 * inner(grad(n), grad(n))*dx

if args.stab_type == "discrete":
    op = cell_avg
else:
    op = lambda x: x

constraint = op(dot(n, n) - 1)
L = E + inner(lmbda, constraint)*dx
v = TestFunction(Z)
w = TrialFunction(Z)

(m, mu) = split(v)
(p, nu) = split(w)

F_orig = derivative(L, z, v)

L_aug = L + gamma/2 * inner(op(dot(n, n) - 1), op(dot(n, n) - 1))*dx
F_aug = derivative(L_aug, z, v)
if args.nonlinear_iteration == "picard":
    J_aug = derivative(F_aug, z, w) - 2*gamma*inner(dot(n, n)-1, dot(p, m))*dx
else:
    J_aug = derivative(F_aug, z, w)

class Mass(AuxiliaryOperatorPC):
  def form(self, pc, test, trial):
      a = -(1 + gamma)**(-1) * inner(test, trial)*dx
      bcs = None
      return (a, bcs)

choice = {"almg-star": solveroptions.fieldsplit_with_mg,
          "allu": solveroptions.fieldsplit_with_chol,
          "lu": solveroptions.splu,
          "mgvanka": solveroptions.vanka,
          "fasvanka": solveroptions.fasvanka,
          "faspardecomp": solveroptions.faspardecomp,
          "almg-pbj": solveroptions.fieldsplit_pbjacobi,
          "almg-j": solveroptions.fieldsplit_jacobi}[args.solver_type]

if args.solver_type in ["fasvanka", "faspardecomp"]:
    solveroptions.common = choice
else:
    if args.improv_constraint == "on":
        solveroptions.common.update(choice)
        solveroptions.common["ksp_rtol"] = 1e-4
        solveroptions.common["ksp_atol"] = 1e-10
    else:
        solveroptions.common.update(choice)

if MPI.COMM_WORLD.size == 1:
    import pprint; pprint.pprint(solveroptions.common)

nvproblem = NonlinearVariationalProblem(F_aug, z, bcs, J=J_aug)
nvsolver  = NonlinearVariationalSolver(nvproblem, solver_parameters=solveroptions.common)

start = datetime.now()
nvsolver.solve()
end = datetime.now()

constraintnorm = sqrt(assemble(inner(constraint, constraint)*dx))
b = assemble(F_orig)
[bc.zero(b) for bc in bcs]
with b.dat.vec_ro as x:
    resnorm = x.norm()

linear_its = nvsolver.snes.getLinearSolveIterations()
nonlinear_its = nvsolver.snes.getIterationNumber()
time = (end-start).total_seconds() / 60
print(BLUE % ("Time taken: %.2f min in %d/%d iterations (%.2f Krylov iters per Newton step)" % (time, linear_its, nonlinear_its, linear_its/float(nonlinear_its))))
print(BLUE % ("||n·n - 1||  = %.14e" % constraintnorm))
print(BLUE % ("||residual|| = %.14e" % resnorm))

(n_, lmbda_) = z.split()
n_.rename("Director")
File("output/director.pvd").write(n_)
