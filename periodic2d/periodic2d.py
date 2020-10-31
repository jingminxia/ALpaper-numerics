# -*- coding: utf-8 -*-
from firedrake import *
from datetime import datetime
import argparse
import sys
sys.path.append("./")
import solveroptions, parserlist

args, _ = parserlist.parser.parse_known_args()

def mdiv(n):
    return n[0].dx(0) + n[1].dx(1)

def mcurl(n):
    return as_vector([n[2].dx(1), -n[2].dx(0), n[1].dx(0) - n[0].dx(1)])

def before(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+1)

def after(dm, i):
    for p in range(*dm.getHeightStratum(1)):
        dm.setLabelValue("prolongation", p, i+2)

K1 = Constant(1.0)
K2 = Constant(1.2)
K3 = Constant(1.0)
theta0 = pi/8
q0 = Constant(0)
gamma = Constant(args.gamma)

class Periodic2DProblem(object):
    def __init__(self, baseN):
        super().__init__()
        self.baseN = baseN

    def periodise(self, m):
        coord_fs = VectorFunctionSpace(m, "DG", 1, dim=2)
        old_coordinates = m.coordinates
        new_coordinates = Function(coord_fs)

        domain = "{[i, j]: 0 <= i < old_coords.dofs and 0 <= j < new_coords.dofs}"
        instructions = """
        <float64> Y = 0
        <float64> pi = 3.141592653589793
        for i
            Y = Y + old_coords[i, 1]
        end
        for j
            new_coords[j, 0] = atan2(old_coords[j, 1], old_coords[j, 0]) / (pi* 2)
            new_coords[j, 0] = if(new_coords[j, 0] < 0, new_coords[j, 0] + 1, new_coords[j, 0])
            new_coords[j, 0] = if(new_coords[j, 0] == 0 and Y < 0, 1, new_coords[j, 0])
            new_coords[j, 0] = new_coords[j, 0] * Lx[0]
            new_coords[j, 1] = old_coords[j, 2] * Ly[0]
        end
        """

        cLx = Constant(1)
        cLy = Constant(1)

        par_loop((domain, instructions), dx,
                 {"new_coords": (new_coordinates, WRITE),
                  "old_coords": (old_coordinates, READ),
                  "Lx": (cLx, READ),
                  "Ly": (cLy, READ)},
                 is_loopy_kernel=True)

        return Mesh(new_coordinates)

    def mesh(self):
        base = CylinderMesh(self.baseN, self.baseN, radius=1, depth=1, longitudinal_direction="z", distribution_parameters=solveroptions.distribution_parameters)

        cymh = MeshHierarchy(base, args.nref, callbacks=(before, after), distribution_parameters=solveroptions.distribution_parameters)

        meshes = tuple(self.periodise(m) for m in cymh)
        mh = HierarchyBase(meshes, cymh.coarse_to_fine_cells, cymh.fine_to_coarse_cells)
        return mh[-1]

    def bcs(self, Z):
        bc1 = DirichletBC(Z.sub(0), Constant((cos(theta0), 0, -sin(theta0))), 1) #y=0
        bc2 = DirichletBC(Z.sub(0), Constant((cos(theta0), 0, +sin(theta0))), 2) #y=1
        bcs = [bc1, bc2] # periodic bcs
        return bcs

    def energy(self, n, mesh):
        kappa = K2/K3
        I = Identity(3)
        P = kappa * outer(n, n) + (I - outer(n, n))
        E = 0.5 * (
             K1 * inner(mdiv(n), mdiv(n))*dx
           + K3 * inner(dot(P, mcurl(n)), mcurl(n))*dx
           + 2*K2 * inner(q0, dot(n, mcurl(n)))*dx
           + K2 * inner(q0, q0)*dx(mesh)
            )
        return E

    def analytical_solution(self, mesh):
        (x, y) = SpatialCoordinate(mesh)
        n_exact = as_vector([cos(theta0 *(2*y-1) ), 0, sin(theta0 *(2*y-1))])
        return n_exact

    @staticmethod
    def analytical_energy():
        return 2*K2* theta0**2

    def render(self, z, imagename):
        if self.coords is None:
            # Function space to store coordinates for postprocessing
            P = FunctionSpace(z.function_space().mesh(), "CG", 1)
            x = interpolate(Expression("x[0]", degree=1), P)
            y = interpolate(Expression("x[1]", degree=1), P)
            self.coords = (x, y)

        # Make output directory
        try:
            os.makedirs("output/mathematicaviz")
        except:
            pass

        csvname = imagename.replace("png", "csv")

        f = open(csvname + ".tmp", "w")
        (x, y) = self.coords
        n = z.split()[0]
        for (x_, y_) in zip(x.vector().array(), y.vector().array()):
            (n1, n2, n3) = n((x_, y_))
            f.write("%.15e,%.15e,%.15e,%.15e,%.15e\n" % ((x_, y_, n1, n2, n3)))
        f.flush()
        f.close()
        os.rename(csvname + ".tmp", csvname) # mv is atomic

        cmd = "WolframScript -script visualize.wl %s %s" % (csvname, imagename)
        try:
            subprocess.call(cmd.split())
        except:
            raise
        finally:
            os.remove(csvname)

if __name__ == "__main__":

    problem = Periodic2DProblem(10)
    mesh = problem.mesh()
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

    n0 = as_vector([1, 0, 0])
    z = Function(Z)
    z.split()[0].interpolate(n0)
    dofs = z.function_space().dim()
    print(GREEN % ("dofs: %s" % dofs))

    (n, lmbda) = split(z)

    if args.stab_type == "discrete":
        op = cell_avg
    else:
        op = lambda x: x

    constraint = op(dot(n, n) - 1)
    L = problem.energy(n, mesh) + inner(lmbda, constraint)*dx
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
        solveroptions.common.update(choice)

    import pprint; pprint.pprint(solveroptions.common)

    bcs = problem.bcs(Z)
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
    
    E = assemble(problem.energy(n_, mesh))
    print(BLUE % ("Final energy: %.6f" % E))
    print(BLUE % ("Analytical energy: %.6f" % problem.analytical_energy()))

    n_exact = Function(Z).split()[0]
    n_exact.interpolate(problem.analytical_solution(mesh))
    l2error = errornorm(n_exact, n_, norm_type="L2")
    print(BLUE % ("L2-norm of error: %.6e" % l2error))
    h1error = errornorm(n_exact, n_, norm_type="H1")
    print(BLUE % ("H1-norm of error: %.6e" % h1error))
