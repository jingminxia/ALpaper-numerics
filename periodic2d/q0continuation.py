# -*- coding: utf-8 -*-
from firedrake import *
from datetime import datetime
import numpy
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--nref", type=int, default=1)
parser.add_argument("--gamma", type=float, default=1000000)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--solver-type", choices=["lu", "allu", "almg-star", "mgvanka", "fasvanka", "faspardecomp", "almg-pbj", "almg-j"], default="almg-star")
parser.add_argument("--stab-type", choices=["discrete", "continuous"], default="continuous")
parser.add_argument("--prolong-type", choices=["auto", "none"], default="none")
parser.add_argument("--pressure-element", choices=["DG0", "CG1"], default="CG1")
args, _ = parser.parse_known_args()

distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

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

class OseenFrankProblem(object):
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
        base = CylinderMesh(self.baseN, self.baseN, radius=1, depth=1, longitudinal_direction="z", distribution_parameters=distribution_parameters)

        cymh = MeshHierarchy(base, args.nref, callbacks=(before, after), distribution_parameters=distribution_parameters)

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

class lcsolver(object):
    def __init__(self, problem):
        self.problem = problem
        self.q0 = q0
        mesh = problem.mesh()
        self.mesh = mesh

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
        self.Z = Z

        n0 = as_vector([1, 0, 0])
        z = Function(Z)
        self.z = z
        z.split()[0].interpolate(n0)
        dofs = z.function_space().dim()
        print(GREEN % ("dofs: %s" % dofs))

        (n, lmbda) = split(z)

        if args.stab_type == "discrete":
            op = cell_avg
        else:
            op = lambda x: x

        self.constraint = op(dot(n, n) - 1)
        L = problem.energy(n, mesh) + inner(lmbda, self.constraint)*dx
        v = TestFunction(Z)
        w = TrialFunction(Z)

        (m, mu) = split(v)
        (p, nu) = split(w)

        self.F_orig = derivative(L, z, v)

        L_aug = L + gamma/2 * inner(op(dot(n, n) - 1), op(dot(n, n) - 1))*dx
        F_aug = derivative(L_aug, z, v)
        J_aug = derivative(F_aug, z, w) - 2*gamma*inner(dot(n, n)-1, dot(p, m))*dx

        self.bcs = problem.bcs(Z)

        import sys
        sys.path.append("../")
        import solveroptions 

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
        #import pprint; pprint.pprint(solveroptions.common)

        nvproblem = NonlinearVariationalProblem(F_aug, z, bcs=self.bcs, J=J_aug)
        nvsolver  = NonlinearVariationalSolver(nvproblem, solver_parameters=solveroptions.common)
        self.solver = nvsolver

    def solve(self, q0):
        print(GREEN % ("Solving for q0 = %s" % q0))
        self.q0.assign(q0)

        try:
            start = datetime.now()
            self.solver.solve()
            end = datetime.now()

            constraintnorm = sqrt(assemble(inner(self.constraint, self.constraint)*dx))
            b = assemble(self.F_orig)
            [bc.zero(b) for bc in self.bcs]
            with b.dat.vec_ro as x:
                resnorm = x.norm()

            linear_its = self.solver.snes.getLinearSolveIterations()
            nonlinear_its = self.solver.snes.getIterationNumber()
            time = (end-start).total_seconds() / 60
            print(BLUE % ("Time taken: %.2f min in %d/%d iterations (%.2f Krylov iters per Newton step)" % (time, linear_its, nonlinear_its, linear_its/float(nonlinear_its))))
            print(BLUE % ("||n·n - 1||  = %.14e" % constraintnorm))
            print(BLUE % ("||residual|| = %.14e" % resnorm))

            (n_, lmbda_) = self.z.split()
            n_.rename("director")
            mesh = self.z.function_space().mesh()

            n_exact = Function(self.Z).split()[0]
            n_exact.interpolate(self.problem.analytical_solution(mesh))
            l2error = errornorm(n_exact, n_, norm_type="L2")
            print(BLUE % ("L2-norm of error: %.6e" % l2error))
            h1error = errornorm(n_exact, n_, norm_type="H1")
            print(BLUE % ("H1-norm of error: %.6e" % h1error))
        except ConvergenceError:
            print("Warning: the solver did not converge at q0=%s" % q0)
            nonlinear_its = 1
            linear_its = 0

        info_dict = {
                "q0": self.q0,
                "linear_iter": linear_its,
                "nonlinear_iter": nonlinear_its,
                "time": time,
                }
        return (n_, info_dict)

if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)

    class Mass(AuxiliaryOperatorPC):
      def form(self, pc, test, trial):
          a = -(1 + gamma)**(-1) * inner(test, trial)*dx
          bcs = None
          return (a, bcs)

    problem = OseenFrankProblem(10)
    solver = lcsolver(problem)

    upvdf = File("output/director.pvd")

    start = 0.0
    end = 8.0
    step = 0.1
    q0s = list(numpy.arange(start,end,step)) +[end]
    results = []
    for q0 in q0s:
        (n, info_dict) = solver.solve(q0)
        upvdf.write(n, time=q0)
        results.append(info_dict)

    linear_iter = [d["linear_iter"] for d in results]
    nonlinear_iter = [d["nonlinear_iter"] for d in results]
    
    pdfname = args.solver_type
    
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.xlabel(r"$q_0$")
    plt.title("%s" % pdfname)
    plt.plot(q0s, nonlinear_iter, 'b', label="SNES iter.")
    plt.legend()
    plt.savefig("output/SNES_q0_%s.pdf" % pdfname)

    plt.figure(2)
    plt.xlabel(r"$q_0$")
    plt.title("%s" % pdfname)
    plt.ylim((0,6))
    plt.plot(q0s, [a/b for a, b in zip(linear_iter, nonlinear_iter)], 'b', label="Aver. KSP iter.")
    plt.legend()
    plt.savefig("output/KSP_q0_" + pdfname + ".pdf")
