# Containing all the solver options here

from firedrake import DistributedMeshOverlapType
distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

common = {
     "snes_max_it":  1000,
     "snes_rtol": 1.0e-16,
     "snes_atol": 1.0e-8,
     "ksp_max_it": 100,
     "ksp_rtol": 1.0e-4,
     "ksp_atol": 1.0e-8,
     "ksp_monitor": None,
     "snes_stol":    0.0,
     "snes_monitor": None,
     "snes_linesearch_type": "basic",
     "snes_linesearch_damping": 1.0,
     "snes_linesearch_maxstep": 1.0,
     "snes_linesearch_monitor": None,
     "snes_converged_reason": None,
     }

splu = {
     "mat_type":    "aij",
     "ksp_type":    "preonly",
     "pc_type":     "lu",
     "pc_factor_mat_solver_type": "mumps",
     "mat_mumps_icntl_24": 1,
     "mat_mumps_icntl_13": 1
     }

vanka = {
     "mat_type":    "nest",
     "ksp_type":    "fgmres",
     "ksp_monitor_true_residual": None,
     "pc_type":     "mg",
     "mg_coarse_pc_type": "python",
     "mg_coarse_pc_python_type": "firedrake.AssembledPC",
     "mg_coarse_assembled_mat_type": "aij",
     "mg_coarse_assembled_ksp_type": "preonly",
     "mg_coarse_assembled_pc_type": "cholesky",
     "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
     "mg_levels_ksp_type": "gmres",
     "mg_levels_ksp_max_it": 3,
     "mg_levels_ksp_convergence_test": "skip",
     #"mg_levels_ksp_monitor_true_residual": None,
     #"mg_levels_ksp_norm_type": "unpreconditioned",
     "mg_levels_pc_type": "python",
     "mg_levels_pc_python_type": "firedrake.PatchPC",
     "mg_levels_patch_pc_patch_precompute_element_tensors": True,
     "mg_levels_patch_pc_patch_save_operators": True,
     "mg_levels_patch_pc_patch_partition_of_unity": True,
     "mg_levels_patch_pc_patch_multiplicative": False,
     "mg_levels_patch_pc_patch_symmetrise_sweep": False,
     "mg_levels_patch_pc_patch_sub_mat_type": "dense",
     "mg_levels_patch_pc_patch_dense_inverse": True,
     "mg_levels_patch_pc_patch_construct_type": "vanka",
     "mg_levels_patch_pc_patch_construct_dim": "0",
     "mg_levels_patch_pc_patch_exclude_dim": "1",
     "mg_levels_patch_sub_ksp_type": "preonly",
     "mg_levels_patch_sub_pc_type": "cholesky",
     }

fasvanka = {
       "mat_type": "matfree",
       "snes_type": "ngmres",
       "snes_monitor": None,
       "snes_ngmres_monitor": None,
       "snes_max_it": 100,
       "snes_atol": 1.0e-9,
       "snes_rtol": 0.0,
       "snes_stol": 0.0,
       "snes_divergence_tolerance": 0.0,
       "snes_npc_side": "right",
       "npc_snes_type": "fas",
       "npc_snes_fas_cycles": 1,
       "npc_snes_fas_type": "full",
       "npc_snes_fas_galerkin": False,
       "npc_snes_fas_smoothup": 1,
       "npc_snes_fas_smoothdown": 1,
       "npc_snes_fas_full_downsweep": False,
       "npc_snes_monitor": None,
       "npc_snes_max_it": 1,
       "npc_snes_convergence_test": "skip",
       "npc_fas_levels_snes_type": "ngmres",
       "npc_fas_levels_snes_monitor": None,
       "npc_fas_levels_snes_ngmres_monitor": None,
       "npc_fas_levels_snes_max_it": 2,
       "npc_fas_levels_snes_npc_side": "right",
       "npc_fas_levels_npc_snes_type": "python",
       "npc_fas_levels_npc_snes_python_type": "firedrake.PatchSNES",
       "npc_fas_levels_npc_snes_max_it": 1,
       "npc_fas_levels_npc_snes_convergence_test": "skip",
       "npc_fas_levels_npc_snes_converged_reason": None,
       "npc_fas_levels_npc_snes_monitor": None,
       "npc_fas_levels_npc_snes_linesearch_type": "l2",
       "npc_fas_levels_npc_snes_linesearch_maxstep": 1.0,
       "npc_fas_levels_npc_snes_linesearch_damping": 1.0,
       "npc_fas_levels_npc_patch_snes_patch_partition_of_unity": True,
       "npc_fas_levels_npc_patch_snes_patch_construct_type": "vanka",
       "npc_fas_levels_npc_patch_snes_patch_construct_dim": 0,
       "npc_fas_levels_npc_patch_snes_patch_exclude_subspaces": "1",
       "npc_fas_levels_npc_patch_snes_patch_vanka_dim": 0,
       "npc_fas_levels_npc_patch_snes_patch_sub_mat_type": "seqaij",
       "npc_fas_levels_npc_patch_snes_patch_local_type": "additive",
       "npc_fas_levels_npc_patch_snes_patch_symmetrise_sweep": False,
       "npc_fas_levels_npc_patch_sub_snes_type": "newtonls",
       "npc_fas_levels_npc_patch_sub_snes_atol": 1e-14,
       "npc_fas_levels_npc_patch_sub_snes_rtol": 1e-14,
       "npc_fas_levels_npc_patch_sub_snes_converged_reason": None,
       "npc_fas_levels_npc_patch_sub_snes_linesearch_type": "basic",
       "npc_fas_levels_npc_patch_sub_ksp_type": "preonly",
       "npc_fas_levels_npc_patch_sub_pc_type": "cholesky",
       "npc_fas_levels_npc_patch_sub_pc_factor_solver_type": "mumps",
       "npc_fas_coarse_snes_type": "newtonls",
       "npc_fas_coarse_snes_monitor": None,
       "npc_fas_coarse_snes_converged_reason": None,
       "npc_fas_coarse_snes_max_it": 100,
       "npc_fas_coarse_snes_atol": 1.0e-10,
       "npc_fas_coarse_snes_rtol": 1.0e-10,
       "npc_fas_coarse_snes_linesearch_monitor": None,
       "npc_fas_coarse_snes_linesearch_type": "l2",
       "npc_fas_coarse_snes_linesearch_maxstep": 1.0,
       "npc_fas_coarse_snes_linesearch_damping": 1.0,
       "npc_fas_coarse_ksp_type": "preonly",
       "npc_fas_coarse_pc_type": "python",
       "npc_fas_coarse_pc_python_type": "firedrake.AssembledPC",
       "npc_fas_coarse_assembled_mat_type": "aij",
       "npc_fas_coarse_assembled_pc_type": "cholesky",
       "npc_fas_coarse_assembled_pc_factor_mat_solver_type": "mumps",
       }

fieldsplit_with_chol_schur = {
     "mat_type":    "nest",
     "ksp_type":    "fgmres",
     "ksp_monitor_true_residual": None,
     "pc_type": "fieldsplit",
     "snes_divergence_tolerance": 1.0e10,
     "pc_fieldsplit_type": "schur",
     "pc_fieldsplit_schur_factorization_type": "full",
     "pc_fieldsplit_schur_precondition": "user",
     "fieldsplit_0_ksp_type": "preonly",
     "fieldsplit_0_pc_type": "python",
     "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
     "fieldsplit_0_assembled_pc_type": "cholesky",
     "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_1_ksp_type": "fgmres",
     "fieldsplit_1_ksp_max_it": 20000,
     "fieldsplit_1_ksp_atol": 1.0e-12,
     "fieldsplit_1_ksp_monitor_true_residual": None,
     "fieldsplit_1_pc_type": "python",
     "fieldsplit_1_pc_python_type": "__main__.Mass",
     "fieldsplit_1_aux_pc_type": "cholesky",
     "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
     }

fieldsplit_with_chol = {
     "mat_type":    "nest",
     "ksp_type":    "fgmres",
     "ksp_monitor_true_residual": None,
     "pc_type": "fieldsplit",
     "snes_divergence_tolerance": 1.0e10,
     "pc_fieldsplit_type": "schur",
     "pc_fieldsplit_schur_factorization_type": "full",
     "pc_fieldsplit_schur_precondition": "user",
     "fieldsplit_0_ksp_type": "preonly",
     "fieldsplit_0_pc_type": "python",
     "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
     "fieldsplit_0_assembled_pc_type": "cholesky",
     "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_0_assembled_mat_mumps_icntl_14": "200",
     "fieldsplit_1_ksp_type": "preonly",
     "fieldsplit_1_pc_type": "python",
     "fieldsplit_1_pc_python_type": "__main__.Mass",
     "fieldsplit_1_aux_pc_type": "cholesky",
     "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_1_aux_mat_mumps_icntl_14": "200",
     }

fieldsplit_jacobi = {
     "mat_type":    "nest",
     "ksp_type":    "fgmres",
     "ksp_monitor_true_residual": None,
     "pc_type": "fieldsplit",
     "snes_divergence_tolerance": 1.0e10,
     "pc_fieldsplit_type": "schur",
     "pc_fieldsplit_schur_factorization_type": "full",
     "pc_fieldsplit_schur_precondition": "user",
     "fieldsplit_0_ksp_type": "richardson",
     "fieldsplit_0_ksp_max_it": 1,
     "fieldsplit_0_ksp_convergence_test": "skip",
     #"fieldsplit_0_ksp_norm_type": "unpreconditioned",
     #"fieldsplit_0_ksp_monitor_true_residual": None,
     "fieldsplit_0_pc_type": "mg",
     "fieldsplit_0_mg_levels_ksp_type": "gmres",
     "fieldsplit_0_mg_levels_ksp_richardson_scale": 1/2,
     #"fieldsplit_0_mg_levels_ksp_norm_type": "unpreconditioned",
     #"fieldsplit_0_mg_levels_ksp_monitor_true_residual": None,
     "fieldsplit_0_mg_levels_ksp_max_it": 3,
     "fieldsplit_0_mg_levels_ksp_convergence_test": "skip",
     "fieldsplit_0_mg_levels_pc_type": "jacobi",
     "fieldsplit_0_mg_coarse_pc_type": "python",
     "fieldsplit_0_mg_coarse_pc_python_type": "firedrake.AssembledPC",
     "fieldsplit_0_mg_coarse_assembled_pc_type": "cholesky",
     "fieldsplit_0_mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_0_mg_coarse_aseembled_mat_mumps_icntl_14": "200",
     "fieldsplit_1_ksp_type": "preonly",
     "fieldsplit_1_pc_type": "python",
     "fieldsplit_1_pc_python_type": "__main__.Mass",
     "fieldsplit_1_aux_pc_type": "cholesky",
     "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_1_aux_mat_mumps_icntl_14": "200",
     }

fieldsplit_pbjacobi = {
     "mat_type":    "nest",
     "ksp_type":    "fgmres",
     "ksp_monitor_true_residual": None,
     "pc_type": "fieldsplit",
     "snes_divergence_tolerance": 1.0e10,
     "pc_fieldsplit_type": "schur",
     "pc_fieldsplit_schur_factorization_type": "full",
     "pc_fieldsplit_schur_precondition": "user",
     "fieldsplit_0_ksp_type": "richardson",
     "fieldsplit_0_ksp_max_it": 1,
     "fieldsplit_0_ksp_convergence_test": "skip",
     #"fieldsplit_0_ksp_norm_type": "unpreconditioned",
     #"fieldsplit_0_ksp_monitor_true_residual": None,
     "fieldsplit_0_pc_type": "mg",
     "fieldsplit_0_mg_levels_ksp_type": "gmres",
     "fieldsplit_0_mg_levels_ksp_richardson_scale": 1/2,
     #"fieldsplit_0_mg_levels_ksp_norm_type": "unpreconditioned",
     #"fieldsplit_0_mg_levels_ksp_monitor_true_residual": None,
     "fieldsplit_0_mg_levels_ksp_max_it": 3,
     "fieldsplit_0_mg_levels_ksp_convergence_test": "skip",
     "fieldsplit_0_mg_levels_pc_type": "pbjacobi",
     "fieldsplit_0_mg_coarse_pc_type": "python",
     "fieldsplit_0_mg_coarse_pc_python_type": "firedrake.AssembledPC",
     "fieldsplit_0_mg_coarse_assembled_pc_type": "cholesky",
     "fieldsplit_0_mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_0_mg_coarse_assembled_mat_mumps_icntl_14": "200",
     "fieldsplit_1_ksp_type": "preonly",
     "fieldsplit_1_pc_type": "python",
     "fieldsplit_1_pc_python_type": "__main__.Mass",
     "fieldsplit_1_aux_pc_type": "cholesky",
     "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_1_aux_mat_mumps_icntl_14": "200",
     }

fieldsplit_with_mg = {
     "mat_type":    "nest",
     "ksp_type":    "fgmres",
     "ksp_monitor_true_residual": None,
     "pc_type": "fieldsplit",
     "snes_divergence_tolerance": 1.0e10,
     "pc_fieldsplit_type": "schur",
     "pc_fieldsplit_schur_factorization_type": "full",
     "pc_fieldsplit_schur_precondition": "user",
     "fieldsplit_0_ksp_type": "richardson",
     "fieldsplit_0_ksp_max_it": 1,
     "fieldsplit_0_ksp_convergence_test": "skip",
     #"fieldsplit_0_ksp_norm_type": "unpreconditioned",
     #"fieldsplit_0_ksp_monitor_true_residual": None,
     "fieldsplit_0_pc_type": "mg",
     "fieldsplit_0_mg_levels_ksp_type": "gmres",
     "fieldsplit_0_mg_levels_ksp_richardson_scale": 1/2,
     #"fieldsplit_0_mg_levels_ksp_norm_type": "unpreconditioned",
     #"fieldsplit_0_mg_levels_ksp_monitor_true_residual": None,
     "fieldsplit_0_mg_levels_ksp_max_it": 3,
     "fieldsplit_0_mg_levels_ksp_convergence_test": "skip",
     "fieldsplit_0_mg_levels_pc_type": "python",
     "fieldsplit_0_mg_levels_pc_python_type": "firedrake.PatchPC",
     "fieldsplit_0_mg_levels_patch_pc_patch_save_operators": True,
     "fieldsplit_0_mg_levels_patch_pc_patch_partition_of_unity": True,
     "fieldsplit_0_mg_levels_patch_pc_patch_sub_mat_type": "dense",
     "fieldsplit_0_mg_levels_patch_pc_patch_dense_inverse": True,
     "fieldsplit_0_mg_levels_patch_pc_patch_local_type": "additive",
     "fieldsplit_0_mg_levels_patch_pc_patch_construct_type": "star",
     "fieldsplit_0_mg_levels_patch_pc_patch_statistics": False,
     "fieldsplit_0_mg_levels_patch_pc_patch_precompute_element_tensors": True,
     "fieldsplit_0_mg_levels_patch_sub_ksp_type": "preonly",
     "fieldsplit_0_mg_levels_patch_sub_pc_type": "lu",
     "fieldsplit_0_mg_levels_patch_sub_pc_factor_mat_solver_type": "petsc",
     "fieldsplit_0_mg_coarse_pc_type": "python",
     "fieldsplit_0_mg_coarse_pc_python_type": "firedrake.AssembledPC",
     "fieldsplit_0_mg_coarse_assembled_pc_type": "cholesky",
     "fieldsplit_0_mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_0_mg_coarse_assembled_mat_mumps_icntl_14": "200",
     "fieldsplit_1_ksp_type": "preonly",
     "fieldsplit_1_pc_type": "python",
     "fieldsplit_1_pc_python_type": "__main__.Mass",
     "fieldsplit_1_aux_pc_type": "cholesky",
     "fieldsplit_1_aux_pc_factor_mat_solver_type": "mumps",
     "fieldsplit_1_aux_mat_mumps_icntl_14": "200",
     }

faspardecomp = {
       "mat_type": "matfree",
       "snes_type": "ngmres",
       "snes_monitor": None,
       "snes_ngmres_monitor": None,
       "snes_max_it": 100,
       "snes_atol": 1.0e-9,
       "snes_rtol": 0.0,
       "snes_stol": 0.0,
       "snes_divergence_tolerance": 0.0,
       "snes_npc_side": "right",
       "npc_snes_type": "fas",
       "npc_snes_fas_cycles": 1,
       "npc_snes_fas_type": "full",
       "npc_snes_fas_galerkin": False,
       "npc_snes_fas_smoothup": 1,
       "npc_snes_fas_smoothdown": 1,
       "npc_snes_fas_full_downsweep": False,
       "npc_snes_monitor": None,
       "npc_snes_max_it": 1,
       "npc_snes_convergence_test": "skip",
       "npc_fas_levels_snes_type": "ngmres",
       "npc_fas_levels_snes_converged_reason": None,
       "npc_fas_levels_snes_atol": 1e-10,
       "npc_fas_levels_snes_rtol": 1e-10,
       "npc_fas_levels_snes_monitor": None,
       "npc_fas_levels_snes_ngmres_monitor": None,
       "npc_fas_levels_snes_ngmres_restart_type": "none",
       "npc_fas_levels_snes_npc_side": "right",
       "npc_fas_levels_npc_snes_type": "python",
       "npc_fas_levels_npc_snes_python_type": "firedrake.PatchSNES",
       "npc_fas_levels_npc_snes_max_it": 1,
       "npc_fas_levels_npc_snes_convergence_test": "skip",
       "npc_fas_levels_npc_snes_converged_reason": None,
       "npc_fas_levels_npc_snes_monitor": None,
       "npc_fas_levels_npc_snes_linesearch_type": "basic",
       "npc_fas_levels_npc_snes_linesearch_maxstep": 1.0,
       "npc_fas_levels_npc_snes_linesearch_damping": 1.0,
       "npc_fas_levels_npc_patch_snes_patch_partition_of_unity": True,
       "npc_fas_levels_npc_patch_snes_patch_construct_type": "pardecomp",
       "npc_fas_levels_npc_patch_snes_patch_pardecomp_overlap": 1,
       "npc_fas_levels_npc_patch_snes_patch_sub_mat_type": "seqaij",
       "npc_fas_levels_npc_patch_snes_patch_local_type": "additive",
       "npc_fas_levels_npc_patch_snes_patch_symmetrise_sweep": False,
       "npc_fas_levels_npc_patch_sub_snes_type": "newtonls",
       "npc_fas_levels_npc_patch_sub_snes_monitor": None,
       "npc_fas_levels_npc_patch_sub_snes_atol": 1e-9,
       "npc_fas_levels_npc_patch_sub_snes_rtol": 1e-9,
       "npc_fas_levels_npc_patch_sub_snes_converged_reason": None,
       "npc_fas_levels_npc_patch_sub_snes_linesearch_type": "basic",
       "npc_fas_levels_npc_patch_sub_ksp_type": "preonly",
       "npc_fas_levels_npc_patch_sub_pc_type": "cholesky",
       "npc_fas_levels_npc_patch_sub_pc_factor_mat_solver_type": "mumps",
       "npc_fas_coarse_snes_type": "newtonls",
       "npc_fas_coarse_snes_monitor": None,
       "npc_fas_coarse_snes_converged_reason": None,
       "npc_fas_coarse_snes_max_it": 100,
       "npc_fas_coarse_snes_atol": 1.0e-9,
       "npc_fas_coarse_snes_rtol": 1.0e-9,
       "npc_fas_coarse_snes_linesearch_monitor": None,
       "npc_fas_coarse_snes_linesearch_type": "l2",
       "npc_fas_coarse_snes_linesearch_maxstep": 1.0,
       "npc_fas_coarse_snes_linesearch_damping": 1.0,
       "npc_fas_coarse_ksp_type": "preonly",
       "npc_fas_coarse_pc_type": "python",
       "npc_fas_coarse_pc_python_type": "firedrake.AssembledPC",
       "npc_fas_coarse_assembled_mat_type": "aij",
       "npc_fas_coarse_assembled_pc_type": "cholesky",
       "npc_fas_coarse_assembled_pc_factor_mat_solver_type": "mumps",
       }
