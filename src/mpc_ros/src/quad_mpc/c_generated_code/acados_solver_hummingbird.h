/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_hummingbird_H_
#define ACADOS_SOLVER_hummingbird_H_

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define HUMMINGBIRD_NX     18
#define HUMMINGBIRD_NZ     0
#define HUMMINGBIRD_NU     4
#define HUMMINGBIRD_NP     0
#define HUMMINGBIRD_NBX    3
#define HUMMINGBIRD_NBX0   18
#define HUMMINGBIRD_NBU    4
#define HUMMINGBIRD_NSBX   0
#define HUMMINGBIRD_NSBU   0
#define HUMMINGBIRD_NSH    0
#define HUMMINGBIRD_NSG    0
#define HUMMINGBIRD_NSPHI  0
#define HUMMINGBIRD_NSHN   0
#define HUMMINGBIRD_NSGN   0
#define HUMMINGBIRD_NSPHIN 0
#define HUMMINGBIRD_NSBXN  0
#define HUMMINGBIRD_NS     0
#define HUMMINGBIRD_NSN    0
#define HUMMINGBIRD_NG     0
#define HUMMINGBIRD_NBXN   0
#define HUMMINGBIRD_NGN    0
#define HUMMINGBIRD_NY0    22
#define HUMMINGBIRD_NY     22
#define HUMMINGBIRD_NYN    18
#define HUMMINGBIRD_N      50
#define HUMMINGBIRD_NH     0
#define HUMMINGBIRD_NPHI   0
#define HUMMINGBIRD_NHN    0
#define HUMMINGBIRD_NPHIN  0
#define HUMMINGBIRD_NR     0

#ifdef __cplusplus
extern "C" {
#endif

// ** capsule for solver data **
typedef struct hummingbird_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */
    // dynamics

    external_function_param_casadi *forw_vde_casadi;
    external_function_param_casadi *expl_ode_fun;




    // cost






    // constraints




} hummingbird_solver_capsule;

hummingbird_solver_capsule * hummingbird_acados_create_capsule(void);
int hummingbird_acados_free_capsule(hummingbird_solver_capsule *capsule);

int hummingbird_acados_create(hummingbird_solver_capsule * capsule);
/**
 * Generic version of hummingbird_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
int hummingbird_acados_create_with_discretization(hummingbird_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
int hummingbird_acados_update_time_steps(hummingbird_solver_capsule * capsule, int N, double* new_time_steps);
int hummingbird_acados_update_params(hummingbird_solver_capsule * capsule, int stage, double *value, int np);
int hummingbird_acados_solve(hummingbird_solver_capsule * capsule);
int hummingbird_acados_free(hummingbird_solver_capsule * capsule);
void hummingbird_acados_print_stats(hummingbird_solver_capsule * capsule);

ocp_nlp_in *hummingbird_acados_get_nlp_in(hummingbird_solver_capsule * capsule);
ocp_nlp_out *hummingbird_acados_get_nlp_out(hummingbird_solver_capsule * capsule);
ocp_nlp_solver *hummingbird_acados_get_nlp_solver(hummingbird_solver_capsule * capsule);
ocp_nlp_config *hummingbird_acados_get_nlp_config(hummingbird_solver_capsule * capsule);
void *hummingbird_acados_get_nlp_opts(hummingbird_solver_capsule * capsule);
ocp_nlp_dims *hummingbird_acados_get_nlp_dims(hummingbird_solver_capsule * capsule);
ocp_nlp_plan *hummingbird_acados_get_nlp_plan(hummingbird_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_hummingbird_H_
