#include "rbf.h"
#include <petscsys.h>
#include <petscts.h>
#include <petscksp.h>
#include <petscsnes.h>

typedef PetscReal(*UniRBF)(PetscReal);

struct _p_rbf_node {
  PetscInt       dim;
  PetscScalar    *loc;

  RBFType        type;
  PetscBool      is_parametric;
  UniRBF         uni_rbf;
  ParametricRBF  para_rbf;

  PetscBool      allocated_ctx;
  void           *para_ctx;
};

struct _p_rbf_problem {
  KDTree         tree;
  Vec            *node_points;
  RBFType        node_type;
  RBFProblemType problem_type;
  TS             ts;
  KSP            ksp;
  PC             pc;
};
  
  
struct _phs_ctx {
  PetscInt ord;
  PetscReal eps;
};

typedef struct _phs_ctx *phsctx;


PetscErrorCode RBFNodeCreate(RBFNode *node, PetscInt dims)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(node);CHKERRQ(ierr);
  if(dims < 1){
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "dims must be > 0, not %d.\n", dims);
  }
  (*node)->dim = dims;
  ierr = PetscCalloc1(dims, &((*node)->loc));CHKERRQ(ierr);
  (*node)->is_parametric = PETSC_FALSE;
  (*node)->allocated_ctx = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeDestroy(RBFNode node)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscFree(node->loc);CHKERRQ(ierr);
  if(node->allocated_ctx){
    ierr = PetscFree(node->para_ctx);CHKERRQ(ierr);
  }
  ierr = PetscFree(node);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeSetLocation(RBFNode node, const PetscScalar *location)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscArraycpy(node->loc, location, node->dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeGetLocation(const RBFNode node, PetscScalar *location)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscArraycpy(location, node->loc, node->dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define _sqr(r) (r) * (r)
static PetscReal gaussian_rbf(PetscReal r, void *ctx)
{
  PetscReal *eps = (PetscReal*)ctx;
  return PetscExpReal(-(_sqr((*eps) * r)));
}

static PetscReal multiquadric_rbf(PetscReal r, void *ctx)
{
  PetscReal *eps = (PetscReal*)ctx;
  return PetscSqrtReal(1.0 + _sqr((*eps) * r));
}

static PetscReal inverse_multiquadric_rbf(PetscReal r, void *ctx)
{
  PetscReal *eps = (PetscReal*)ctx;
  return 1.0/PetscSqrtReal(1.0 + _sqr((*eps) * r));
}

static PetscReal inverse_quadratic_rbf(PetscReal r, void *ctx)
{
  PetscReal *eps = (PetscReal*)ctx;
  return 1.0/(1.0+_sqr((*eps)*r));
}

static PetscReal phs_rbf(PetscReal r, void *ctx)
{
  phsctx pct = (phsctx)ctx;
  PetscReal eps = pct->eps;
  PetscInt ord = pct->ord;

  if(ord % 2 == 0){
    if(r < 1.0e-10){
      /* for stable computation*/
      return 0.0;
    }
    return PetscPowReal((eps * r), ord) * PetscLogReal(eps * r);
  } else {
    return PetscPowReal(eps * r, ord);
  }
}

static ParametricRBF _preset_rbfs[] = {gaussian_rbf, multiquadric_rbf,
				       inverse_multiquadric_rbf,
				       inverse_quadratic_rbf,
				       phs_rbf};

PetscErrorCode RBFNodeSetType(RBFNode node, RBFType type, void *ctx)
{
  PetscFunctionBeginUser;
  node->type = type;
  if(type == RBF_CUSTOM){
    if(ctx){
      node->is_parametric = PETSC_TRUE;
      node->para_ctx = ctx;
    }
  } else {
    node->is_parametric = PETSC_TRUE;
    node->para_rbf = _preset_rbfs[type];
    if(ctx){
      node->para_ctx = ctx;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeSetEpsilon(RBFNode node, PetscReal eps)
{
  PetscErrorCode ierr;
  PetscReal      *epsilon;
  PetscFunctionBeginUser;
  ierr = PetscCalloc1(1, &epsilon);CHKERRQ(ierr);
  *epsilon = eps;
  node->allocated_ctx = PETSC_TRUE;
  node->para_ctx = epsilon;
  PetscFunctionReturn(0);
}
  
  
  
PetscErrorCode RBFNodeSetPHS(RBFNode node, PetscInt ord, PetscReal eps)
{
  PetscErrorCode ierr;
  phsctx         ctx;
  PetscFunctionBeginUser;
  if(ord < 1){
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "PHS RBF order must be >= 1, not %d.\n", ord);
  }
  node->type = RBF_PHS;
  node->is_parametric = PETSC_TRUE;
  node->para_rbf = phs_rbf;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->ord = ord;
  ctx->eps = eps;
  node->para_ctx = ctx;
  node->allocated_ctx = PETSC_TRUE;
  PetscFunctionReturn(0);
}
  
PetscErrorCode RBFNodeSetParametricRBF(RBFNode node, ParametricRBF rbf, void *ctx)
{
  PetscFunctionBeginUser;
  node->type = RBF_CUSTOM;
  node->is_parametric = PETSC_TRUE;
  node->para_rbf = rbf;
  node->para_ctx = ctx;
  PetscFunctionReturn(0);
}


PetscErrorCode RBFNodeEvaluateAtDistance(const RBFNode node, PetscReal r, PetscReal *phi)
{
  PetscFunctionBeginUser;
  if(PetscLikely(node->is_parametric)){
    *phi = node->para_rbf(r, node->para_ctx);
  } else {
    *phi = node->uni_rbf(r);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeEvaluateAtPoint(const RBFNode node, PetscScalar *point, PetscReal *phi)
{
  PetscReal dist = 0.0;
  PetscInt  i;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  for(i=0; i<node->dim; ++i){
    dist += _sqr(point[i] - node->loc[i]);
  }
  ierr = RBFNodeEvaluateAtDistance(node, PetscSqrtReal(dist), phi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode RBFProblemCreate(RBFProblem *prob, PetscInt ndim)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(prob);CHKERRQ(ierr);
  ierr = KDTreeCreate(&((*prob)->tree), ndim);CHKERRQ(ierr);
  ierr = KDTreeSetNodeDestructor((*prob)->tree, (NodeDestructor)RBFNodeDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RBFProblemDestroy(RBFProblem prob)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = KDTreeDestroy(prob->tree);CHKERRQ(ierr);
  if(prob->ts){
    ierr = TSDestroy(&prob->ts);
  }
  ierr = PetscFree(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  

PetscErrorCode RBFProblemSetType(RBFProblem prob, RBFProblemType type)
{
  PetscFunctionBeginUser;
  prob->problem_type = type;
  PetscFunctionReturn(0);
}

PetscErrorCode RBFProblemSetNodeType(RBFProblem prob, RBFType type)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  prob->node_type = type;
  PetscFunctionReturn(0);
}
  

PetscErrorCode RBFProblemSetNodes(RBFProblem prob, Vec *locs, void *node_ctx)
{
  PetscErrorCode    ierr;
  PetscInt          i, j, k;
  const PetscScalar *x;
  PetscFunctionBeginUser;

  ierr = KDTreeGetK(prob->tree, &k);
  PetscInt sz[k];
  PetscScalar loc[k];
  for(i=0; i<k; ++i){
    ierr = VecGetLocalSize(locs[i], &sz[i]);CHKERRQ(ierr);
    if(sz[i] != sz[0]){
      SETERRQ3(PETSC_COMM_WORLD, 1, "RBF node location vectors must have the same number of components (%d), but dimension %d has %d components!\n", sz[0], i, sz[i]);
    }
  }
  /* ^ confirms that sz[i] are equal for all i */
  for(j=0; j<sz[0]; ++j){
    RBFNode node;
    for(i=0; i<k; ++i){
      ierr = VecGetArrayRead(locs[i], &x);CHKERRQ(ierr);
      loc[i] = x[j];
      ierr = VecRestoreArrayRead(locs[i], &x);CHKERRQ(ierr);
    }
    ierr = RBFNodeCreate(&node, k);CHKERRQ(ierr);
    ierr = RBFNodeSetType(node, prob->node_type, node_ctx);CHKERRQ(ierr);
    ierr = RBFNodeSetLocation(node, loc);CHKERRQ(ierr);
    
    ierr = KDTreeInsert(prob->tree, loc, node);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RBFProblemGetTree(RBFProblem prob, KDTree *tree)
{
  PetscFunctionBeginUser;
  *tree = prob->tree;
  PetscFunctionReturn(0);
}
    
  
