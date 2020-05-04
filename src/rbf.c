#include "rbf.h"
#include <petscsys.h>
#include <petscts.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <limits.h>
#include <petscfe.h>

typedef PetscReal(*UniRBF)(PetscReal);

struct _p_rbf_node {
  PetscInt       dim, poly_order;
  PetscScalar    *loc, *centered_loc;

  RBFType        type;
  PetscSpace     polyspace;
  PetscBool      is_parametric;
  UniRBF         uni_rbf;
  ParametricRBF  para_rbf;

  PetscBool      allocated_ctx;
  void           *para_ctx;
};

struct _p_rbf_problem {
  PetscInt       dims, global_poly_order;
  KDTree         tree;
  Vec            *node_points;
  RBFType        node_type;
  RBFProblemType problem_type;
  RBFNode        *nodes;
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
  ierr = PetscCalloc2(dims, &((*node)->loc), dims, &((*node)->centered_loc));CHKERRQ(ierr);
  ierr = PetscSpaceCreate(PETSC_COMM_WORLD, &((*node)->polyspace));CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables((*node)->polyspace, dims);CHKERRQ(ierr);
  ierr = PetscSpaceSetType((*node)->polyspace, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp((*node)->polyspace);CHKERRQ(ierr);
  (*node)->is_parametric = PETSC_FALSE;
  (*node)->allocated_ctx = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeDestroy(RBFNode node)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscSpaceDestroy(&(node->polyspace));CHKERRQ(ierr);
  ierr = PetscFree(node->loc);CHKERRQ(ierr);
  ierr = PetscFree(node->centered_loc);CHKERRQ(ierr);
  if(node->allocated_ctx){
    ierr = PetscFree(node->para_ctx);CHKERRQ(ierr);
  }
  ierr = PetscFree(node);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeSetPolyOrder(RBFNode node, PetscInt poly_order)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(!node){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "You must pass a non-NULL node (first argument) to RBFNodeSetPolyOrder!\n");
  }
  if(poly_order < 0){
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "You must pass a positive or zero polynomial order to RBFNodeSetPolyOrder, not %d!\n",poly_order);
  }
  node->poly_order = poly_order;
  ierr = PetscSpaceSetDegree(node->polyspace, poly_order, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(node->polyspace);CHKERRQ(ierr);
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


PetscErrorCode RBFNodeEvaluateRBFAtDistance(const RBFNode node, PetscReal r, PetscReal *phi)
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
  PetscReal *vals;
  PetscInt  i,N;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(!node){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "You must pass a non-NULL node (first argument) to RBFNodeEvaluateAtPoint!\n");
  }
  for(i=0; i<node->dim; ++i){
    dist += _sqr(point[i] - node->loc[i]);
  }
  /* get RBF part */
  ierr = RBFNodeEvaluateRBFAtDistance(node, PetscSqrtReal(dist), phi);CHKERRQ(ierr);
  /* get poly part */
  ierr = PetscSpaceGetDimension(node->polyspace, &N);CHKERRQ(ierr);
  ierr = PetscCalloc1(N, &vals);CHKERRQ(ierr);
  
  ierr = PetscSpaceEvaluate(node->polyspace, 1, point, vals, NULL, NULL);CHKERRQ(ierr);
  for(i=0; i<N; ++i){
    *phi += vals[i];
  }
  PetscFunctionReturn(0);
}


PetscErrorCode RBFProblemCreate(RBFProblem *prob, PetscInt ndim)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(prob);CHKERRQ(ierr);
  (*prob)->dims = ndim;
  (*prob)->global_poly_order=0;
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
  ierr = PetscCalloc1(sz[0], &prob->nodes);CHKERRQ(ierr);
  /* ^ confirms that sz[i] are equal for all i */
  for(j=0; j<sz[0]; ++j){
    for(i=0; i<k; ++i){
      RBFNode node;
      ierr = PetscNew(&node);CHKERRQ(ierr);
      ierr = VecGetArrayRead(locs[i], &x);CHKERRQ(ierr);
      loc[i] = x[j];
      ierr = VecRestoreArrayRead(locs[i], &x);CHKERRQ(ierr);
    }
    ierr = RBFNodeCreate(&(prob->nodes[j]), k);CHKERRQ(ierr);
    ierr = RBFNodeSetType(prob->nodes[j], prob->node_type, node_ctx);CHKERRQ(ierr);
    ierr = RBFNodeSetLocation(prob->nodes[j], loc);CHKERRQ(ierr);
    ierr = RBFNodeSetPolyOrder(prob->nodes[j], prob->global_poly_order);CHKERRQ(ierr);
    ierr = KDTreeInsert(prob->tree, loc, prob->nodes[j]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RBFProblemGetTree(RBFProblem prob, KDTree *tree)
{
  PetscFunctionBeginUser;
  *tree = prob->tree;
  PetscFunctionReturn(0);
}

PetscErrorCode RBFProblemSetPolynomialDegree(RBFProblem prob, PetscInt degree)
{
  PetscFunctionBeginUser;
  if(!prob){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "Cannot pass NULL first argument to RBFProblemSetPolynomialDegree!\n");
  }
  if(degree < 0){
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "You must pass a positive or zero polynomial degree to RBFNodeSetPolynomialDegree, not %d!\n",degree);
  }
  prob->global_poly_order = degree;
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeViewPolynomialBasis(const RBFNode node)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscSpaceView(node->polyspace, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*static PetscErrorCode
rbf_interp_get_node_weights(RBFProblem  prob,
			    PetscInt    stencil_size,
			    RBFNode     *stencil_nodes,
			    PetscScalar coeff,
			    const PetscScalar *target_point,
			    PetscScalar *fdweights)*/

static PetscErrorCode rbf_interp_get_weight_problem(RBFProblem  prob,
						    PetscInt    stencil_size,
						    RBFNode     *stencil_nodes,
						    PetscScalar coeff,
						    const PetscScalar *target_point,
						    Mat *AP,
						    Vec *L)
{
  PetscErrorCode ierr;
  PetscInt       i, j, k, nc;
  PetscReal      r2, dr, phi;
  ierr = PetscSpaceGetDimension(stencil_nodes[0]->polyspace,&nc);CHKERRQ(ierr);
  PetscScalar    *ll,lpl[nc], av[stencil_size][stencil_size], pvu[stencil_size][nc],pvl[nc][stencil_size], prow[nc];
  PetscInt       arowcol[stencil_size],pcol[nc];

  
  /* create right-hand-side Vec */
  ierr = VecCreateSeq(PETSC_COMM_SELF,L);CHKERRQ(ierr);
  ierr = VecSetSizes(*L,PETSC_DECIDE,stencil_size+nc);CHKERRQ(ierr);
  ierr = VecSet(*L,0.0);CHKERRQ(ierr);
  ierr = VecSetUp(*L);CHKERRQ(ierr);

  ierr = VecGetArray(*L, &ll);CHKERRQ(ierr);
  for(i=0; i<stencil_size; ++i){
    r2 = 0.0
    for(j=0; j<prob->dim; ++j){
      /* get locally-centered node locations and distance to this node*/
      dr = stencil_nodes[i]->centered_loc[j] = stencil_nodes[i]->loc[j] - target_point[j];
      r2 += _sqr(dr);
    }
    ierr = RBFNodeEvaluateRBFAtDistance(stencil_nodes[i],PetscSqrtReal(r2),&phi);CHKERRQ(ierr);
    ll[i] = coeff*phi;
  }
  /* since polynomial spaces are not locally-centered, it doesn't matter which node's polyspace we 
     evaluate here, they all give the exact same values at the same point in the domain */
  ierr = PetscSpaceEvaluate(stencil_nodes[0]->polyspace,1,target_point,lpl);CHKERRQ(ierr);
  for(i=stencil_size; i<stencil_size+nc; ++i){
    ll[i] = coeff*lpl[i-stencil_size];
  }
  
  ierr = VecRestoreArray(*L,&ll);CHKERRQ(ierr);

  /* create Mats for left-hand side */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,stencil_size+nc,stencil_size+nc,NULL,AP);CHKERRQ(ierr);
  ierr = MatSetOption(*AP,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetUp(*AP);CHKERRQ(ierr);

  /* RBF A matrix */
  for(i=0; i<stencil_size; ++i){
    arowcol[i] = i;
    for(j=0; j<stencil_size; ++j){
      r2=0.0;
      for(k=0; k<prob->dims; ++k){
	dr = stencil_nodes[i]->centered_loc[k] - stencil_nodes[j]->centered_loc[k];
	r2 += _sqr(dr);
      }
      ierr = RBFNodeEvaluateRBFAtDistance(stencil_nodes[i],PetscSqrtReal(r2),&av[i][j]);CHKERRQ(ierr);
    }
  }
  ierr = MatSetValues(*AP,stencil_size,arowcol,stencil_size,arowcol,av,INSERT_VALUES);CHKERRQ(ierr);
  /*ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);*.

  ierr = MatCreateSeqDense(PETSC_COMM_SELF,stencil_size,nc,NULL,&P);CHKERRQ(ierr);
  ierr = MatSetUp(P);CHKERRQ(ierr);

  /* polynomial component */
  for(i=0; i<stencil_size; ++i){
    ierr = PetscSpaceEvaluate(stencil_nodes[i],1,stencil_nodes[i]->centered_loc,prow);CHKERRQ(ierr);
    for(j=0; j<nc; ++j){
      pvu[i][j] = prow[j];
      pvl[j][i] = prow[j];
    }
  }

  for(i=0; i<nc; ++i){
    pcol[i] = stencil_size+i;
  }
  /* insert upper-right block */
  ierr = MatSetValues(*AP,stencil_size,arowcol,nc,pcol,pvu,INSERT_VALUES);CHKERRQ(ierr);
  for(i=0; i<stencil_size; ++i){
    arowcol[i] = i;
  }
  for(i=0; i<nc; ++i){
    pcol[i] = stencil_size+i;
  }
  /* insert lower-left block and assemble*/
  ierr = MatSetValues(*AP,nc,pcol,stencil_size,arowcol,pvl,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*AP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*AP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

