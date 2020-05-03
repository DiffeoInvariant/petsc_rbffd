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

static long small_factorial(PetscInt N)
{
  long i, nf = 1;
  for(i=1; i<=N; ++i){
    nf *= i;
  }
  return nf;
}

static long long range_factorial(int start, int end)
{
  int i;
  long long rf=1;
  for(i=start; i<=end; ++i){
    rf *= i;
  }
  return rf;
}

static long long monomial_basis_cardinality(PetscInt dims, PetscInt order)
{
  PetscInt d = dims*order;
  long long topfact = range_factorial(dims, d+dims-1);
  long long bfact = (long long)small_factorial(d);
  if(topfact % bfact == 0){
    return topfact/bfact;
  } else {
    return topfact/bfact + 1;
  }
}
/*
static PetscErrorCode monomial_basis_fill_term(PetscInt *basis, PetscInt terms,
					       PetscInt dims, PetscInt order)
{
  PetscInt i, j;
  if(order > 8){
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_RANGE, "cannot create monomial basis with order %d! Maximum order is 8.");
  }
  switch(order){
  case 0:*/
  /* constant term *//*
    for(i=0; i<dims; ++i){
      basis[i] = 0;
    }
    break;
    case 1:*/
  /* linear terms *//*
    for(i=0; i<dims; ++i){
      for(j=0; j<dims; ++j){
	if(i == j){
	  basis[dims * (i+1) + j] = 1;
	} else {
	  basis[dims * (i+1) + j] = 0;
	}
      }
    }
    break;
  case 2:*/
    /* quadratic terms: xx, xy (x2),xz (x2),
                        0, yy, yz(x2),
			0, 0, zz
       ... (dims*(dims+1) elements)
       1,2,
       0,1, 
    */
       
    
/*
static PetscErrorCode MonomialCreate(Monomial *mono, PetscInt dims, PetscInt order)
{
  PetscErrorCode ierr;
  long long      terms;
  PetscInt       i;
  PetscFunctionBeginUser;
  ierr = PetscNew(mono);CHKERRQ(ierr);
  (*mono)->dims = dims;
  (*mono)->order = order;*/
 
  /*
  terms = monomial_basis_cardinality(dims, order);
  if(terms < INT_MIN || terms > INT_MAX){
     SETERRQ2(PETSC_COMM_WORLD, 1, "Error, MonomialCreate with dims=%d and order=%d has too large basis cardinality, try again with smaller dimensions and/or order.\n", dims, order);
  }
  (*mono)->terms = terms;
  ierr = PetscCalloc2(terms*dim, &((*mono)->basis), terms, &((*mono)->coeffs));CHKERRQ(ierr);
  for(i=0; i<terms; ++i){
    (*mono)->coeffs[i] = 1.0;
    }*/
  /*
  PetscFunctionReturn(0);
  }*/


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
  ierr = 
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


#if 0
static PetscErrorCode
rbf_problem_get_node_weights(RBFProblem prob,
			     PetscInt   stencil_size,
			     RBFNode    *stencil_nodes,
			     const PetscScalar *target_point,
			     PetscInt derivative_order,
			     PetscInt polynomial_order,
			     PetscScalar *fdweights)
{
  PetscErrorCode ierr;
  PetscInt       i, j;
 
  /* center stencil points */
  for(i=0; i<stencil_size; ++i){
    for(j=0; j<prob->dims; ++j){
      stencil_nodes[i]->centered_loc[j] = stencil_nodes[i]->loc[j] - target_point[j];
    }
  }

}
#endif
