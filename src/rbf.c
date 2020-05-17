#include "rbf.h"
#include <petscsys.h>
#include <petscts.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <limits.h>
#include <petscfe.h>
#include <petsc/private/petscimpl.h>

typedef PetscReal(*UniRBF)(PetscReal, PetscReal);

struct _p_rbf_node_ops {
  UniRBF         uni_rbf;
  ParametricRBF  para_rbf;
  PetscBool      is_parametric;
  PetscBool      allocated_ctx;
  PetscReal      eps;/* use para_ctx instead of eps if you
			need more than one parameter */
  void           *para_ctx;
  PetscSpace     polyspace;
};

typedef struct _p_rbf_node_ops *RBFNodeOps;


struct _p_rbf_node {
  PETSCHEADER(struct _p_rbf_node_ops);
  PetscInt       dims, poly_order;
  PetscScalar    *loc, *centered_loc, val;
  RBFType        type;
};


struct _p_rbf_prob_ops {
  PetscErrorCode   (*interp_local_operator)(RBFProblem,PetscInt,RBFNode *,PetscScalar,const PetscScalar *,PetscInt);
  PetscErrorCode   (*interp_eval)(RBFProblem,PetscInt,RBFNode *,PetscScalar,const PetscScalar *,PetscInt,PetscScalar *);
};

static PetscErrorCode rbf_interp_get_matrix(RBFProblem  prob,
					    PetscInt    stencil_size,
					    RBFNode     *stencil_nodes,
					    PetscScalar coeff,
					    const PetscScalar *target_point,
					    PetscInt global_op_id);

static PetscErrorCode rbf_interp_evaluate_interpolant(RBFProblem prob,
						      PetscInt stencil_size,
						      RBFNode *stencil_nodes,
						      PetscScalar coeff,
						      const PetscScalar *target_pt,
						      PetscInt global_weight_id,
						      PetscScalar *val);


struct _p_rbf_problem {
  PETSCHEADER(struct _p_rbf_prob_ops);
  PetscInt       dims,global_poly_order,N;
  KDTree         tree;
  Vec            *node_points,*weights;
  RBFType        node_type;
  RBFProblemType problem_type;
  RBFNode        *nodes;
  PetscBool      setup;
  Mat            *A;
  TS             ts;
  KSP            ksp;
  PC             pc;
};
  
  
struct _phs_ctx {
  PetscInt ord;
  PetscReal eps;
};

typedef struct _phs_ctx *phsctx;


static PetscErrorCode rbf_node_check_args(PetscInt dims)
{
  if(dims < 1){
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "dims must be > 0, not %d.\n", dims);
  }
  return 0;
}

static PetscClassId RBF_NODE_CLASSID, RBF_PROBLEM_CLASSID;

static PetscErrorCode RBFNodeOpsSetSpace(RBFNode node, PetscInt dim)
{
  PetscErrorCode ierr;
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)node), &node->ops->polyspace);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(node->ops->polyspace, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(node->ops->polyspace, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  /*ierr = PetscSpaceSetNumComponents((*node)->polyspace, dims);CHKERRQ(ierr);*/
  ierr = PetscSpaceSetUp(node->ops->polyspace);CHKERRQ(ierr);
  return 0;
}

static const char *const RBF_type_names[] = {"GA","MQ","IMQ","IQ","PHS","CUSTOM","RBF_",0};

PetscErrorCode RBFNodeCreate(MPI_Comm comm, PetscInt dims, RBFNode *rbfnode)
{
  PetscErrorCode   ierr;
  static PetscBool registered_type = PETSC_FALSE;
  RBFType          type=RBF_GA;
  PetscReal        eps=1.0e-8;
  PetscInt         poly_order=1;
  RBFNode          node;
  PetscFunctionBeginUser;
  if (!registered_type) {
    PetscClassIdRegister("Radial Basis Function Node",&RBF_NODE_CLASSID);
    registered_type = PETSC_TRUE;
  }
  ierr = rbf_node_check_args(dims);CHKERRQ(ierr);
  PetscHeaderCreate(node,RBF_NODE_CLASSID,"RBFNode","Radial Basis Function Node","",comm,RBFNodeDestroy,RBFNodeView);
  
  node->dims = dims;
  PetscOptionsBegin(comm,NULL,"RBF Node options","");
  {
    PetscOptionsEnum("-rbf_type","Type of pre-set RBF to use","",RBF_type_names,(PetscEnum)type, (PetscEnum*)&type,NULL);
    node->type = type;
    
    PetscOptionsReal("-eps","Value of epsilon (for pre-set RBF types that have an epsilon, such as RBF_GA)","",eps,&eps,NULL);
    node->ops->eps = eps;

    PetscOptionsInt("-poly_order","Order of the pseudospectral polynomials","",poly_order,&poly_order,NULL);
    node->poly_order = poly_order;
  }
  PetscOptionsEnd();
      
  ierr = PetscCalloc2(dims, &node->loc, dims, &node->centered_loc);CHKERRQ(ierr);
  ierr = RBFNodeOpsSetSpace(node, dims);
  
  *rbfnode = node;
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeDestroy(RBFNode *node)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(!*node) PetscFunctionReturn(0);
  if(--((PetscObject)(*node))->refct > 0) { *node=NULL; PetscFunctionReturn(0);}
  
  ierr = PetscSpaceDestroy(&((*node)->ops->polyspace));CHKERRQ(ierr);
  ierr = PetscFree((*node)->loc);CHKERRQ(ierr);
  ierr = PetscFree((*node)->centered_loc);CHKERRQ(ierr);
  if((*node)->ops->allocated_ctx){
    ierr = PetscFree((*node)->ops->para_ctx);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(node);CHKERRQ(ierr);
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
  ierr = PetscSpaceSetDegree(node->ops->polyspace, poly_order, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(node->ops->polyspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

  
PetscErrorCode RBFNodeSetLocation(RBFNode node, const PetscScalar *location)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscArraycpy(node->loc, location, node->dims);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeGetLocation(const RBFNode node, PetscScalar *location)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscArraycpy(location, node->loc, node->dims);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define _sqr(r) (r) * (r)
static PetscReal gaussian_rbf(PetscReal r, PetscReal eps)
{
  return PetscExpReal(-(_sqr(eps * r)));
}

static PetscReal multiquadric_rbf(PetscReal r, PetscReal eps)
{
  return PetscSqrtReal(1.0 + _sqr(eps * r));
}

static PetscReal inverse_multiquadric_rbf(PetscReal r, PetscReal eps)
{
  return 1.0/PetscSqrtReal(1.0 + _sqr(eps * r));
}

static PetscReal inverse_quadratic_rbf(PetscReal r, PetscReal eps)
{
  return 1.0/(1.0+_sqr(eps*r));
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

static const UniRBF _preset_uni_rbfs[] = {gaussian_rbf, multiquadric_rbf,
					  inverse_multiquadric_rbf,
					  inverse_quadratic_rbf};

PetscErrorCode RBFNodeSetType(RBFNode node, RBFType type, void *ctx)
{
  PetscFunctionBeginUser;
  node->type = type;
  if(type == RBF_CUSTOM){
    if(ctx){
      node->ops->is_parametric = PETSC_TRUE;
      node->ops->para_ctx = ctx;
    }
  } else {
    if(type == RBF_PHS){
      node->ops->is_parametric = PETSC_TRUE;
      node->ops->para_rbf = phs_rbf;
      if(ctx){
	node->ops->para_ctx = ctx;
      }
    } else {
      node->ops->is_parametric = PETSC_FALSE;
      node->ops->uni_rbf = _preset_uni_rbfs[type];
      if(ctx){
	node->ops->eps = *((PetscReal*)ctx);
      }
    }
  }
  
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeSetEpsilon(RBFNode node, PetscReal eps)
{
  PetscFunctionBeginUser;
  if(eps <= 0.0){
    SETERRQ1(PetscObjectComm((PetscObject)node),PETSC_ERR_ARG_OUTOFRANGE,"Epsilon must be > 0, not %f!",eps);
  }
  node->ops->eps = eps;
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
  node->ops->is_parametric = PETSC_TRUE;
  node->ops->para_rbf = phs_rbf;
  ierr = PetscNew(&ctx);CHKERRQ(ierr);
  ctx->ord = ord;
  ctx->eps = eps;
  node->ops->para_ctx = ctx;
  node->ops->allocated_ctx = PETSC_TRUE;
  PetscFunctionReturn(0);
}
  
PetscErrorCode RBFNodeSetParametricRBF(RBFNode node, ParametricRBF rbf, void *ctx)
{
  PetscFunctionBeginUser;
  node->type = RBF_CUSTOM;
  node->ops->is_parametric = PETSC_TRUE;
  node->ops->para_rbf = rbf;
  node->ops->para_ctx = ctx;
  PetscFunctionReturn(0);
}


PetscErrorCode RBFNodeEvaluateRBFAtDistance(const RBFNode node, PetscReal r, PetscReal *phi)
{
  PetscFunctionBeginUser;
  if(PetscUnlikely(node->ops->is_parametric)){
    *phi = node->ops->para_rbf(r, node->ops->para_ctx);
  } else {
    *phi = node->ops->uni_rbf(r, node->ops->eps);
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
  for(i=0; i<node->dims; ++i){
    dist += _sqr(point[i] - node->loc[i]);
  }
  /* get RBF part */
  ierr = RBFNodeEvaluateRBFAtDistance(node, PetscSqrtReal(dist), phi);CHKERRQ(ierr);
  /* get poly part */
  ierr = PetscSpaceGetDimension(node->ops->polyspace, &N);CHKERRQ(ierr);
  ierr = PetscCalloc1(N, &vals);CHKERRQ(ierr);
  
  ierr = PetscSpaceEvaluate(node->ops->polyspace, 1, point, vals, NULL, NULL);CHKERRQ(ierr);
  for(i=0; i<N; ++i){
    *phi += vals[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeViewPolynomialBasis(const RBFNode node)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscSpaceView(node->ops->polyspace, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



PetscErrorCode RBFNodeView(const RBFNode node)
{
  PetscInt i;
  PetscFunctionBeginUser;
  PetscPrintf(PetscObjectComm((PetscObject)node),"RBFNode in %d spatial dimensions with RBF type %s, polynomial order %d, at location\n",node->dims,RBF_type_names[node->type],node->poly_order);
  PetscPrintf(PetscObjectComm((PetscObject)node),"[");
  for(i=0; i<node->dims-1; ++i){
    PetscPrintf(PetscObjectComm((PetscObject)node),"%5.4e,",node->loc[i]);
  }
  PetscPrintf(PetscObjectComm((PetscObject)node),"%5.4e].\n",node->loc[node->dims-1]);
  PetscFunctionReturn(0);
}

PetscErrorCode RBFNodeTreeDtor(void **node)
{
  PetscErrorCode ierr;
  RBFNode *rnode = (RBFNode*)node;
  ierr = RBFNodeDestroy(rnode);CHKERRQ(ierr);
  return 0;
}

static const char *const RBF_problem_types[] = {"INTERPOLATE","PDE_SOLVE","PDE_STEP","RBF_",0};

PetscErrorCode RBFProblemCreate(MPI_Comm comm, PetscInt dims, RBFProblem *rbfprob)
{
  PetscErrorCode   ierr;
  static PetscBool registered_type = PETSC_FALSE;
  RBFProblem       prob;
  RBFProblemType   ptype=RBF_INTERPOLATE;
  RBFType          ntype=RBF_GA;
  /*PetscReal        eps=1.0e-8;*/
  PetscInt         poly_order=1;
  PetscFunctionBeginUser;
  if(!registered_type){
    PetscClassIdRegister("Radial Basis Function problem manager and solver",&RBF_PROBLEM_CLASSID);
    registered_type = PETSC_TRUE;
  }
  if(dims < 1){
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "dims must be > 0, not %d.\n", dims);
  }
  
  PetscHeaderCreate(prob,RBF_PROBLEM_CLASSID,"RBFProblem","Radial Basis Function Problem Solver","",comm,RBFProblemDestroy,RBFProblemView);
  
  prob->dims = dims;
  prob->global_poly_order=poly_order;
  prob->N = 0;

  prob->ops->interp_local_operator = rbf_interp_get_matrix;
  prob->ops->interp_eval = rbf_interp_evaluate_interpolant;
  PetscOptionsBegin(comm,NULL,"RBF Problem options","");
  {
    PetscOptionsEnum("-rbf_type","Type of pre-set RBF to use","",RBF_type_names,(PetscEnum)ntype, (PetscEnum*)&ntype,NULL);
    prob->node_type = ntype;

    PetscOptionsEnum("-rbf_problem","What problem to solve? (Options are interpolation, steady-state PDE solving, and PDE timestepping)","",RBF_problem_types,(PetscEnum)ptype, (PetscEnum*)&ptype,NULL);
    prob->problem_type = ptype;

    PetscOptionsInt("-poly_order","Order of the pseudospectral polynomials","",poly_order,&poly_order,NULL);
    prob->global_poly_order = poly_order;
  }
  PetscOptionsEnd();

  ierr = KDTreeCreate(comm,dims,&prob->tree);CHKERRQ(ierr);
  ierr = KDTreeSetNodeDestructor(prob->tree, (NodeDestructor)RBFNodeTreeDtor);CHKERRQ(ierr);
  prob->setup = PETSC_TRUE;
  
  *rbfprob = prob;
  PetscFunctionReturn(0);
}

PetscErrorCode RBFProblemDestroy(RBFProblem *prob)
{
  PetscErrorCode ierr;
  PetscInt i;
  PetscFunctionBeginUser;
  if (!*prob) PetscFunctionReturn(0);
  if (--((PetscObject)(*prob))->refct > 0) { *prob=NULL; PetscFunctionReturn(0);}

  if ((*prob)->A){
    for (i=0; i<(*prob)->N; ++i) {
      ierr = MatDestroy(&((*prob)->A[i]));CHKERRQ(ierr);
      if ((*prob)->weights) {
	ierr = VecDestroy(&((*prob)->weights[i]));CHKERRQ(ierr);
      }
    }
  } 
  ierr = PetscFree((*prob)->A);
  ierr = PetscFree((*prob)->weights);
  if ((*prob)->node_points) {
    for (i=0; i<(*prob)->dims; ++i) {
      ierr = VecDestroy(&((*prob)->node_points[i]));CHKERRQ(ierr);
    }
  }
  ierr = PetscFree((*prob)->node_points);CHKERRQ(ierr);
  
  /* NOTE: KDTreeDestroy calls RBFNodeDestroy on all of the allocated nodes */
  ierr = KDTreeDestroy(&((*prob)->tree));CHKERRQ(ierr);
  if ((*prob)->nodes) {
    ierr = PetscFree((*prob)->nodes);CHKERRQ(ierr);
  }
  ierr = TSDestroy(&((*prob)->ts));CHKERRQ(ierr);
  ierr = KSPDestroy(&((*prob)->ksp));CHKERRQ(ierr);
  ierr = PCDestroy(&((*prob)->pc));CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(prob);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  

PetscErrorCode RBFProblemSetType(RBFProblem prob, RBFProblemType type)
{
  PetscFunctionBeginUser;
  prob->problem_type = type;
  PetscFunctionReturn(0);
}

static RBFType _set_nodes_to_t;
static void    *_set_node_ctx;

static void set_node_type(void* node)
{
  RBFNode rnode = (RBFNode)node;
  RBFNodeSetType(rnode,_set_nodes_to_t,_set_node_ctx);
}

PetscErrorCode RBFProblemSetNodeType(RBFProblem prob, RBFType type, void *ctx)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  prob->node_type = type;
  _set_nodes_to_t = type;
  _set_node_ctx = ctx;
  ierr = KDTreeApply(prob->tree,set_node_type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  

PetscErrorCode RBFProblemSetNodes(RBFProblem prob, Vec *locs, void *node_ctx)
{
  PetscErrorCode    ierr;
  PetscInt          i, j, k;
  const PetscScalar *x;
  PetscFunctionBeginUser;

  ierr = KDTreeGetK(prob->tree,&k);
  PetscInt sz[k];
  PetscScalar loc[k];
  for (i=0; i<k; ++i) {
    ierr = VecGetLocalSize(locs[i],&sz[i]);CHKERRQ(ierr);
    if (sz[i] != sz[0]) {
      SETERRQ3(PetscObjectComm((PetscObject)prob), 1, "RBF node location vectors must have the same number of components (%d), but dimension %d has %d components!\n", sz[0], i, sz[i]);
    }
  }
  prob->N = sz[0];
  ierr = PetscCalloc4(sz[0],&prob->nodes,sz[0],&prob->A,sz[0],&prob->weights,k,&prob->node_points);CHKERRQ(ierr);
  /* ^ confirms that sz[i] are equal for all i */
  for (i=0; i<k; ++i) {
    ierr = VecDuplicate(locs[i],&prob->node_points[i]);CHKERRQ(ierr);
    ierr = VecCopy(locs[i],prob->node_points[i]);CHKERRQ(ierr);
  }
    
  for(j=0; j<sz[0]; ++j){
    for(i=0; i<k; ++i){
      ierr = VecGetArrayRead(locs[i],&x);CHKERRQ(ierr);
      loc[i] = x[j];
      ierr = VecRestoreArrayRead(locs[i],&x);CHKERRQ(ierr);
    }
    ierr = RBFNodeCreate(PetscObjectComm((PetscObject)prob),k,&(prob->nodes[j]));CHKERRQ(ierr);
    ierr = RBFNodeSetType(prob->nodes[j],prob->node_type,node_ctx);CHKERRQ(ierr);
    ierr = RBFNodeSetLocation(prob->nodes[j],loc);CHKERRQ(ierr);
    ierr = RBFNodeSetPolyOrder(prob->nodes[j],prob->global_poly_order);CHKERRQ(ierr);
    ierr = KDTreeInsert(prob->tree,loc,prob->nodes[j]);CHKERRQ(ierr);
  }
  prob->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode RBFProblemGetTree(RBFProblem prob, KDTree *tree)
{
  PetscFunctionBeginUser;
  *tree = prob->tree;
  PetscFunctionReturn(0);
}

static PetscInt _set_node_order;

static void set_node_degree(void* node)
{
  RBFNode rnode = (RBFNode)node;
  RBFNodeSetPolyOrder(rnode,_set_node_order);
}


PetscErrorCode RBFProblemSetPolynomialDegree(RBFProblem prob, PetscInt degree)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(!prob){
    SETERRQ(PetscObjectComm((PetscObject)prob),PETSC_ERR_ARG_NULL,"Cannot pass NULL first argument to RBFProblemSetPolynomialDegree!\n");
  }
  if(degree < 0){
    SETERRQ1(PetscObjectComm((PetscObject)prob),PETSC_ERR_ARG_OUTOFRANGE,"You must pass a positive or zero polynomial degree to RBFNodeSetPolynomialDegree, not %d!\n",degree);
  }
  prob->global_poly_order = degree;
  _set_node_order = degree;
  ierr = KDTreeApply(prob->tree,set_node_degree);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode RBFProblemView(const RBFProblem prob)
{
  PetscInt       N;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = KDTreeSize(prob->tree,&N);CHKERRQ(ierr);
  PetscPrintf(PetscObjectComm((PetscObject)prob),"RBF Problem with task RBF_%s in %d spatial dimensions with %D nodes.\n",RBF_problem_types[prob->problem_type],prob->dims,N);
  PetscFunctionReturn(0);
}


/*static PetscErrorCode
rbf_interp_get_node_weights(RBFProblem  prob,
			    PetscInt    stencil_size,
			    RBFNode     *stencil_nodes,
			    PetscScalar coeff,
			    const PetscScalar *target_point,
			    PetscScalar *fdweights)*/

PetscErrorCode rbf_interp_get_matrix(RBFProblem  prob,
				     PetscInt    stencil_size,
				     RBFNode     *stencil_nodes,
				     PetscScalar coeff,
				     const PetscScalar *target_point,
				     PetscInt global_op_id)
{
  PetscErrorCode ierr;
  PetscInt       i, j, k, nc=0;
  PetscReal      r2, dr, phi;
  if (prob->global_poly_order > 0) {
    ierr = PetscSpaceGetDimension(stencil_nodes[0]->ops->polyspace,&nc);CHKERRQ(ierr);
  }
  PetscScalar    av[stencil_size][stencil_size], pvu[stencil_size][nc],pvl[nc][stencil_size], prow[nc];
  PetscInt       arowcol[stencil_size],pcol[nc];

  if (global_op_id >= prob->N) {
    SETERRQ2(PetscObjectComm((PetscObject)prob),PETSC_ERR_ARG_OUTOFRANGE,"RBFProblem only has %D matrices available to fill, but you requested we fill matrix number %D!",prob->N,global_op_id);
  }

  /*PetscPrintf(PETSC_COMM_WORLD,"stencil_size: %d, nc: %d.\n", stencil_size, nc);*/
  /* create Mats for left-hand side */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,stencil_size+nc,stencil_size+nc,NULL,&prob->A[global_op_id]);CHKERRQ(ierr);
  ierr = MatSetOption(prob->A[global_op_id],MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetUp(prob->A[global_op_id]);CHKERRQ(ierr);

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
  ierr = MatSetValues(prob->A[global_op_id],stencil_size,arowcol,stencil_size,arowcol,&av[0][0],INSERT_VALUES);CHKERRQ(ierr);

  /* polynomial component */
  if (nc > 0) {
    for(i=0; i<stencil_size; ++i){
      ierr = PetscSpaceEvaluate(stencil_nodes[i]->ops->polyspace,1,stencil_nodes[i]->centered_loc,prow,NULL,NULL);CHKERRQ(ierr);
      for(j=0; j<nc; ++j){
	pvu[i][j] = PetscIsNanReal(prow[j]) ? 0.0 : prow[j];
	pvl[j][i] = pvu[i][j];
      }
    }

    for (i=0; i<nc; ++i){
      pcol[i] = stencil_size+i;
    }
    /* insert upper-right block */
    ierr = MatSetValues(prob->A[global_op_id],stencil_size,arowcol,nc,pcol,&pvu[0][0],INSERT_VALUES);CHKERRQ(ierr);
    for(i=0; i<stencil_size; ++i){
      arowcol[i] = i;
    }
    for(i=0; i<nc; ++i){
      pcol[i] = stencil_size+i;
    }
    /* insert lower-left block and assemble*/
    ierr = MatSetValues(prob->A[global_op_id],nc,pcol,stencil_size,arowcol,&pvl[0][0],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(prob->A[global_op_id], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(prob->A[global_op_id], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode rbf_interp_evaluate_interpolant(RBFProblem prob,
						      PetscInt stencil_size,
						      RBFNode *stencil_nodes,
						      PetscScalar coeff,
						      const PetscScalar *target_point,
						      PetscInt global_weight_id,
						      PetscScalar *val)
{
  PetscReal      r2,dr,phi;
  PetscInt       i,j,nc=0;
  PetscScalar    *ll,lpl[nc];
  Vec            L;
  PetscErrorCode ierr;
  if (prob->global_poly_order > 0) {
    ierr = PetscSpaceGetDimension(stencil_nodes[0]->ops->polyspace,&nc);CHKERRQ(ierr);
  }
  

  if (global_weight_id >= prob->N) {
    SETERRQ2(PetscObjectComm((PetscObject)prob),PETSC_ERR_ARG_OUTOFRANGE,"RBFProblem only has %D weight vectors available to fill, but you requested we fill number %D!",prob->N,global_weight_id);
  }
  /* evaluate basis functions */
  ierr = VecCreateSeq(PETSC_COMM_SELF,stencil_size+nc,&L);CHKERRQ(ierr);
  ierr = VecSet(L,0.0);CHKERRQ(ierr);
  ierr = VecSetUp(L);CHKERRQ(ierr);

  ierr = VecGetArray(L, &ll);CHKERRQ(ierr);
  for(i=0; i<stencil_size; ++i){
    r2 = 0.0;
    for(j=0; j<prob->dims; ++j){
      /* get locally-centered node locations and distance to this node*/
      dr = stencil_nodes[i]->centered_loc[j] = stencil_nodes[i]->loc[j] - target_point[j];
      r2 += _sqr(dr);
    }
    ierr = RBFNodeEvaluateRBFAtDistance(stencil_nodes[i],PetscSqrtReal(r2),&phi);CHKERRQ(ierr);
    ll[i] = coeff*phi;
  }
  /* since polynomial spaces are not locally-centered, it doesn't matter which node's polyspace we 
     evaluate here, they all give the exact same values at the same point in the domain */
  if (nc > 0) {
    ierr = PetscSpaceEvaluate(stencil_nodes[0]->ops->polyspace,1,target_point,lpl,NULL,NULL);CHKERRQ(ierr);
    for(i=stencil_size; i<stencil_size+nc; ++i){
      ll[i] = PetscIsNanReal(lpl[i-stencil_size]) ? 0.0 : coeff*lpl[i-stencil_size];
    }
  }
  
  ierr = VecRestoreArray(L,&ll);CHKERRQ(ierr);

  /*basis functions dot weights */
  ierr = VecDot(L,prob->weights[global_weight_id],val);CHKERRQ(ierr);
  ierr = VecDestroy(&L);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
