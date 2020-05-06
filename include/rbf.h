#ifndef RBF_RBF_H
#define RBF_RBF_H
#include "kdtree.h"
#include <petscdm.h>
#ifdef __cplusplus
extern "C" {
#endif

  typedef struct _p_rbf_node *RBFNode;

  typedef struct _p_rbf_problem *RBFProblem;

  typedef enum {
		RBF_GA=0,
		RBF_MQ=1,
		RFB_IMQ=2,
		RBF_IQ=3,
		RBF_PHS=4,
		RBF_CUSTOM=5
  } RBFType;

  typedef enum {
		RBF_INTERPOLATE,
		RBF_PDE_SOLVE, /* solve F(u, grad(u), div(u), ...) = 0 */
		RBF_PDE_STEP /* timestepping for a PDE */
  } RBFProblemType;
  
  /* dims is number of spatial dimensions */
  extern PetscErrorCode RBFNodeCreate(MPI_Comm comm, PetscInt dims, RBFNode *rbfnode);
  
  extern PetscErrorCode RBFNodeDestroy(RBFNode *node);
  
  extern PetscErrorCode RBFNodeSetLocation(RBFNode node, const PetscScalar *location);

  extern PetscErrorCode RBFNodeGetLocation(const RBFNode node, PetscScalar *location);

  extern PetscErrorCode RBFNodeSetType(RBFNode node, RBFType type, void *ctx);

  /* use this for GA, MQ, QU, PHS, and IQ, or if you have a custom RBF
   that takes only one real-valued parameter (other than r)*/
  extern PetscErrorCode RBFNodeSetEpsilon(RBFNode node, PetscReal eps);

  extern PetscErrorCode RBFNodeView(const RBFNode node);

  extern PetscErrorCode RBFNodeSetPolyOrder(RBFNode node, PetscInt poly_order);

  extern PetscErrorCode RBFNodeSetPHS(RBFNode node, PetscInt ord, PetscReal eps);

  extern PetscErrorCode RBFNodeViewPolynomialBasis(const RBFNode node);

  typedef PetscReal(*ParametricRBF)(PetscReal, void*);
  
  extern PetscErrorCode RBFNodeSetParametricRBF(RBFNode node, ParametricRBF rbf, void *ctx);

  extern PetscErrorCode RBFNodeEvaluateRBFAtDistance(const RBFNode node, PetscReal r, PetscReal *phi);

  /* while EvaluateRBFAtDistance only evaluates the RBF, this evaluates the polynomial too */
  extern PetscErrorCode RBFNodeEvaluateAtPoint(const RBFNode node, PetscScalar *point, PetscReal *phi);

  extern PetscErrorCode RBFProblemCreate(MPI_Comm comm, PetscInt ndim, RBFProblem *rbfprob);

  extern PetscErrorCode RBFProblemDestroy(RBFProblem *prob);

  extern PetscErrorCode RBFProblemSetType(RBFProblem prob, RBFProblemType type);

  extern PetscErrorCode RBFProblemSetNodeType(RBFProblem prob, RBFType type, void *ctx);

  extern PetscErrorCode RBFProblemSetPolynomialDegree(RBFProblem prob, PetscInt degree);

  /* locs is an array of Vecs, containing as many as there are spatial dimensions, and node_ctx is the context struct for the node (e.g. just a pointer to PetscReal for GA, etc) */
  extern PetscErrorCode RBFProblemSetNodes(RBFProblem prob, Vec *locs, void *node_ctx);

  extern PetscErrorCode RBFProblemGetTree(RBFProblem prob, KDTree *tree);

  extern PetscErrorCode RBFProblemGetWeights(RBFProblem prob);

  extern PetscErrorCode RBFProblemView(RBFProblem prob);
  

#ifdef __cplusplus
}
#endif

#endif /* RBF_RBF_H */
