#ifndef RBF_RBF_H
#define RBF_RBF_H
#include "kdtree.h"
#ifdef __cplusplus
extern "C" {
#endif

  typedef struct _p_rbf_node *RBFNode;

  typedef enum {
		RBF_GA=0,
		RBF_MQ=1,
		RFB_IMQ=2,
		RBF_IQ=3,
		RBF_PHS=4,
		RBF_CUSTOM=5
  } RBFType;
  
  /* dims is number of spatial dimensions */
  PetscErrorCode RBFNodeCreate(RBFNode *node, PetscInt dims);
  
  PetscErrorCode RBFNodeDestroy(RBFNode node);
  
  PetscErrorCode RBFNodeSetLocation(RBFNode node, const PetscScalar *location);

  PetscErrorCode RBFNodeGetLocation(const RBFNode node, PetscScalar *location);

  /* pass the RBF context to ctx if and only if you're setting RBF_CUSTOM and you want a custom context, else pass NULL */
  PetscErrorCode RBFNodeSetType(RBFNode node, RBFType type, void *ctx);

  /* use this for GA, MQ, QU, PHS, and IQ, or if you have a custom RBF
   that takes only one real-valued parameter (other than r)*/
  PetscErrorCode RBFNodeSetEpsilon(RBFNode node, PetscReal eps);

  PetscErrorCode RBFNodeSetPHS(RBFNode node, PetscInt ord, PetscReal eps);

  typedef PetscReal(*ParametricRBF)(PetscReal, void*);
  
  PetscErrorCode RBFNodeSetParametricRBF(RBFNode node, ParametricRBF rbf, void *ctx);

  PetscErrorCode RBFNodeEvaluateAtDistance(const RBFNode node, PetscReal r, PetscReal *phi);

  PetscErrorCode RBFNodeEvaluateAtPoint(const RBFNode node, PetscScalar *point, PetscReal *phi);
  
  


#ifdef __cplusplus
}
#endif

#endif /* RBF_RBF_H */
