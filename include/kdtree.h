#ifndef RBF_KDTREE_H
#define RBF_KDTREE_H

#include <petscvec.h>

#ifdef __cplusplus
extern "C" {
#endif

  
  /*  struct _p_kdtree;
      struct _p_kd_values;*/
  
  typedef struct _p_kdtree *KDTree;
  typedef struct _p_kd_values *KDValues;

  PetscErrorCode KDTreeCreate(MPI_Comm comm, PetscInt k, KDTree *tree);

  PetscErrorCode KDTreeDestroy(KDTree *tree);

  PetscErrorCode KDTreeGetK(KDTree tree, PetscInt *k);

  PetscErrorCode KDTreeClear(KDTree tree);

  /* applies a unary function to each node */
  PetscErrorCode KDTreeApply(KDTree tree, void(*unaryfunc)(void*));

  PetscErrorCode KDTreeInsert(KDTree tree, const PetscScalar *loc, void *node_data);
  
  PetscErrorCode KDTreeInsert3D(KDTree tree, PetscScalar x, PetscScalar y, PetscScalar z, void *node_data);

  typedef PetscErrorCode (*NodeDestructor)(void **);

  PetscErrorCode KDTreeSetNodeDestructor(KDTree tree, NodeDestructor dtor);

  PetscErrorCode KDTreeFindNearest(const KDTree tree, const PetscScalar *loc, KDValues *nearest);

  PetscErrorCode KDTreeFindNearest3D(const KDTree tree, PetscScalar x, PetscScalar y, PetscScalar z, KDValues *nearest);

  PetscErrorCode KDTreeFindWithinRange(const KDTree tree, const PetscScalar *loc, PetscReal range, KDValues *nodes);

  PetscErrorCode KDTreeFindWithinRange3D(const KDTree tree, PetscScalar x, PetscScalar y, PetscScalar z, PetscReal range);

  PetscErrorCode KDTreeSize(const KDTree tree, PetscInt *N);

  PetscErrorCode KDValuesSize(const KDValues vals, PetscInt *n);

  /* set loc to NULL if you don't want to set the node's location */
  PetscErrorCode KDValuesGetNodeData(const KDValues vals, void **nodedata, const PetscScalar *loc); 

  PetscErrorCode KDValuesGetNodeDistance(const KDValues vals, PetscReal *dist);

  /* iterator functions for result values */
  PetscErrorCode KDValuesBegin(KDValues vals);

  PetscErrorCode KDValuesEnd(const KDValues vals);

  PetscErrorCode KDValuesNext(KDValues vals);

  PetscErrorCode KDValuesDestroy(KDValues vals);
  

#ifdef __cplusplus
}
#endif

#endif
