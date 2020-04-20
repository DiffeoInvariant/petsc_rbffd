/*
This file (author: Zane Jakobs) is based on code containing the following copyleft license:

This file is part of ``kdtree'', a library for working with kd-trees.
Copyright (C) 2007-2011 John Tsiombikas <nuclear@member.fsf.org>
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/

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

  PetscErrorCode KDTreeCreate(KDTree *tree, PetscInt k);

  PetscErrorCode KDTreeDestroy(KDTree tree);

  PetscErrorCode KDTreeClear(KDTree tree);

  PetscErrorCode KDTreeInsert(KDTree tree, const PetscScalar *loc, void *node_data);
  
  PetscErrorCode KDTreeInsert3D(KDTree tree, PetscScalar x, PetscScalar y, PetscScalar z, void *node_data);

  typedef PetscErrorCode (*NodeDestructor)(void *);

  PetscErrorCode KDTreeSetNodeDestructor(KDTree tree, NodeDestructor dtor);

  PetscErrorCode KDTreeFindNearest(const KDTree tree, const PetscScalar *loc, KDValues *nearest);

  PetscErrorCode KDTreeFindNearest3D(const KDTree tree, PetscScalar x, PetscScalar y, PetscScalar z, KDValues *nearest);

  PetscErrorCode KDTreeFindWithinRange(const KDTree tree, const PetscScalar *loc, PetscReal range, KDValues *nodes);

  PetscErrorCode KDTreeFindWithinRange3D(const KDTree tree, PetscScalar x, PetscScalar y, PetscScalar z, PetscReal range);

  PetscErrorCode KDValuesSize(const KDValues vals, PetscInt *n);

  /* set loc to NULL if you don't want to set the node's location */
  PetscErrorCode KDValuesGetNodeData(const KDValues vals, void *node, const PetscScalar *loc);

  /* iterator functions for result values */
  PetscErrorCode KDValuesBegin(KDValues vals);

  PetscErrorCode KDValuesEnd(const KDValues vals);

  PetscErrorCode KDValuesNext(KDValues vals);

  PetscErrorCode KDValuesDestroy(KDValues vals);
  

#ifdef __cplusplus
}
#endif

#endif
