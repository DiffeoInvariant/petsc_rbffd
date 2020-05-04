#include "rbf.h"
#include <stdlib.h>

int main(int argc, char **argv)
{

  PetscInt i,j, k=3, N=10000;
  PetscReal d, eps=0.1;
  Vec       X[3];
  PetscScalar val, *x, loc[3], eloc[3];
  RBFProblem prob;
  RBFNode    node, *nodes;
  KDTree     tree;
  KDValues   nns;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  /* create node locations */
  for(i=0; i<k; ++i){
    ierr = VecCreate(PETSC_COMM_WORLD, &X[i]);CHKERRQ(ierr);
    ierr = VecSetFromOptions(X[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(X[i], PETSC_DECIDE, N);CHKERRQ(ierr);
    ierr = VecGetArray(X[i], &x);CHKERRQ(ierr);
    for(j=0; j<N; ++j){
      x[j] = 10*((PetscReal)((rand() / ((RAND_MAX + 1u)/30))) - 15.0);
    }
    ierr = VecRestoreArray(X[i], &x);CHKERRQ(ierr);
  }

  /* create problem data structures */
  ierr = RBFProblemCreate(&prob, k);CHKERRQ(ierr);

  ierr = RBFProblemSetPolynomialDegree(prob, 3);CHKERRQ(ierr);
  ierr = RBFProblemSetType(prob, RBF_INTERPOLATE);CHKERRQ(ierr);
  ierr = RBFProblemSetNodeType(prob, RBF_GA);CHKERRQ(ierr);
  ierr = RBFProblemSetNodes(prob, X, &eps);CHKERRQ(ierr);

  for(i=0; i<k; ++i){
    ierr = VecDestroy(&X[i]);CHKERRQ(ierr);
  }
  ierr = RBFProblemGetTree(prob, &tree);CHKERRQ(ierr);
  ierr = KDTreeSize(tree, &N);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Tree contains %d nodes.\n", N);

  loc[0] = 1.0; loc[1] = 1.0; loc[2] = 1.2;
  ierr = KDTreeFindWithinRange(tree, loc, 15.0, &nns);CHKERRQ(ierr);
  ierr = KDValuesSize(nns, &k);CHKERRQ(ierr);
  ierr = PetscCalloc1(k, &nodes);CHKERRQ(ierr);
  
  PetscPrintf(PETSC_COMM_WORLD, "Result contains %d elements within distance 1500 of (1.0, 1.0, 1.2).\n", k);
  ierr = KDValuesGetNodeDistance(nns, &d);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Closest node is at distance %.4f.\n", d);
  ierr = KDValuesGetNodeData(nns, (void**)&node, NULL);CHKERRQ(ierr);
  j=0;
  nodes[j] = node;
  ierr = RBFNodeGetLocation(node, loc);CHKERRQ(ierr);
  ierr = RBFNodeViewPolynomialBasis(node);CHKERRQ(ierr);
  eloc[0] = loc[0] + 0.1;
  eloc[1] = loc[1] + 0.2;
  eloc[2] = loc[2] - 0.4;
  ierr = RBFNodeEvaluateAtPoint(node, eloc, &val);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Node with location [%4.4f, %4.4f, %4.4f] evaluated at [%4.4f, %4.4f, %4.4f] gives value %4.4f\n", loc[0],loc[1],loc[2], eloc[0],eloc[1],eloc[2], val);
  i = KDValuesNext(nns);
  while(i != KDValuesEnd(nns)){
    ++j;
    ierr = KDValuesGetNodeData(nns, (void**)&node, NULL);CHKERRQ(ierr);
    nodes[j] = node;
    ierr = KDValuesGetNodeDistance(nns, &d);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Next-closest node is at distance %.4f.\n", d);
    i = KDValuesNext(nns);
  }

  Mat A;
  Vec L;
  ierr = rbf_interp_get_weight_problem(prob,k,nodes,1,eloc,&A,&L);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"A matrix:\n");
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  PetscPrintf(PETSC_COMM_WORLD,"L:\n");
  ierr = VecView(L, PETSC_VIEWER_STDOUT_WORLD);

  MatDestroy(&A);
  VecDestroy(&L);
  
  ierr = KDValuesDestroy(nns);CHKERRQ(ierr);
  ierr = RBFProblemDestroy(prob);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return 0;
}
  
  
