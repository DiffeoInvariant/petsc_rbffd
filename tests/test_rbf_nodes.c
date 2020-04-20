#include "rbf.h"


int main(int argc, char **argv)
{

  PetscInt i,j, k=3, N=100;
  PetscReal d, eps=0.1;
  Vec       X[3];
  PetscScalar *x, loc[3];
  RBFProblem prob;
  KDTree     tree;
  KDValues   nns;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  for(i=0; i<k; ++i){
    ierr = VecCreate(PETSC_COMM_WORLD, &X[i]);CHKERRQ(ierr);
    ierr = VecSetFromOptions(X[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(X[i], PETSC_DECIDE, N);CHKERRQ(ierr);
    ierr = VecGetArray(X[i], &x);CHKERRQ(ierr);
    for(j=0; j<N; ++j){
      x[j] = 0.5*j;
    }
    ierr = VecRestoreArray(X[i], &x);CHKERRQ(ierr);
  }

  ierr = RBFProblemCreate(&prob, k);CHKERRQ(ierr);
  ierr = RBFProblemSetType(prob, RBF_INTERPOLATE);CHKERRQ(ierr);
  ierr = RBFProblemSetNodeType(prob, RBF_GA);CHKERRQ(ierr);
  ierr = RBFProblemSetNodes(prob, X, &eps);CHKERRQ(ierr);

  ierr = RBFProblemGetTree(prob, &tree);CHKERRQ(ierr);
  ierr = KDTreeSize(tree, &N);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Tree contains %d nodes.\n", N);

  loc[0] = 1.0; loc[1] = 1.0; loc[2] = 1.2;
  ierr = KDTreeFindWithinRange(tree, loc, 1.5, &nns);CHKERRQ(ierr);
  ierr = KDValuesSize(nns, &k);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Result contains %d elements within distance 1.5 of (1.0, 1.0, 1.2) (expected 3).\n", k);
  ierr = KDValuesGetNodeDistance(nns, &d);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Closest node is at distance %.4f.\n", d);
  ierr = KDValuesDestroy(nns);CHKERRQ(ierr);
  ierr = RBFProblemDestroy(prob);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
  
  
