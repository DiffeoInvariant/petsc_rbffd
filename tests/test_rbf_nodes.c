#include "rbf.h"


int main(int argc, char **argv)
{

  PetscInt i,j, k=3, N=100;
  PetscReal eps=0.1;
  Vec       X[3];
  PetscScalar *x;
  RBFProblem prob;
  KDTree     tree;
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
  ierr = RBFProblemDestroy(prob);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
  
  
