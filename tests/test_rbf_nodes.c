#include "rbf.h"
#include <stdlib.h>
#include <petscksp.h>
int main(int argc, char **argv)
{

  PetscInt i,j, k=3, N=100;
  PetscReal d, eps=0.001;
  Vec       V,X[3];
  PetscScalar val, *x, loc[3], eloc[3];
  RBFProblem prob;
  RBFNode    node, *nodes;
  KDTree     tree;
  KDValues   nns;
  PetscErrorCode ierr;

  
  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  srand(0);
  /* create node locations */
  for(i=0; i<k; ++i){
    ierr = VecCreate(PETSC_COMM_WORLD, &X[i]);CHKERRQ(ierr);
    ierr = VecSetFromOptions(X[i]);CHKERRQ(ierr);
    ierr = VecSetSizes(X[i], PETSC_DECIDE, N);CHKERRQ(ierr);
    ierr = VecGetArray(X[i], &x);CHKERRQ(ierr);
    for(j=0; j<N; ++j){
      x[j] = 0.5*((PetscReal)((rand() / ((RAND_MAX + 1u)/30))) - 15.);
    }
    ierr = VecRestoreArray(X[i], &x);CHKERRQ(ierr);
  }
  ierr = VecCreate(PETSC_COMM_WORLD, &V);CHKERRQ(ierr);
  ierr = VecSetFromOptions(V);CHKERRQ(ierr);
  ierr = VecSetSizes(V, PETSC_DECIDE, N);CHKERRQ(ierr);
  ierr = VecSet(V,3.0);CHKERRQ(ierr);
  /* create problem data structures */
  ierr = RBFProblemCreate(PETSC_COMM_WORLD,k,&prob);CHKERRQ(ierr);

  ierr = RBFProblemSetPolynomialDegree(prob,0);CHKERRQ(ierr);
  ierr = RBFProblemSetType(prob,RBF_INTERPOLATE);CHKERRQ(ierr);
  ierr = RBFProblemSetNodeType(prob,RBF_GA,NULL);CHKERRQ(ierr);
  ierr = RBFProblemSetNodesWithValues(prob,V,X,&eps);CHKERRQ(ierr);
  ierr = RBFProblemSetNodeWeightRadius(prob,10.0);CHKERRQ(ierr);
  ierr = RBFProblemView(prob);CHKERRQ(ierr);

  for(i=0; i<k; ++i){
    ierr = VecDestroy(&X[i]);CHKERRQ(ierr);
  }
  ierr = RBFProblemGetTree(prob, &tree);CHKERRQ(ierr);
  ierr = KDTreeSize(tree, &N);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Tree contains %d nodes.\n", N);

  loc[0] = 0.0; loc[1] = 0.0; loc[2] = 0.0;
  PetscReal mdist = 5;
  ierr = KDTreeFindWithinRange(tree, loc, mdist, &nns);CHKERRQ(ierr);
  ierr = KDValuesSize(nns, &k);CHKERRQ(ierr);
  ierr = PetscCalloc1(k, &nodes);CHKERRQ(ierr);
 
  PetscPrintf(PETSC_COMM_WORLD, "Result contains %d elements within distance %4.3e of (0.0, 0.0, 0.0).\n", k, mdist);
  ierr = KDValuesGetNodeDistance(nns, &d);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Closest node is at distance %.4f.\n", d);
  ierr = KDValuesGetNodeData(nns, (void**)&node, NULL);CHKERRQ(ierr);
  j=0;
  nodes[j] = node;
  ierr = RBFNodeGetLocation(node, loc);CHKERRQ(ierr);
  ierr = RBFNodeViewPolynomialBasis(node);CHKERRQ(ierr);
  eloc[0] = loc[0] + 0.1;
  eloc[1] = loc[1] + 0.2;
  eloc[2] = loc[2] - 0.16;
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

  #if 0
  PetscErrorCode rbf_interp_get_weight_problem(RBFProblem  prob,
						    PetscInt    stencil_size,
						    RBFNode     *stencil_nodes,
						    PetscScalar coeff,
						    const PetscScalar *target_point,
						    Mat *AP,
					       Vec *L);
  Mat A;
  Vec L,V;
  KSP ksp;
  ierr = rbf_interp_get_weight_problem(prob,k,nodes,1,eloc,&A,&L);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"A matrix:\n");
  ierr = MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  PetscPrintf(PETSC_COMM_WORLD,"L:\n");
  ierr = VecView(L, PETSC_VIEWER_STDOUT_WORLD);

  KSPCreate(PETSC_COMM_WORLD,&ksp);
  KSPSetOperators(ksp,A,A);
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
  /*interpolating f(x)=3*/
  VecDuplicate(L,&V);
  VecSet(V,3.0);

  KSPSolve(ksp,V,V);

  PetscPrintf(PETSC_COMM_WORLD,"Interpolation weights:\n");
  VecView(V,PETSC_VIEWER_STDOUT_WORLD);

  
  PetscReal fx;
  VecDot(V,L,&fx);
  PetscPrintf(PETSC_COMM_WORLD,"Interpolated value (target 3.0): %g\n",fx);
  /*VecView(V,PETSC_VIEWER_STDOUT_WORLD);*/

  KSPDestroy(&ksp);
  MatDestroy(&A);
  VecDestroy(&L);
  VecDestroy(&V);
  #endif

  #if 1
  ierr = RBFProblemSetUp(prob);CHKERRQ(ierr);
  #endif
  
  ierr = KDValuesDestroy(nns);CHKERRQ(ierr);
  ierr = RBFProblemDestroy(&prob);CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return 0;
}
  
  
