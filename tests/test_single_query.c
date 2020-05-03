#include "kdtree.h"


struct _s_star_info {
  PetscReal x0, y0, z0, m;
};

typedef struct _s_star_info *star_info;

int main(int argc, char **argv)
{
  star_info star1, star2, star3;
  PetscErrorCode   ierr;
  KDTree           tree;
  KDValues         results;
  PetscInt         k=3;
  PetscReal        x[3];
  /*PetscReal        x0, y0, z0, m;*/
  /*PetscBool        flg=PETSC_FALSE;*/

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscCalloc3(1, &star1, 1, &star2, 1, &star3);CHKERRQ(ierr);
  /*
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Test KDTree", NULL);CHKERRQ(ierr);
  k=3;
  ierr = PetscOptionsInt("-k", "Dimensions", NULL, k, &k, NULL);CHKERRQ(ierr);
  */

  star1->x0 = 0.0; star1->y0 = 0.1; star1->z0 = 0.0;
  star1->m = 1.0;

  star2->x0 = 1.0; star2->y0 = -0.97; star2->z0 = 3.0;
  star2->m = 1.3;

  star3->x0 = -0.4; star3->y0 = 0.83; star3->z0 = 1.02;
  star3->m = 0.85;

  ierr = KDTreeCreate(&tree, k);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Created tree\n");
  x[0] = star1->x0; x[1] = star1->y0; x[2] = star1->z0;
  ierr = KDTreeInsert(tree, x, star1);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "inserted a star\n");
  x[0] = star2->x0; x[1] = star2->y0; x[2] = star2->z0;
  ierr = KDTreeInsert(tree, x, &star2);CHKERRQ(ierr);
  x[0] = star3->x0; x[1] = star3->y0; x[2] = star3->z0;
  ierr = KDTreeInsert(tree, x, &star3);CHKERRQ(ierr);
  
  ierr = KDTreeFindNearest(tree, x, &results);CHKERRQ(ierr);
  ierr = KDValuesSize(results, &k);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Result contains %d element.\n", k);

  ierr = KDValuesGetNodeData(results, (void**)&star1, NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Closest node to (-0.4, 0.83, 1.02): (%.3f, %.3f, %.3f).\n", star1->x0, star1->y0, star1->z0);
  ierr = KDValuesDestroy(results);CHKERRQ(ierr);

  x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
  ierr = KDTreeFindWithinRange(tree, x, 3.0, &results);CHKERRQ(ierr);
  ierr = KDValuesSize(results, &k);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Result contains %d elements within distance 3.0.\n", k);
  
  ierr = KDValuesDestroy(results);CHKERRQ(ierr);
  ierr = KDTreeDestroy(tree);CHKERRQ(ierr);

  return 0;
}
