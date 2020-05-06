#include <petscfe.h>
#include <petscdm.h>

/* NOTE: these are global (i.e. pseudospectral) monomials, UNLIKE PETSc's Polynomial classes,
   which is evaluated at points on a reference element. */


const char *const RBFMonomialBasisTypes[] = {"Standard",0};
