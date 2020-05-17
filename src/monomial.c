#include <petscfe.h>
#include <petscdm.h>
#include <rbf/private/monomial.h>
/* NOTE: these are global (i.e. pseudospectral) monomials, UNLIKE PETSc's Polynomial classes,
   which is evaluated at points on a reference element. */

static PetscErrorCode PetscSpaceInitialize_Monomial(PetscSpace sp)
{
  sp->ops->setfromoptions = PetscSpaceSetFromOptions_Monomial;
  sp->ops->setup          = PetscSpaceSetUp_Monomial;
  sp->ops->view           = PetscSpaceView_Monomial;
  sp->ops->destroy        = PetscSpaceDestroy_Monomial;
  sp->ops->getdimension   = PetscSpaceGetDimension_Monomial;
  sp->ops->evaluate       = PetscSpaceEvaluate_Monomial;
  return 0;
}

static PetscErrorCode PetscSpaceCreate_Polynomial(PetscSpace sp)
{
  PetscSpaceInitialize_Monomial(sp);
  return 0;
}

static PetscErrorCode PetscSpaceGetDimension_Monomial(PetscSpace sp, PetscInt *dim)
{
  PetscInt  i,deg=sp->degree, dims=sp->Nv;
  PetscReal D=1.0;

  for (i = 1; i <= dims; ++i) {
    D *= ((PetscReal)(deg+i))/i;
  }
  *dim = (PetscInt)(D+0.5);
  return 0;
}

static PetscErrorCode FillMonomialBasisTerm(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscInt order, PetscInt dims)
{
  PetscInt i,j;
  for (i = 0; i < npoints; ++i) {
    for (j = 0; j < dims; ++j) {
      B[dims*i + j] = PetscPowReal(points[dims*i + j],(PetscReal)order);
    }
  } 
  return 0;
}

static PetscErrorCode PetscSpaceEvaluate_Monomial(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscInt       i,nterms,dims=sp->Nv,deg=sp->degree;
  DM             dm = sp->dm;
  PetscErrorCode ierr;
  
  PetscSpaceGetDimension(sp,&nterms);
  for (i = 0; i < deg; ++i) {
    FillMonomialBasisTerm(sp,npoints
  
