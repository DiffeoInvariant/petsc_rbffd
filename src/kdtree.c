#include "kdtree.h"
#include <pthread.h>

struct kdbox {
  PetscInt    k;
  PetscScalar *min, *max;
};

struct kdnode {
  PetscScalar   *loc;
  PetscInt      dir;

  void          *data;
  struct kdnode *left, *right;
};


struct result_node {
  struct kdnode      *node;
  PetscReal          dist;
  struct result_node *next; /* for iterating over results of a search */
};


struct _p_kdtree {
  PetscInt      k, N;
  struct kdnode *root;
  struct kdbox  *bounding_box;
  NodeDestructor dtor;
};

struct _p_kd_values {
  KDTree             tree;
  PetscInt           size;
  struct result_node *reslist, *resiter;
};

static struct result_node* allocate_result_node();
static void free_result_node(struct result_node *node);
  
static void clear_box(struct kdnode *node, NodeDestructor dtor);
static PetscErrorCode insert_box(struct kdnode **node, const PetscScalar *loc, void *data, PetscInt dir, PetscInt k);
static PetscErrorCode result_list_insert(struct result_node *list, struct kdnode *dat, PetscReal dist);
static PetscErrorCode clear_result_list(KDValues set);

static struct kdbox* kdbox_create(PetscInt k, const PetscScalar *min, const PetscScalar *max);
static PetscErrorCode kdbox_destroy(struct kdbox *box);
static struct kdbox* kdbox_copy(const struct kdbox *box);

static PetscErrorCode kdbox_extend(struct kdbox *box, const PetscScalar *loc);
static PetscErrorCode kdbox_square_dist(struct kdbox *box, const PetscScalar *loc, PetscReal *dsq);


PetscErrorCode KDTreeCreate(KDTree *tree, PetscInt k)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(tree);CHKERRQ(ierr);
  (*tree)->N = 0;
  (*tree)->k = k;
  (*tree)->root = NULL;
  (*tree)->bounding_box = NULL;
  (*tree)->dtor = NULL;
  PetscFunctionReturn(ierr);
}

PetscErrorCode KDTreeDestroy(KDTree tree)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = KDTreeClear(tree);CHKERRQ(ierr);
  tree->k = 0;
  ierr = PetscFree(tree);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode KDTreeGetK(KDTree tree, PetscInt *k)
{
 PetscFunctionBeginUser;
 *k = tree->k;
 PetscFunctionReturn(0);
}

static void clear_box(struct kdnode *node, NodeDestructor dtor)
{
  if(!node) return;

  clear_box(node->left, dtor);
  clear_box(node->right, dtor);

  if(dtor) dtor(node->data);

  PetscFree(node->loc);
  PetscFree(node);
}
  

PetscErrorCode KDTreeClear(KDTree tree)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  clear_box(tree->root, tree->dtor);
  tree->root = NULL;
  if(tree->bounding_box){
    ierr = kdbox_destroy(tree->bounding_box);CHKERRQ(ierr);
    tree->bounding_box = NULL;
  }
  tree->N = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode KDTreeSetNodeDestructor(KDTree tree, NodeDestructor dtor)
{
  PetscFunctionBeginUser;
  tree->dtor = dtor;
  PetscFunctionReturn(0);
}


static PetscErrorCode insert_box(struct kdnode **node, const PetscScalar *loc, void *data, PetscInt dir, PetscInt k)
{
  PetscInt newdir;
  struct kdnode *nnd;
  PetscErrorCode ierr;
  
  if(!*node){
    ierr = PetscNew(&nnd);CHKERRQ(ierr);
    ierr = PetscCalloc1(k, &(nnd->loc));CHKERRQ(ierr);
    ierr = PetscArraycpy(nnd->loc, loc, k);CHKERRQ(ierr);
    nnd->data = data;
    nnd->dir = dir;
    nnd->left = nnd->right = NULL;
    *node = nnd;
    return 0;
  }

  nnd = *node;
  newdir = (nnd->dir + 1) % k;
  if(loc[nnd->dir] < nnd->loc[nnd->dir]){
    return insert_box(&(*node)->left, loc, data, newdir, k);
  }
  return insert_box(&(*node)->right, loc, data, newdir, k);
}

PetscErrorCode KDTreeInsert(KDTree tree, const PetscScalar *loc, void *node_data)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = insert_box(&tree->root, loc, node_data, 0, tree->k);CHKERRQ(ierr);
  tree->N += 1;
  if(!(tree->bounding_box)){
    tree->bounding_box = kdbox_create(tree->k, loc, loc);
  } else {
    ierr = kdbox_extend(tree->bounding_box, loc);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode KDTreeInsert3D(KDTree tree, PetscScalar x, PetscScalar y, PetscScalar z, void *node_data)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  PetscScalar loc[3] = {x, y, z};
  ierr = KDTreeInsert(tree, loc, node_data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
#define _sqr(x) (x) * (x)

static void kdtree_nearest_helper(struct kdnode *node, const PetscScalar *loc,
				  struct kdnode **res, PetscReal *ressqdist,
				  struct kdbox *box)
{
  PetscInt i, dir = node->dir;
  PetscReal tmp, sqdst;
  struct kdnode *close_subtree, *far_subtree;
  PetscScalar   *close_box_coord, *far_box_coord;
  PetscErrorCode ierr;
  
  tmp = loc[dir] - node->loc[dir];
  if(tmp <= 0){
    close_subtree = node->left;
    far_subtree = node->right;
    close_box_coord = box->max + dir;
    far_box_coord = box->min + dir;
  } else {
    close_subtree = node->right;
    far_subtree = node->left;
    close_box_coord = box->min + dir;
    far_box_coord = box->max + dir;
  }
  if(close_subtree){
    /* chop the box up to get a bounding box for the close subtree */
    tmp = *close_box_coord;
    *close_box_coord = node->loc[dir];
    kdtree_nearest_helper(close_subtree, loc, res, ressqdist, box);
    *close_box_coord = tmp;
  }

  sqdst = 0.0;
  for(i=0; i<box->k; ++i){
    sqdst += _sqr(node->loc[i] - loc[i]);
  }
  if(sqdst < *ressqdist){
    *res = node;
    *ressqdist = sqdst;
  }

  PetscReal fsd;
  if(far_subtree){
    tmp = *far_box_coord;
    *far_box_coord = node->loc[dir];

    /* if the closest point in the bounding box is closer than 
       closest dist, we've gotta go down the farther tree */
    ierr = kdbox_square_dist(box, loc, &fsd);
    if(fsd < *ressqdist){
      kdtree_nearest_helper(far_subtree, loc, res, ressqdist, box);
    }
    *far_box_coord = tmp;
  }
}
    
    
PetscErrorCode KDTreeSize(const KDTree tree, PetscInt *N)
{
  PetscFunctionBeginUser;
  *N = tree->N;
  PetscFunctionReturn(0);
}

static PetscErrorCode KDValuesCreate(KDValues *vals, KDTree parent_tree)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(vals);CHKERRQ(ierr);
  (*vals)->reslist = allocate_result_node();
  if(!(*vals)->reslist){
    PetscFree(*vals);
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_MEM, "Could not allocate result list!\n");
  }
  (*vals)->reslist->next = NULL;
  (*vals)->tree = parent_tree;
  PetscFunctionReturn(0);
}

PetscErrorCode KDTreeFindNearest(const KDTree tree, const PetscScalar *loc, KDValues *nearest)
{
  struct kdbox   *box;
  struct kdnode  *resnode;
  KDValues       resset;
  PetscReal      square_dist;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  if(!tree) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "tree argument to KDTreeFindNearest cannot be NULL!");

  if(!tree->bounding_box) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL, "The bounding box member of the tree argument to KDTreeFindNearest cannot be NULL!\n");

  
  ierr = PetscNew(&resset);CHKERRQ(ierr);
  resset->reslist = allocate_result_node();
  if(!resset->reslist){
    PetscFree(resset);
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_MEM, "Could not allocate result list!\n");
  }
  resset->reslist->next = NULL;
  resset->tree = tree;

  box = kdbox_copy(tree->bounding_box);
  if(!box){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_MEM, "Count not copy bounding box!\n");
  }

  /* start at root, search through tree */
  resnode = tree->root;
  square_dist = 0.0;
  for(i=0; i< tree->k; ++i){
    square_dist += _sqr(resnode->loc[i] - loc[i]);
  }

  kdtree_nearest_helper(tree->root, loc, &resnode, &square_dist, box);
  ierr = kdbox_destroy(box);

  if(resnode){
    ierr = result_list_insert(resset->reslist, resnode, PetscSqrtReal(square_dist));CHKERRQ(ierr);
    resset->size = 1;
    KDValuesBegin(resset);
    *nearest = resset;
    PetscFunctionReturn(0);
  } else {
    KDValuesDestroy(resset);
    SETERRQ(PETSC_COMM_WORLD, 1, "KDTreeFindNearest failed to find a nearest node");
  }
}


PetscErrorCode KDTreeFindNearest3D(const KDTree tree, PetscScalar x, PetscScalar y, PetscScalar z,KDValues *nearest)
{
  PetscScalar loc[3] = {x, y, z};
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = KDTreeFindNearest(tree, loc, nearest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscInt find_nearest_node(struct kdnode *node, const PetscScalar *loc, PetscReal max_dist, struct result_node *reslist, PetscBool ordered, PetscInt k)
{
  PetscReal dsq, dx;
  PetscInt i, stat, total=0;
  PetscErrorCode ierr;

  if(!node) return 0;

  dsq = 0.0;
  for(i=0; i<k; ++i){
    dsq += _sqr(node->loc[i] - loc[i]);
  }
  if(dsq <= _sqr(max_dist)){
    ierr = result_list_insert(reslist, node, ordered ? PetscSqrtReal(dsq) : -1.0);CHKERRQ(ierr);
    total = 1;
  }

  dx = loc[node->dir] - node->loc[node->dir];
  stat = find_nearest_node(dx > 0 ? node->right : node->left, loc, max_dist, reslist, ordered, k);
  if(stat >= 0 && PetscAbsReal(dx) < max_dist){
    total += stat;
    stat = find_nearest_node(dx > 0 ? node->left : node->right, loc, max_dist, reslist, ordered, k);
  }
  if(stat == -1){
    return -1;
  }
  total += stat;
  return total;
}

PetscErrorCode KDTreeFindWithinRange(const KDTree tree, const PetscScalar *loc, PetscReal range, KDValues *nodes)
{
  PetscInt nnode;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = KDValuesCreate(nodes, tree);CHKERRQ(ierr);
  nnode = find_nearest_node(tree->root, loc, range, (*nodes)->reslist, PETSC_TRUE, tree->k);
  if(nnode < 0){
    SETERRQ(PETSC_COMM_WORLD, 1, "Error, KDTreeFindWithinRange could not find any nodes!\n");
  }
  (*nodes)->size = nnode;
  ierr = KDValuesBegin(*nodes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct result_node *free_nodes;

static pthread_mutex_t node_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

static struct result_node* allocate_result_node()
{
  struct result_node *node;
  pthread_mutex_lock(&node_alloc_mutex);

  if(!free_nodes){
    PetscNew(&node);
  } else {
    node = free_nodes;
    free_nodes = free_nodes->next;
    node->next = NULL;
  }

  pthread_mutex_unlock(&node_alloc_mutex);

  return node;
}

static void free_result_node(struct result_node *node)
{
  pthread_mutex_lock(&node_alloc_mutex);
  node->next = free_nodes;
  free_nodes = node;
  pthread_mutex_unlock(&node_alloc_mutex);
}

static PetscErrorCode result_list_insert(struct result_node *list, struct kdnode *dat, PetscReal dist)
{
  PetscFunctionBeginUser;
  struct result_node *rnode = allocate_result_node();
  if(!rnode){
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_MEM, "Could not allocate result node!\n");
  }

  rnode->node = dat;
  rnode->dist = dist;
  if(dist >= 0.0){
    while(list->next && list->next->dist < dist){
      list = list->next;
    }
  }

  rnode->next = list->next;
  list->next = rnode;
  PetscFunctionReturn(0);
}

static PetscErrorCode clear_result_list(KDValues set)
{
  struct result_node *tmp, *node = set->reslist->next;

  while(node){
    tmp = node;
    node = node->next;
    free_result_node(tmp);
  }

  set->reslist->next = NULL;

  return 0;
}


PetscErrorCode KDValuesBegin(KDValues vals)
{
  PetscFunctionBeginUser;
  vals->resiter = vals->reslist->next;
  PetscFunctionReturn(0);
}

PetscInt KDValuesEnd(const KDValues vals)
{
  return 0;
}

PetscInt KDValuesNext(KDValues vals)
{
  vals->resiter = vals->resiter->next;
  return vals->resiter ? 1 : 0;
}

PetscErrorCode KDValuesSize(KDValues vals, PetscInt *n)
{
  PetscFunctionBeginUser;
  if(!vals){
    *n = 0;
    PetscFunctionReturn(0);
  } else {
    *n = vals->size;
    PetscFunctionReturn(0);
  }
}

PetscErrorCode KDValuesGetNodeData(const KDValues vals, void *nodedata, const PetscScalar *loc)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(vals->resiter){
    if(loc){
      ierr = PetscArraycpy(vals->resiter->node->loc, loc, vals->tree->k);CHKERRQ(ierr);
    }
    nodedata = vals->resiter->node->data;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode KDValuesGetNodeDistance(const KDValues vals, PetscReal *dist)
{
  PetscFunctionBeginUser;
  if(vals->resiter){
    *dist = vals->resiter->dist;
  } else {
    SETERRQ(PETSC_COMM_WORLD, 1, "Cannot get node distance for KDValues without any values!\n");
  }
  PetscFunctionReturn(0);
}
  
PetscErrorCode KDValuesDestroy(KDValues vals)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = clear_result_list(vals);CHKERRQ(ierr);
  free_result_node(vals->reslist);
  ierr = PetscFree(vals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
static struct kdbox*
kdbox_create(PetscInt k, const PetscScalar *min, const PetscScalar *max)
{ 
  struct kdbox *box;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(&box); if(ierr) return NULL;

  box->k = k;
  ierr = PetscCalloc2(k, &(box->min), k, &(box->max));
  if(ierr){
    if(box->min) PetscFree(box->min);
    if(box->max) PetscFree(box->max);
    PetscFree(box);
    return NULL;
  }
  PetscArraycpy(box->min, min, k);
  PetscArraycpy(box->max, max, k);

  return box;
}

static PetscErrorCode kdbox_destroy(struct kdbox *box)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(box->min){
    ierr = PetscFree(box->min);CHKERRQ(ierr);
  }
  if(box->max){
    ierr = PetscFree(box->max);CHKERRQ(ierr);
  }
  ierr = PetscFree(box);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode kdbox_extend(struct kdbox *box, const PetscScalar *loc)
{
  PetscInt i;
  PetscFunctionBeginUser;
  for(i=0; i<box->k; ++i){
    if(loc[i] < box->min[i]){
      box->min[i] = loc[i];
    }
    if(loc[i] > box->max[i]){
      box->max[i] = loc[i];
    }
  }
  PetscFunctionReturn(0);
}


static PetscErrorCode kdbox_square_dist(struct kdbox *box, const PetscScalar *loc, PetscReal *dsq)
{
  PetscInt i;
  PetscReal sqd = 0;
  PetscFunctionBeginUser;
  for(i=0; i<box->k; ++i){
    if(loc[i] < box->min[i]){
      sqd += _sqr(box->min[i] - loc[i]);
    } else if(loc[i] > box->max[i]){
      sqd += _sqr(box->max[i] - loc[i]);
    }
  }
  *dsq = sqd;
  PetscFunctionReturn(0);
}

static struct kdbox* kdbox_copy(const struct kdbox *box)
{
  return kdbox_create(box->k, box->min, box->max);
}
