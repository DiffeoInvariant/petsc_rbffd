PETSC_DIR=/usr/local/petsc
PETSC_ARCH=arch-linux-cxx-debug
CC=clang
CFLAGS=-std=c99 -O3 -march=native -mtune=native -fPIC -g
LDFLAGS=-shared $(PETSC_WITH_EXTERNAL_LIB) 

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

.PHONY: all testkd testrbf kdtree rbf

KDTREE_INCL=include
RBF_INCL=include

all: kdtree rbf testrbf testkd

kdtree: src/kdtree.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(PETSC_CC_INCLUDES) $^ -I$(KDTREE_INCL) -o bin/libkdtree.so

rbf: src/rbf.c src/kdtree.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(PETSC_CC_INCLUDES) $^ -I$(RBF_INCL) -o bin/librbf.so

testrbf: tests/test_rbf_nodes.c
	$(CC) -std=c99 -O3 -march=native -mtune=native -g  $(PETSC_CC_INCLUDES) $(PETSC_WITH_EXTERNAL_LIB) $^ -I$(KDTREE_INCL) -Lbin/ -Wl,-rpath=bin/ -lrbf -o tests/trbfn

testkd: tests/test_single_query.c
	$(CC) -std=c99 -O3 -march=native -mtune=native -g  $(PETSC_CC_INCLUDES) $(PETSC_WITH_EXTERNAL_LIB) $^ -I$(KDTREE_INCL) -Lbin/ -Wl,-rpath=bin/ -lkdtree -o tests/tsq
