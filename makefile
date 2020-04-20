PETSC_DIR=/usr/local/petsc
PETSC_ARCH=arch-linux-cxx-debug
CC=clang
CFLAGS=-std=c99 -O3 -march=native -mtune=native -fPIC -g
LDFLAGS=-shared $(PETSC_WITH_EXTERNAL_LIB) 

include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscvariables
include $(PETSC_DIR)/$(PETSC_ARCH)/lib/petsc/conf/petscrules

.PHONY: all kdtree rbf

KDTREE_INCL=include
RBF_INCL=include

all: kdtree rbf

kdtree: src/kdtree.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(PETSC_CC_INCLUDES) $^ -I$(KDTREE_INCL) -o bin/libkdtree.so

rbf: src/rbf.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(PETSC_CC_INCLUDES) $^ -I$(RBF_INCL) -Lbin/ -lkdtree  -o bin/librbf.so

test: tests/test_single_query.c
	$(CC) -std=c99 -O3 -march=native -mtune=native -g  $(PETSC_CC_INCLUDES) $(PETSC_WITH_EXTERNAL_LIB) $^ -I$(KDTREE_INCL) -Lbin/ -lkdtree -o tests/tsq
