#!/bin/bash
cd "${0%/*}" || exit
. ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions
#------------------------------------------------------------------------------

## run parallel
mpirun -np $(getNumberOfProcessors) --bind-to none  pimpleFoam  -parallel > log.pimpleFoam 2>&1 &

## run single
# runApplication $(getApplication)
