#!/bin/bash
cd "${0%/*}" || exit
set -e

#------------------------------------------------------------------------------

cleanCase()
{
    zeros=""
    while [ ${#zeros} -lt 8 ]
    do
        timeDir="0.${zeros}[1-9]*"
        rm -rf ./${timeDir} ./-${timeDir} > /dev/null 2>&1
        zeros="0$zeros"
    done
    rm -rf ./[1-9]* ./-[1-9]* ./log ./log.* ./log-* ./logSummary.* ./.fxLock ./*.xml ./ParaView* ./paraFoam* ./*.OpenFOAM ./*.blockMesh ./.setSet > /dev/null 2>&1

    if [ -d system -a -d dynamicCode ]
    then
        rm -rf dynamicCode > /dev/null 2>&1
    fi

    rm -rf postProcessing > /dev/null 2>&1
    rm -rf probes* > /dev/null 2>&1
    rm -rf forces* > /dev/null 2>&1
    rm -rf graphs* > /dev/null 2>&1
    rm -rf sets > /dev/null 2>&1
    rm -rf surfaceSampling > /dev/null 2>&1
    rm -rf cuttingPlane > /dev/null 2>&1
    rm -rf system/machines > /dev/null 2>&1

    rm -rf VTK > /dev/null 2>&1
    rm -f 0/cellLevel 0/pointLevel 0/cellDist constant/cellDecomposition
}

cleanPrecice()
{
    rm -rf ./preCICE-output/
    rm -rf ./preCICE-*/
}

(
    cleanCase
    cleanPrecice
)
