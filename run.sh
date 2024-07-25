#!/bin/bash
export AFRAME_CONDOR_DIR=/home/ethan.marx/projects/a3d3-ligo-dataset/data/condor
export AFRAME_TMPDIR=/home/ethan.marx/projects/a3d3-ligo-dataset/data/tmp
LAW_CONFIG_FILE=./config.cfg poetry run law run a3d3_ligo_dataset.pipeline.A3D3Dataset --dev --workers 5
