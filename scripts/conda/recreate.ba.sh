#!/usr/bin/env bash

rm -rf ./.conda
conda env create -f ./rwep_dev_conda_env.yml --prefix .conda
cp pip.conf .conda/
