#!/usr/bin/env bash

WORKSPACE='./scripts'

bash ${WORKSPACE}/conda/recreate.ba.sh
source ${WORKSPACE}/conda/activate.env.sh

bash ${WORKSPACE}/pip/install.ba.sh
