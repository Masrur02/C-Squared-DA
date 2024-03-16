#!/bin/bash

TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.json"
CHECKPOINT_FILE="${TEST_ROOT}/latest.pth"
SHOW_DIR="${TEST_ROOT}/preds"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE


python3 -m tools.inf_ros ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --opacity 0.5 --dataset "rugd"