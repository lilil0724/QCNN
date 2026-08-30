#!/usr/bin/env bash
# Source this file to configure the paths used by QCNN experiments.
#
# Usage:
#   source scripts/setup_qcnn_env.sh
#   source scripts/setup_qcnn_env.sh /path/to/a/specific/pair

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced so its variables remain in the current shell:"
    echo "  source scripts/setup_qcnn_env.sh [optional_pair_dir]"
    exit 1
fi

export HF_HOME=/home/pcs5060ti/Desktop/hf
export HF_HUB_CACHE=/home/pcs5060ti/Desktop/hf/hub
export QCNN_PAIRS_ROOT=/home/pcs5060ti/Desktop/qcnn_data/pairs

_qcnn_requested_pair="${1:-}"

if [[ -n "$_qcnn_requested_pair" ]]; then
    if [[ ! -f "$_qcnn_requested_pair/manifest.csv" ]]; then
        echo "QCNN environment error: manifest.csv not found under:"
        echo "  $_qcnn_requested_pair"
        unset _qcnn_requested_pair
        return 1
    fi
    export PAIR="$_qcnn_requested_pair"
elif [[ -n "${PAIR:-}" && -f "$PAIR/manifest.csv" ]]; then
    : # Keep an already valid selection.
else
    _qcnn_manifests=()
    if [[ -d "$QCNN_PAIRS_ROOT" ]]; then
        mapfile -t _qcnn_manifests < <(
            find "$QCNN_PAIRS_ROOT" -mindepth 2 -maxdepth 2 \
                -type f -name manifest.csv -print | sort
        )
    fi

    if [[ ${#_qcnn_manifests[@]} -eq 1 ]]; then
        export PAIR="${_qcnn_manifests[0]%/manifest.csv}"
    elif [[ ${#_qcnn_manifests[@]} -eq 0 ]]; then
        unset PAIR
        echo "QCNN environment warning: no manifest.csv found under:"
        echo "  $QCNN_PAIRS_ROOT"
    else
        unset PAIR
        echo "QCNN environment: multiple pair directories were found:"
        for _qcnn_manifest in "${_qcnn_manifests[@]}"; do
            echo "  ${_qcnn_manifest%/manifest.csv}"
        done
        echo "Select one with:"
        echo "  source scripts/setup_qcnn_env.sh /full/path/to/pair"
    fi
fi

echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "QCNN_PAIRS_ROOT=$QCNN_PAIRS_ROOT"
if [[ -n "${PAIR:-}" ]]; then
    echo "PAIR=$PAIR"
fi

unset _qcnn_requested_pair _qcnn_manifest _qcnn_manifests
