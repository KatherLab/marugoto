#!/bin/bash
# Builds and runs a marugoto container.
#
# Usage:
# ./marugoto-container.sh \
#      -v /path/to/wsis:/wsis [other podman options] \
#      -- \
#      marugoto.mil.train --target_label isMSIH [other marugoto options]
#
# i.e. a -- is used to separate the podman options from marugoto options

set -e

podman_args=()
while [[ $# -gt  0 ]]; do
    case $1 in
        --) shift; break;; # we're done with podman args
        *) podman_args+=("$1"); shift;; # append to podman args
    esac
done

image_id=$(podman build -q $(dirname -- "$0"))

podman run --rm -ti \
    --security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d/ \
    "${podman_args[@]}" \
    "$image_id" \
    "$@"
