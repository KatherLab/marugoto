#!/bin/bash
# Builds and runs a marugoto container.
#
# Usage:
# ./marugoto-container.sh \
#      marugoto.mil.train --target_label isMSIH [other marugoto options]

podman build --target deploy $(dirname -- "$0")
image_id=$(podman build --target deploy --quiet $(dirname -- "$0"))

podman run --rm -ti --shm-size 16g \
	--security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d/ \
	--volume $HOME:$HOME \
	--volume /mnt:/mnt \
	--volume /run/media:/run/media \
	"$image_id" \
	"$@"
