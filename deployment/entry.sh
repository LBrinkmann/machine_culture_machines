#!/bin/bash
# set -e

. /appenv/bin/activate

echo "$@"

exec "$@"