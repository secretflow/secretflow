#!/bin/bash

set -e

show_help() {
    echo "Usage: bash build.sh [OPTION]... -v {the_version}"
    echo "  -v  --version"
    echo "          the secretflow version to build with."
    echo "  -l --latest"
    echo "          tag this version as latest and push to docker repo."
    echo
}

if [[ "$#" -lt 2 ]]; then
    show_help
    exit
fi

while [[ "$#" -ge 1 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift
            if [[ "$#" -eq 0 ]]; then
                echo "Version shall not be empty."
                echo ""
                show_help
                exit 1
            fi
            shift
        ;;
        -l|--latest)
            LATEST=1
            shift
        ;;
        *)
            echo "Unknown argument passed: $1"
            exit 1
        ;;
    esac
done


if [[ -z ${VERSION} ]]; then
    echo "Please specify the version."
    exit 1
fi


GREEN="\033[32m"
NO_COLOR="\033[0m"

IMAGE_TAG=secretflow/secretflow-anolis8:${VERSION}
LATEST_TAG=secretflow/secretflow-anolis8:latest
IMAGE_LITE_TAG=secretflow/secretflow-lite-anolis8:${VERSION}
LATEST_LITE_TAG=secretflow/secretflow-lite-anolis8:latest

echo -e "Building ${GREEN}${IMAGE_TAG}${NO_COLOR}"
docker build . -f anolis.Dockerfile -t ${IMAGE_TAG} --build-arg sf_version=${VERSION}
echo -e "Finish building ${GREEN}${IMAGE_TAG}${NO_COLOR}"
docker push ${IMAGE_TAG}

echo -e "Building ${GREEN}${IMAGE_LITE_TAG}${NO_COLOR}"
docker build . -f anolis-lite.Dockerfile -t ${IMAGE_LITE_TAG} --build-arg sf_version=${VERSION}
echo -e "Finish building ${GREEN}${IMAGE_LITE_TAG}${NO_COLOR}"
docker push ${IMAGE_LITE_TAG}

if [[ LATEST -eq 1 ]]; then
    echo -e "Tag and push ${GREEN}${LATEST_TAG}${NO_COLOR} ..."
    docker tag ${IMAGE_TAG} ${LATEST_TAG}
    docker push ${LATEST_TAG}
    echo -e "Tag and push ${GREEN}${LATEST_LITE_TAG}${NO_COLOR} ..."
    docker tag ${IMAGE_LITE_TAG} ${LATEST_LITE_TAG}
    docker push ${LATEST_LITE_TAG}
fi
