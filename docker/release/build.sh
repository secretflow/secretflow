#!/bin/bash
# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

IMAGE_TAG=secretflow/secretflow-ubuntu:${VERSION}
LATEST_TAG=secretflow/secretflow-ubuntu:latest
IMAGE_LITE_TAG=secretflow/secretflow-lite-ubuntu:${VERSION}
LATEST_LITE_TAG=secretflow/secretflow-lite-ubuntu:latest

# wait pypi to be updated.
sleep 10s

echo -e "Building ${GREEN}${IMAGE_TAG}${NO_COLOR}"
(cd ../ && cp *.yml release/)
#Enable or switch to a specific Docker buildx builder )
docker buildx create --name secretflow
docker buildx use secretflow

#Building Secretflow Lite Multi Platform Images
docker buildx build \
  --platform linux/arm64/v8,linux/amd64 \
  -f ubuntu-lite.Dockerfile \
  -t ${IMAGE_LITE_TAG} \
  --build-arg sf_version=${VERSION} \
  --build-arg config_templates="$(cat config_templates.yml)" \
  --build-arg deploy_templates="$(cat deploy_templates.yml)" \
  . --push

#Output construction completion information
echo -e "Finish building ${GREEN}${IMAGE_LITE_TAG} for linux/arm64 and linux/amd64${NO_COLOR}"

#Building multi platform images
docker buildx build \
  --platform linux/arm64/v8,linux/amd64 \
  -f ubuntu.Dockerfile \
  -t ${IMAGE_TAG} \
  --build-arg sf_version=${VERSION} \
  --build-arg config_templates="$(cat config_templates.yml)" \
  --build-arg deploy_templates="$(cat deploy_templates.yml)" \
  . --push

#Output construction completion information
echo -e "Finish building ${GREEN}${IMAGE_TAG} for linux/arm64 and linux/amd64${NO_COLOR}"

if [[ LATEST -eq 1 ]]; then
    echo -e "Tag and push ${GREEN}${LATEST_LITE_TAG}${NO_COLOR} ..."
    docker buildx imagetools create --tag ${LATEST_LITE_TAG} ${IMAGE_LITE_TAG}

    echo -e "Tag and push ${GREEN}${LATEST_TAG}${NO_COLOR} ..."
    docker buildx imagetools create --tag ${LATEST_TAG} ${IMAGE_TAG}
fi

rm *.yml
rm *.whl