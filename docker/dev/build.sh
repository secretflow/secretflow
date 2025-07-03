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

set -ex

show_help() {
    echo "Usage: bash build.sh [OPTION]... -v {version}"
    echo "  -v  --version"
    echo "          the version to build with."
    echo "  -r  --reg"
    echo "          docker reg to upload."
    echo "  -l  --latest"
    echo "          tag this version as latest."
    echo "  -u  --upload"
    echo "          upload to docker registry."
    echo "  -t  --tar"
    echo "          save the built image as a tar file named sf-dev-ubuntu-{version}.tar."
}

if [[ "$#" -lt 2 ]]; then
    show_help
    exit
fi

DOCKER_REG="secretflow"

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
        -r|--reg)
            DOCKER_REG="$2"
            shift
            if [[ "$#" -eq 0 ]]; then
                echo "Docker reg shall not be empty."
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
        -u|--upload)
            UPLOAD=1
            shift
        ;;
        -t|--tar)
            CREATE_TAR=1
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

IMAGE_NAME=sf-dev-ubuntu:${VERSION}
IMAGE_TAG=${DOCKER_REG}/${IMAGE_NAME}
LATEST_TAG=${DOCKER_REG}/sf-dev-ubuntu:latest
echo -e "Building ${GREEN}${IMAGE_TAG}${NO_COLOR}"
(cd ../.. && rm -rf dist/ build/)

docker run -it --rm -e SF_BUILD_DOCKER_NAME=${IMAGE_NAME} --mount type=bind,source="$(pwd)/../../../secretflow",target=/home/admin/src -w /home/admin --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --cap-add=NET_ADMIN --privileged=true secretflow/release-ci:latest /home/admin/src/docker/dev/entry.sh

(cd ../ && cp -r release/.nsjail dev/ && cp release/.condarc dev/ && cp *.yml dev/)
docker build . -f Dockerfile -t ${IMAGE_TAG} --build-arg config_templates="$(cat config_templates.yml)" --build-arg deploy_templates="$(cat deploy_templates.yml)"
echo -e "Finish building ${GREEN}${IMAGE_TAG}${NO_COLOR}"

# Save the image as a tar file if the option is set
if [[ ${CREATE_TAR} -eq 1 ]]; then
    TAR_NAME="sf-dev-ubuntu-${VERSION}.tar"
    echo -e "Saving image as ${GREEN}${TAR_NAME}${NO_COLOR}"
    docker save -o ${TAR_NAME} ${IMAGE_TAG}
fi

rm -rf .nsjail
rm -f .condarc
rm -f *.whl
rm -f *.yml
if [[ UPLOAD -eq 1 ]]; then
    docker push ${IMAGE_TAG}
fi


if [[ LATEST -eq 1 ]]; then
    echo -e "Tag and push ${GREEN}${LATEST_TAG}${NO_COLOR} ..."
    docker tag ${IMAGE_TAG} ${LATEST_TAG}
    if [[ UPLOAD -eq 1 ]]; then
        docker push ${LATEST_TAG}
    fi
fi