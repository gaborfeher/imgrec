#!/bin/bash -e

mkdir -p deps
pushd deps

GOOGLE_TEST="googletest-release-1.8.0"
curl -o "${GOOGLE_TEST}.tar.gz" https://codeload.github.com/google/googletest/tar.gz/release-1.8.0

tar xf "${GOOGLE_TEST}.tar.gz"
pushd $GOOGLE_TEST/googletest
cmake .

popd  # $GOOGLE_TEST

popd  # deps
