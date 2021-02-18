#!/bin/bash

cat | \
./usr-url.sh | \
./usr-url-start-end.sh | \
sed 's/@ usr/@usr/g' | \
sed 's/@ url/@url/g'