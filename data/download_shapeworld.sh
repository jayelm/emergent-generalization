#!/usr/bin/env bash

for fname in shapeworld.tar.gz shapeworld_ref.tar.gz shapeworld_all.tar.gz; do
    echo "$fname"
    wget "http://nlp.stanford.edu/data/muj/emergent-generalization/shapeworld/$fname"
    tar -xzf "$fname"
    rm "$fname"
done
