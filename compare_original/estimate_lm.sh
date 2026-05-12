#!/bin/bash

# loop over all .train files in output_train/ and estimate a 4-gram LM using KenLM, saving the ARPA file in the same directory
for file in output_train/*.train; do
    echo "Estimating LM for $file..."
    output_arpa="${file%.train}.arpa"
    docker run --rm -v $(pwd)/output_train:/data mpenagar/kenlm lmplz -o 4 --skip_symbols --discount_fallback --memory 20G --arpa /data/$(basename $output_arpa) --text /data/$(basename $file)

    # now make a binary file for faster loading
    # output_binary="${file%.train}.bin"
    # docker run --rm -v $(pwd)/output_train:/data mpenagar/ken lm bash -c \
    # "build_binary -a 255 -q 8 /data/$(basename $output_arpa) /data/$(basename $output_binary)"
done