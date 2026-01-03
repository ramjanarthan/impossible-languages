cd data_generation/outputs/impossible_blimp/v3
timestamp=$(date +%Y%m%d_%H%M%S)
for f in *; do
    if [[ -f "$f" ]]; then
        mv "$f" "${f%.*}_$timestamp.${f##*.}"
    fi
done