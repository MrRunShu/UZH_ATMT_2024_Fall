#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd="$(dirname "$(readlink -f "$0")")"
base="$pwd/../.."
src=fr
tgt=en
data="$base/data/$tgt-$src/"

# change into base directory to ensure paths are valid
cd "$base"

# create BPE-specific directories
mkdir -p "$data/preprocessed_BPE_8000/"
mkdir -p "$data/prepared_BPE_8000/"

# normalize and tokenize raw data
cat "$data/raw/train.$src" | perl moses_scripts/normalize-punctuation.perl -l "$src" | perl moses_scripts/tokenizer.perl -l "$src" -a -q > "$data/preprocessed_BPE_8000/train.$src.p"
cat "$data/raw/train.$tgt" | perl moses_scripts/normalize-punctuation.perl -l "$tgt" | perl moses_scripts/tokenizer.perl -l "$tgt" -a -q > "$data/preprocessed_BPE_8000/train.$tgt.p"

# train truecase models
perl moses_scripts/train-truecaser.perl --model "$data/preprocessed_BPE_8000/tm.$src" --corpus "$data/preprocessed_BPE_8000/train.$src.p"
perl moses_scripts/train-truecaser.perl --model "$data/preprocessed_BPE_8000/tm.$tgt" --corpus "$data/preprocessed_BPE_8000/train.$tgt.p"

# apply truecase models to splits
cat "$data/preprocessed_BPE_8000/train.$src.p" | perl moses_scripts/truecase.perl --model "$data/preprocessed_BPE_8000/tm.$src" > "$data/preprocessed_BPE_8000/train.$src"
cat "$data/preprocessed_BPE_8000/train.$tgt.p" | perl moses_scripts/truecase.perl --model "$data/preprocessed_BPE_8000/tm.$tgt" > "$data/preprocessed_BPE_8000/train.$tgt"

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat "$data/raw/$split.$src" | perl moses_scripts/normalize-punctuation.perl -l "$src" | perl moses_scripts/tokenizer.perl -l "$src" -a -q | perl moses_scripts/truecase.perl --model "$data/preprocessed_BPE_8000/tm.$src" > "$data/preprocessed_BPE_8000/$split.$src"
    cat "$data/raw/$split.$tgt" | perl moses_scripts/normalize-punctuation.perl -l "$tgt" | perl moses_scripts/tokenizer.perl -l "$tgt" -a -q | perl moses_scripts/truecase.perl --model "$data/preprocessed_BPE_8000/tm.$tgt" > "$data/preprocessed_BPE_8000/$split.$tgt"
done

# Learn BPE on the combined truecased training data
subword-nmt learn-bpe -s 8000 < "$data/preprocessed_BPE_8000/train.$src" > "$data/preprocessed_BPE_8000/bpe.codes.$src"
subword-nmt learn-bpe -s 8000 < "$data/preprocessed_BPE_8000/train.$tgt" > "$data/preprocessed_BPE_8000/bpe.codes.$tgt"

# Apply BPE to each data split
for split in train valid test tiny_train; do
    echo "Data directory: $data/preprocessed_BPE_8000"
    echo "Applying BPE to train.$src: $data/preprocessed_BPE_8000/train.$src"
    subword-nmt apply-bpe -c "$data/preprocessed_BPE_8000/bpe.codes.$src" < "$data/preprocessed_BPE_8000/$split.$src" > "$data/preprocessed_BPE_8000/$split.bpe.$src"
    subword-nmt apply-bpe -c "$data/preprocessed_BPE_8000/bpe.codes.$tgt" < "$data/preprocessed_BPE_8000/$split.$tgt" > "$data/preprocessed_BPE_8000/$split.bpe.$tgt"
done

# remove tmp files
rm "$data/preprocessed_BPE_8000/train.$src.p"
rm "$data/preprocessed_BPE_8000/train.$tgt.p"

# preprocess all files for model training
#python preprocess.py --target-lang "$tgt" --source-lang "$src" --dest-dir "$data/prepared_BPE_8000/" --train-prefix "$data/preprocessed_BPE_8000/train.bpe" --valid-prefix "$data/preprocessed_BPE_8000/valid.bpe" --test-prefix "$data/preprocessed_BPE_8000/test.bpe" --tiny-train-prefix "$data/preprocessed_BPE_8000/tiny_train.bpe" --threshold-src 1 --threshold-tgt 1 --num-words-src 8000 --num-words-tgt 8000

echo "BPE Preprocessing completed successfully!"
