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
mkdir -p "$data/preprocessed_BPE/"
mkdir -p "$data/prepared_BPE/"

# normalize and tokenize raw data
cat "$data/raw/train.$src" | perl moses_scripts/normalize-punctuation.perl -l "$src" | perl moses_scripts/tokenizer.perl -l "$src" -a -q > "$data/preprocessed_BPE/train.$src.p"
cat "$data/raw/train.$tgt" | perl moses_scripts/normalize-punctuation.perl -l "$tgt" | perl moses_scripts/tokenizer.perl -l "$tgt" -a -q > "$data/preprocessed_BPE/train.$tgt.p"

# train truecase models
perl moses_scripts/train-truecaser.perl --model "$data/preprocessed_BPE/tm.$src" --corpus "$data/preprocessed_BPE/train.$src.p"
perl moses_scripts/train-truecaser.perl --model "$data/preprocessed_BPE/tm.$tgt" --corpus "$data/preprocessed_BPE/train.$tgt.p"

# apply truecase models to splits
cat "$data/preprocessed_BPE/train.$src.p" | perl moses_scripts/truecase.perl --model "$data/preprocessed_BPE/tm.$src" > "$data/preprocessed_BPE/train.$src"
cat "$data/preprocessed_BPE/train.$tgt.p" | perl moses_scripts/truecase.perl --model "$data/preprocessed_BPE/tm.$tgt" > "$data/preprocessed_BPE/train.$tgt"

# prepare remaining splits with learned models
for split in valid test tiny_train
do
    cat "$data/raw/$split.$src" | perl moses_scripts/normalize-punctuation.perl -l "$src" | perl moses_scripts/tokenizer.perl -l "$src" -a -q | perl moses_scripts/truecase.perl --model "$data/preprocessed_BPE/tm.$src" > "$data/preprocessed_BPE/$split.$src"
    cat "$data/raw/$split.$tgt" | perl moses_scripts/normalize-punctuation.perl -l "$tgt" | perl moses_scripts/tokenizer.perl -l "$tgt" -a -q | perl moses_scripts/truecase.perl --model "$data/preprocessed_BPE/tm.$tgt" > "$data/preprocessed_BPE/$split.$tgt"
done

# Learn BPE on the combined truecased training data
subword-nmt learn-bpe -s 4000 < "$data/preprocessed_BPE/train.$src" > "$data/preprocessed_BPE/bpe.codes.$src"
subword-nmt learn-bpe -s 4000 < "$data/preprocessed_BPE/train.$tgt" > "$data/preprocessed_BPE/bpe.codes.$tgt"

# Apply BPE to each data split
for split in train valid test tiny_train; do
    subword-nmt apply-bpe -c "$data/preprocessed_BPE/bpe.codes.$src" < "$data/preprocessed_BPE/$split.$src" > "$data/preprocessed_BPE/$split.bpe.$src"
    subword-nmt apply-bpe -c "$data/preprocessed_BPE/bpe.codes.$tgt" < "$data/preprocessed_BPE/$split.$tgt" > "$data/preprocessed_BPE/$split.bpe.$tgt"
done

# remove tmp files
rm "$data/preprocessed_BPE/train.$src.p"
rm "$data/preprocessed_BPE/train.$tgt.p"

# preprocess all files for model training
python preprocess.py --target-lang "$tgt" --source-lang "$src" --dest-dir "$data/prepared_BPE/" --train-prefix "$data/preprocessed_BPE/train.bpe" --valid-prefix "$data/preprocessed_BPE/valid.bpe" --test-prefix "$data/preprocessed_BPE/test.bpe" --tiny-train-prefix "$data/preprocessed_BPE/tiny_train.bpe" --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000

# For convenience, We decide to execute the last part on conda env with torch package installed
# python train.py --data "D:\UZH\2024Fall\advanced techniques of machine translation\assignments\assignment1\atmt_2024\data\en-fr\prepared_BPE" --source-lang fr --target-lang en --save-dir "D:\UZH\2024Fall\advanced techniques of machine translation\assignments\assignment1\atmt_2024\assignments\03\baseline\checkpoints\bpe\tiny" --train-on-tiny --batch-size 64

echo "BPE Preprocessing completed successfully!"
