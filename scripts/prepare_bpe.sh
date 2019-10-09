# Prepare BPE
for lang in af nso ts
do
    python baseline/subwords.py train \
    --model_prefix assignment2/data/en_${lang}/subwords \
    --vocab_size 8000 \
    --model_type bpe \
    --input assignment2/data/en_$lang/en${lang}_parallel.train.$lang,assignment2/data/en_$lang/en${lang}_parallel.train.en
done
# Apply BPE
for lang in af nso ts
do
    for split in train dev test
    do
        for l in $lang en
        do
            python baseline/subwords.py segment \
            --model assignment2/data/en_${lang}/subwords.model \
            < assignment2/data/en_$lang/en${lang}_parallel.$split.$l \
            > assignment2/data/en_$lang/en${lang}_parallel.bpe.$split.$l
        done
    done
done