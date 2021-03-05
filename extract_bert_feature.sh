bert_model_dir=$1
text_file=$2
# feat_name cannot include /
feat_name=$3

echo "text lines: $(wc -l "${text_file}")"

cut -d " " -f 1 < "${text_file}" > line_name.tmp
cut -d " " -f 2- < "${text_file}" > pure_text.tmp

# bert config
bert_layers="-1"
max_seq_len="128"
batch_size="16"
echo "bert model_dir: ${bert_model_dir}, layers: ${bert_layers}, max_seq_len: ${max_seq_len}, batch: ${batch_size}"
echo "extracting features..."
python3 extract_features.py \
    --input_file pure_text.tmp \
    --output_file "${feat_name}.jsonl" \
    --vocab_file "${bert_model_dir}/vocab.txt" \
    --bert_config_file "${bert_model_dir}/bert_config.json" \
    --init_checkpoint "${bert_model_dir}/bert_model.ckpt" \
    --layers=${bert_layers} \
    --max_seq_length=${max_seq_len} \
    --batch_size=${batch_size}

echo "converting jsonl file to ark..."
python3 jsonl2ark.py \
    --jsonl "${feat_name}.jsonl" \
    --lid line_name.tmp \
    --out_name "${feat_name}"

for x in line_name.tmp pure_text.tmp; do
    if [ -f "${x}" ]; then rm ${x}; fi
done

echo "done"


