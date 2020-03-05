export BERT_BASE_DIR=/home/name/code/clones/bert/.models/multi_cased_L-12_H-768_A-12
export GLUE_DIR=/home/name/data/glue_data
export TRAINED_CLASSIFIER=/tmp/mrpc_output/


python run_classifier.py \
  --task_name=MRPC \
  --do_predict=false \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --do_lower_case=false \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
