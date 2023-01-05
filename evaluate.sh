python source/evaluate.py \
--weights ./result/runs_human_attributes_2/best.pt \
--logfile human_attribute_evaluate.txt \
--data config/human_attribute_2/data_config.yaml \
--batch_size 128 \
--device cuda:0