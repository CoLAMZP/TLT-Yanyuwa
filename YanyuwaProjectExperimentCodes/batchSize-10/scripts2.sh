
python -u main_code1.py --num_epochs 150 --task pos --learning_rate 0.00001 --data_file data10.xlsx --active_num 10 --run_ix bert_ner_1.0 --model_name bert-base-multilingual-cased --mtr 2   >mbert_2_pos2.txt

python -u main_code1.py --num_epochs 150 --task ner --learning_rate 0.00001 --data_file data10.xlsx --active_num 10 --run_ix bert_ner_1.0 --model_name bert-base-cased --mtr 2   >bert_2_ner2.txt