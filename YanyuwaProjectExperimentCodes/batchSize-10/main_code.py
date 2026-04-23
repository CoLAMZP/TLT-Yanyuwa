import ast, random
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification, AutoTokenizer 
from huggingface_hub import login
import torch
import evaluate
import shutil, os,time
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, auc,precision_score, f1_score, accuracy_score, recall_score
import datetime
from scipy.spatial import distance
import argparse, pickle
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics import multilabel_confusion_matrix #NEW
from sklearn.metrics import ConfusionMatrixDisplay #NEW
import matplotlib.pyplot as plt #NEW

def get_sbert_embeddings(input_string,sb):
    embeddings = sb.encode([input_string])
    return embeddings.flatten()
    
    
    
def get_top_diversity_session(data2):
    
    sess = data2['session_id']
    embs = [sess_emb_dict[s] for s in sess]

    kmeans = KMeans(n_clusters = min(active_num, len(embs)), random_state = 43)
    kmeans.fit(embs)
    predicted_clusters = kmeans.predict(embs)
    
    emb_cluster = list(zip(embs, predicted_clusters))
    sess_cluster = list(zip(sess, predicted_clusters))
    
    centers = kmeans.cluster_centers_
    #closest_points = []
    closest_sess = []

    for i, center in enumerate(centers):

        cluster_embs = [e for e, c in emb_cluster if c == i]
        cluster_sess = [s for s, c in sess_cluster if c == i]

        distances = np.linalg.norm(cluster_embs - center, axis = 1)
        closest_index = np.argmin(distances)
        #closest_points.append(cluster_embs[closest_index])
        closest_sess.append(cluster_sess[closest_index])
        
    #print('yes you can return')
        
    return closest_sess
    
    
    
    
def get_top_length_session(data2):
    
    sess = data2['session_id']
    lens = data2['token_count']
    res = list(zip(sess,lens))
    top_session = sorted( res, key = lambda x: x[1], reverse = True )
    
    return [a for a, b in top_session]
    
    
    
#NEW
def display_confusion_matrices_text(conf_matrices, labels):
    for i, matrix in enumerate(conf_matrices):
        print(f"\nConfusion Matrix for Label: '{labels[i]}'")
        print("                 Predicted")
        print("                0           1")
        print(f"True 0   {matrix[0,0]}         {matrix[0,1]}")
        print(f"     1   {matrix[1,0]}         {matrix[1,1]}")

    
    

    
def Active_Selection(model, data):

    global active_word_set
    model.eval()
    print('--------Active_Selection: {}--------'.format(str(datetime.datetime.now())[:19]))
    t1 = time.time()
    
    def help_func(example, top_session):
        if example['train_ix']=='train' and example['initial_flag']==0 and example['session_id'] in top_session:
            return 1
        else:
            return example['initial_flag']  
    
    prompt_data = data[(data.train_ix == 'train') & (data.initial_flag == 0)]

    if len(prompt_data) == 0:
        print('No unlabeled data is found, the program will now stop.')
        raise SystemExit
        
        
    if mtr not in ('-2','-3'):
        all_sessions, all_essays, all_labels, all_detail_labels, new_all_essays, all_sents = get_sessions_essays_brys(prompt_data)  
        sess_uncertainty_pair = []
             
        for sess, essay in zip(all_sessions, all_essays):
            if diversity_or_uncertainty == 'uncertainty':
                essay_uncertainty, real_lb = predict_label_for_one_instance(model, essay, tokenizer)
                sess_uncertainty_pair.append((sess, essay_uncertainty))
            elif diversity_or_uncertainty == 'diversity':    
                ess = [e for e in ast.literal_eval(essay[0]) if len(e) >= 2]
                unseen_word_ratio_of_ess = np.random.random() * 0.02 + len( [e.lower() for e in ess if e.lower() not in active_word_set] ) * 1.0 / len( ess ) if len(ess)>=1 else np.random.random() -0.4
                sess_uncertainty_pair.append((sess, unseen_word_ratio_of_ess))
                
                
        if mtr == '0':   # random selection as a baseline
            top_uncertainty_pair = sorted(sess_uncertainty_pair, key = lambda x: np.random.random(), reverse = True)[:active_num]
        else:         
            top_uncertainty_pair = sorted(sess_uncertainty_pair, key = lambda x: x[1], reverse = True)[:active_num]
        top_session = [s for s,b in top_uncertainty_pair]
        
        
    else:
        if mtr == '-3':
            top_session = get_top_length_session(prompt_data)[:active_num]      
        elif mtr =='-2':
            top_session = get_top_diversity_session(prompt_data)[:active_num]
    
        
    data['initial_flag'] = data.apply(lambda x: help_func(x, top_session), axis = 1)
    
      
    active_word_set = []

    #[ast.literal_eval() for e in test_data['tokens']]

    ll = [ast.literal_eval(e.replace("""''‘.'""", """'‘‘.'""").replace("""'','""", """'’,'""").replace("""'!''""", """'!’'""").replace("""'?''""","""'?’'""").replace("""'?','""", """'?’,'""").replace("''.'", "'‘.'").replace("'.''", "'.‘'").replace("'''",""" "'" """.strip()).replace("""''‘.'""", """'.'""")) for e in data[(data.train_ix == 'train') & (data.initial_flag == 1)]['tokens'] ]  
    for e in ll:
        active_word_set += [element for element in e if len(element) >= 2]
    active_word_set = [e.lower() for e in active_word_set]
    active_word_set = set(active_word_set)
    
    print('--------Ending Active_Selection: {}--------'.format(str(datetime.datetime.now())[:19]))
    return data
 




def uncertain_metric(logits):
    if type(logits) != type([]):
        logits = logits[0][1:-1]
        # mtr == '-1' means BALD
        if mtr =='0': # 0 means random_Active_Learning
            #print('++-+-+-+-+-+-+-+-+--+++-+-++-+-+-++--+-++-+-+-+-+-+-+-+-')
            return np.random.random()
        elif mtr == '1':      
        # metric 1 minmax
            logits = F.softmax(logits, dim = 1)
            res = 1.0 - torch.min(torch.max(logits, dim = 1).values).item()
            return res
            
        elif mtr =='2':
        # metric 2 avgmax
            logits = F.softmax(logits, dim = 1)
            res = 1.0 - torch.mean(torch.max(logits, dim = 1).values).item()
            return res
            
        elif mtr == '3':
        # metric 2 TokenEntropy
            logits = F.softmax(logits, dim = 1)
            p_logp = -logits * torch.log(logits + 1e-10)
            sum_p_logp = torch.sum(p_logp, dim = 1)
            sample_p_logp = torch.sum(sum_p_logp, dim = 0)
            TokenEntropy = sample_p_logp.item() / sum_p_logp.shape[0]
            return TokenEntropy
        elif mtr == '4':  
        # metric 3 N*  TokenEntropy
            logits = F.softmax(logits, dim = 1)
            p_logp = -logits * torch.log(logits + 1e-10)
            sum_p_logp = torch.sum(p_logp, dim = 1)
            sample_p_logp = torch.sum(sum_p_logp, dim = 0)
            TokenEntropy = sample_p_logp.item()
            return TokenEntropy
            
    else:  # mtr== -1 --> BLAD
        dropout_logits = [lg[0][1:-1] for lg in logits]   
        dropout_logits = [F.softmax(lg, dim = 1) for lg in dropout_logits]
        dropout_logits = torch.stack(dropout_logits)
        avg_logits = torch.mean(dropout_logits, dim = 0)

        part1_p_logp = -avg_logits * torch.log(avg_logits + 1e-10)
        part1_sum_p_logp = torch.sum(part1_p_logp, dim = 1)

        part2_p_logp = dropout_logits * torch.log(dropout_logits + 1e-10) * 0.25
        part2_sum_p_logp = torch.sum(part2_p_logp, dim = [0, 2])

        bald_logits = part1_sum_p_logp + part2_sum_p_logp
        sample_bald_uncertainty = torch.mean(bald_logits, dim=0).item()
        
        return sample_bald_uncertainty
        
        
            
        
    

    
def get_n_bert_examples(data):    # checked  2024-06-16

    active_sessions = data[(data.train_ix == 'train') & (data.initial_flag == 1)]['session_id']
    def get_one_single_bert_example(data, sess):                # checked    2024-06-16       
        random_record = data[(data.session_id == sess)].sample(n = 1).to_dict(orient='records')   
        text_lbl=[(  ast.literal_eval(e['tokens']), ast.literal_eval(e[tags_string])) for e in random_record]
        text_combined = text_lbl[0][0]
        lbl = text_lbl[0][1]
        
        example = (text_combined, lbl)
        return example
        
    res = []
    for sess in active_sessions:
        res.append( get_one_single_bert_example(data, sess) )
        
    random.shuffle(res)
        
    return res 



def tokenize_and_align_labels(examples):              # checked  2024-06-16
    tokenized_inputs = tokenizer(examples["tokens"], truncation = True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[tags_string]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx]) # refer to this example of an label list:[None, 0, 1, 2, 3, 3, 3, 4, None] ## generated by huggingface tokenizer
            else:
                label_ids.append(-100)  # This is not the start of a word, we don't need a label for it. thus -100.
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_one_bert_epoch_data(data, tokenizer): # checked   2024-06-16
    
    d = get_n_bert_examples(data = data) 
    epoch_df = pd.DataFrame()
    essay_series, lbl_series= pd.Series(  [sent for sent, lbl in d] ), pd.Series(  [lbl for sent, lbl in d] )
    epoch_df['tokens'] = essay_series
    epoch_df[tags_string] = lbl_series

    dataset = Dataset.from_pandas(epoch_df)
    
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched = True)

    cols_to_remove = [col for col in tokenized_datasets.features.keys() if col not in ('attention_mask', 'input_ids', 'token_type_ids', 'labels')]
    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)
    #tokenized_datasets.set_format("torch")
    dataloader = DataLoader(tokenized_datasets, shuffle = True, batch_size = 1)
    
    return dataloader 

def predict_label_for_one_instance(model, sents, tokenizer): # checked   2024-06-16
    # 只需要 real_predictions     
    sent_loader = convert_sents_to_loader(sents, tokenizer)
    sent_predictions = []
    real_predictions = []
    
    original_tokens = []
    word_ids = []
    uncertainty_list = []

    for batch in sent_loader:#.dataset:
        with torch.no_grad():
            if mtr != '-1':
                outputs = model(**batch)  
                uncertainty_list.append( uncertain_metric(outputs.logits) )
            elif mtr == '-1':
                dropout_outputs = [] 
                model.train()
                for _ in range(4):
                    outputs = model(**batch) 
                    dropout_outputs.append(outputs.logits)
                model.eval()
                uncertainty_list.append( uncertain_metric(dropout_outputs) )
                    

        predictions = torch.argmax(outputs.logits, dim =- 1)#.item()
        predictions = torch.flatten(predictions).tolist()
        predicted_token_class = [model.config.id2label[t] for t in predictions ]
        tokens = tokenizer.convert_ids_to_tokens(torch.flatten(batch["input_ids"]).tolist())
        word_id = batch.word_ids()       

        res_pair = dict()
        for ix, e in enumerate(word_id):
            if e not in res_pair and e is not None:
                res_pair[e] = ix
        real_pred = [predictions[v] for k,v in res_pair.items()]  
            
        real_predictions.append(real_pred)
        original_tokens.append(tokens)
        word_ids.append( word_id )
        sent_predictions.append( predictions )        
        
    return uncertainty_list, real_predictions



def convert_sents_to_loader(tokens, tokenizer):  # checked 2024-06-16
    
    tok = [ast.literal_eval(e) for e in tokens]
    res = []
    for t in tok:
        inputs = tokenizer(t, return_tensors = "pt", is_split_into_words = True).to(device)     
        res.append(inputs)
        
    return res
        



def get_sessions_essays_brys(df):   # checked 2024-06-16   # all_sessions 跟 all_essay 是匹配的， <sessin_1, [sent_1]>
    # 只需要 new_all_labels这个output  ---->  [[2,2,1,0],[3,4,1,0],[2,0]], 也就是需要 output 的第 四 个term
    all_essays = []
    all_brys = []
    all_labels = []
    all_sessions = list(df['session_id'])
    all_sents = []

    res = []
    for e in all_sessions:
        if e not in res:
            res.append(e)
    all_sessions = res    

    for session in all_sessions:
        single_essay = df[df.session_id == session].sort_values(by = 'sent_id', ascending = True)['tokens'].tolist()
        single_sents = df[df.session_id == session].sort_values(by = 'sent_id', ascending = True)['sent_text'].tolist()
        single_label = df[df.session_id == session].sort_values(by = 'sent_id', ascending = True)[tags_string].tolist()
     
        all_essays.append(single_essay)
        all_sents.append(single_sents) 
        all_labels.append(single_label)   #---> single_label是单篇essay的word级别labels --> [[2,2,1,0],[3,4,1,0],[2,0]]
          
    new_all_labels = []
    new_all_essays = []
    for labels in all_labels:
        tmp = [ast.literal_eval(e) for e in labels]
        new_all_labels.append(tmp)
           
        
    new_all_essays = []
    for essay in all_essays:
        tmp = [ast.literal_eval(e) for e in essay]
        new_all_essays.append(tmp)
                          
    return all_sessions, all_essays, [], new_all_labels, new_all_essays, all_sents




def get_evaluation_result(predicted, lbls, all_essays):  # checked 2024-06-16
    
    def flatten(t):
        while type(t[0]) == type([1]):
            tmp = []
            for e in t:
                tmp += e
            t = tmp
        return t
    
    predicted = flatten(predicted)
    lbls = flatten(lbls)
    all_essays = flatten(all_essays)
    all_essays_masked = [0 if w.lower() in train_word_set else 1 for w in all_essays] 
    
    filtered_lbls=[]
    filtered_predicted = []
    
    for l, p, c in zip(lbls, predicted, all_essays_masked):
        if c == 1:
            filtered_lbls.append(l)
            filtered_predicted.append(p) 

    # NEW - Multi
    #combined_conf_matrix = confusion_matrix(lbls, predicted)
    
    # NEW
    return round(accuracy_score(lbls, predicted), 3), round(precision_score(lbls, predicted, average = 'macro' ), 3), round(recall_score(lbls, predicted, average = 'macro'), 3), round(f1_score(lbls, predicted, average = 'macro'), 3), multilabel_confusion_matrix(lbls, predicted), round(accuracy_score(filtered_lbls, filtered_predicted), 3), round(precision_score(filtered_lbls, filtered_predicted, average = 'macro' ), 3), round(recall_score(filtered_lbls, filtered_predicted, average = 'macro'), 3), round(f1_score(filtered_lbls, filtered_predicted, average = 'macro'), 3)



    
"""

def get_evaluation_result(predicted, lbls):  # checked 2024-06-16
    
    tpl = list(zip(predicted, lbls))
    all_acc = []
    all_pred = []
    all_rec = []
    all_f1 = []
      
    for pp, ll in tpl:
        p = pp[0]
        l = ll[0]
        
        all_acc.append(round(accuracy_score(l, p), 3))       
        all_pred.append(round(precision_score(l, p, average = 'macro' ), 3))   
        all_rec.append(round(recall_score(l, p, average = 'macro'), 3))
        all_f1.append(round(f1_score(l, p, average = 'macro'), 3))
    
    return -1.0, round( np.mean(all_acc), 3), round( np.mean(all_pred), 3), round( np.mean(all_rec), 3), round( np.mean(all_f1), 3)

"""




def evaluate_NER_model(model, data, test_type): # checked 2024-06-16
    model.eval()
    print('--------Start Evaluating: {}--------'.format(str(datetime.datetime.now())[:19]))
    t1 = time.time()
    if test_type == 'valid':
        
        
        prompt_data = data[
            (data.train_ix == 'valid') #& (data.sentence_source != 'prompt')
        ]
        
        
    elif test_type == 'test':  
        prompt_data = data[
            (data.train_ix == 'test') #& (data.sentence_source != 'prompt')
        ]
             
        
    all_predicted_labels = []
    all_predicted_real_labels = []
    all_sessions, all_essays, all_labels, all_detail_labels, new_all_essays, all_sents = get_sessions_essays_brys(prompt_data)
    #all_ranked_brys = []
    for essay in all_essays:
        lb,real_lb = predict_label_for_one_instance(model, essay, tokenizer)
        all_predicted_labels.append(lb)
        all_predicted_real_labels.append(real_lb)
        
        
    acc, precision, recall, f1_score, conf_matrices, filtered_acc, filtered_precision, filtered_recall, filtered_f1_score = get_evaluation_result(all_predicted_real_labels, all_detail_labels, all_essays)
    
    print('--------------------{}ing------------------------'.format(test_type))
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('f1_score: {}'.format(f1_score))

    # NEW
    print("Confusion Matrices:")
    print(conf_matrices)
    label_names = list(model.config.label2id.keys())  
    display_confusion_matrices_text(conf_matrices, label_names)
    
    #print('Unseen_Accuracy: {}'.format(filtered_acc))
    #print('Unseen_Precision: {}'.format(filtered_precision))
    #print('Unseen_Recall: {}'.format(filtered_recall))
    #print('Unseen_f1_score: {}'.format(filtered_f1_score))    

        
    print('--------Ending Evaluating: {}--------'.format(str(datetime.datetime.now())[:19]))
        
    return f1_score



def train_one_epoch(model, train_loader, tokenizer, learning_rate):#checked
    
    model.train()
    try:
        os.makedirs('model')
    except:
        pass
    training_args = TrainingArguments(
                output_dir = "model",
                learning_rate = learning_rate,
                per_device_train_batch_size = 1,
                per_device_eval_batch_size = 1,
                num_train_epochs = 1,
                weight_decay = 0.01,
                evaluation_strategy = "no",
                save_strategy = "no",
                load_best_model_at_end = False,
                push_to_hub = False,
                disable_tqdm=False,
                gradient_accumulation_steps = 16,
                lr_scheduler_type='constant'
                #remove_unused_columns=False
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_loader.dataset,
        #eval_dataset = train_data.dataset,
        tokenizer = tokenizer,
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        #compute_metrics = compute_metrics
    )



    trainer.train()
    time.sleep(1)
    shutil.rmtree('model')




def train_model(model, data, lr, num_epochs, tokenizer): #checked
    
    
    for i in range(num_epochs):
        
        train_loader = get_one_bert_epoch_data(data, tokenizer)  #train_loader.dataset

        print('-------start training epoch {}-time:{}------'.format(i+1,str(datetime.datetime.now())[:19]))
        train_one_epoch(model, train_loader, tokenizer, learning_rate = lr)
        lr = lr * 0.998
    
        print('--------now evaluating on valid set----------')
        valid_performance = evaluate_NER_model(model, data, test_type = 'test')  
        print('Number of active samples: ', len(data[data.initial_flag == 1]))
        data = Active_Selection(model, data)   # Selecting the highest informative samples into the active training set
        # So that in next epoch, we will be training on the updated (expanded) training set.
        print('Number of words in Active_word_set', len(active_word_set))
        



if __name__ == "__main__":
      
    login(token='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')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seqeval = evaluate.load("seqeval")

    
    parser = argparse.ArgumentParser(description="This is a description")
    parser.add_argument('--num_epochs',dest='num_epochs',required = True,type = int)
    parser.add_argument('--active_num', dest='active_num', required = True, type = int)
    parser.add_argument('--mtr', dest='mtr',required = True,type = str)
    parser.add_argument('--diversity_or_uncertainty', dest='diversity_or_uncertainty',required = False,type = str)
    parser.add_argument('--task',dest='task',required = True,type = str)
    parser.add_argument('--learning_rate',dest='learning_rate',required = True,type = float) 
    parser.add_argument('--data_file', dest='data_file', required = True, type = str)  
    parser.add_argument('--model_name', dest='model_name', required = True, type = str) 
    parser.add_argument('--run_ix',dest='run_ix', required = True,type = str)
    args = parser.parse_args()  
    
    
    num_epochs = args.num_epochs
    mtr = args.mtr 
    diversity_or_uncertainty = 'uncertainty'#args.diversity_or_uncertainty
    lr = args.learning_rate
    active_num = args.active_num
    task = args.task.lower()   # only two options --> pos and ner
    data_file = args.data_file
    run_ix = args.run_ix
    model_name = args.model_name #'distilbert-base-uncased'
    
    
    print('num_epochs: {}'.format(num_epochs))
    print('mtr: {}'.format(mtr))
    print('diversity_or_uncertainty: {}'.format(diversity_or_uncertainty))
    print('learning_rate: {}'.format(lr))
    print('task: {}'.format(task))
    print('data_file: {}'.format(data_file))
    print('active_num: {}'.format(active_num))
    print('model_name: {}'.format(model_name))
    print('run_ix: {}'.format(run_ix))
    
    

    data = pd.read_excel(data_file)
    ll = [ast.literal_eval(e.replace("""''‘.'""", """'‘‘.'""").replace("""'','""", """'’,'""").replace("""'!''""", """'!’'""").replace("""'?''""","""'?’'""").replace("""'?','""", """'?’,'""").replace("''.'", "'‘.'").replace("'.''", "'.‘'").replace("'''",""" "'" """.strip()).replace("""''‘.'""", """'.'""")) for e in data[data.train_ix == 'train']['tokens'] ]
    train_word_set = []
    for e in ll:
        train_word_set += e
    train_word_set = [e.lower() for e in train_word_set]
    train_word_set = set(train_word_set)
    
    
    # Words appeared at the active words set; # 一个初始的 active word set
    ll = [ast.literal_eval(e.replace("""''‘.'""", """'‘‘.'""").replace("""'','""", """'’,'""").replace("""'!''""", """'!’'""").replace("""'?''""","""'?’'""").replace("""'?','""", """'?’,'""").replace("''.'", "'‘.'").replace("'.''", "'.‘'").replace("'''",""" "'" """.strip()).replace("""''‘.'""", """'.'""")) for e in data[(data.train_ix == 'train') & (data.initial_flag == 1)]['tokens'] ]
    active_word_set = []
    for e in ll:
        active_word_set += [element for element in e if len(element) >= 2]
    active_word_set = [e.lower() for e in active_word_set]
    active_word_set = set(active_word_set)
    
    
    # create a session_embedding_dict
    if mtr == '-2':
        sb = SentenceTransformer('all-mpnet-base-v2')
        
        sess = data['session_id']
        sents = data['sent_text']
        embs = []
        for _,sent in zip(sess,sents):
            embs.append( get_sbert_embeddings(sent, sb) )
        sess_emb_dict = dict(zip(sess, embs))
        
    
    
    
    if model_name=='roberta-base':
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name) 
    if task.lower()=='pos':
        tags_string = 'pos_tags'
        label2id = {'ADJ':0,
             'ADP':1,
             'ADV':2,
             'AUX':3,
             'CCONJ':4,
             'DET':5,
             'INTJ':6,
             'NOUN':7,
             'NUM':8,
             'PRON':9,
             'PROPN':10,
             'RPOPN':11,
             'SCONJ':12,
             'VERB':13,
             'X':14
        }
        id2label = {v:k for k,v in label2id.items()}
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = 15, id2label = id2label, label2id = label2id)
    else:
        tags_string = 'ner_tags'
        label2id = {'AM':0,
             'KS':1,
             'L':2,
             'P':3,
             'PT':4,
             '0':5
           }
        id2label = {v:k for k,v in label2id.items()}
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = 6, id2label = id2label, label2id = label2id)
    
    model.to(device)   
    train_model(model, data, lr, num_epochs, tokenizer)
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    