import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from torchtext import data
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from transformers import XLMRobertaModel
from models.MultiHead_BiRU import MultiheadBIRUAttention
from models.bert_cnn_model import BERTCNNSentiment
from transformers import XLMRobertaTokenizer

SEED = 6548
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
CUDA_LAUNCH_BLOCKING = 1

max_input_length = 64
BATCH_SIZE = 128
train_name = "training.csv"
test_name = "test.csv"
val_name = "validation.csv"
model_save_names = ['./checkpoint/modelA.txt', "./checkpoint/model2.txt" ]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
print('XLM Roberta Tokenizer Loaded...')
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id
print("Max input length: %d" %(max_input_length))

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens

UID=data.Field(
    sequential=False,
    use_vocab=False,
    pad_token=None
    )

TEXT=data.Field(batch_first=True,
                  use_vocab=False,
                  tokenize=tokenize_and_cut,
                  preprocessing=tokenizer.convert_tokens_to_ids,
                  init_token=init_token_idx,
                  eos_token=eos_token_idx,
                  pad_token=pad_token_idx,
                  unk_token=unk_token_idx
                  )
LABEL=data.LabelField()

fields = [('id',UID),('text', TEXT),('category', LABEL)]
train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path='./',
                                        train=train_name,
                                        test=test_name,
                                        validation=val_name,
                                        format='csv',
                                        fields=fields,
                                        skip_header=True)

print('Data loading complete')
print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of test examples: {len(test_data)}")

LABEL.build_vocab(train_data, valid_data)
train_iterator, valid_iterator, test_iterator=data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort_key=lambda x: len(x.text), 
    batch_size = BATCH_SIZE,
    device = device,
    )
bert = XLMRobertaModel.from_pretrained('xlm-roberta-base')

OUTPUT_DIM = 5
DROPOUT = 0.3
N_FILTERS = 100
FILTER_SIZES = [2, 3, 5, 7, 9]

model_names = ["A", "B"]

models = [
    BERTCNNSentiment(bert, OUTPUT_DIM, DROPOUT, N_FILTERS, FILTER_SIZES),
    MultiheadBIRUAttention(bert, 128, 100, 100, 4, 768, 5)
]
          
def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

optimizers = [optim.Adam(models[0].parameters()), optim.Adam(models[1].parameters())]
criterion = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()
log_softmax = nn.LogSoftmax()

for i in range(2):
    models[i] = models[i].to(device)

criterion=criterion.to(device)
nll_loss=nll_loss.to(device)
log_softmax=log_softmax.to(device)

def categorical_accuracy(preds, y):
    device = y.device
    count0,count1,count2,count3,count4 = torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1)
    count0,count1,count2,count3,count4 = count0.to(device) ,count1.to(device) ,count2.to(device) ,count3.to(device) ,count4.to(device) 
    total0,total1,total2,total3,total4 = torch.FloatTensor(1),torch.FloatTensor(1),torch.FloatTensor(1),torch.FloatTensor(1),torch.FloatTensor(1)
    max_preds = preds.argmax(dim = 1, keepdim = True) 
    correct = max_preds.squeeze(1).eq(y)
    predictions = max_preds.squeeze(1)
    true_correct = [0,0,0,0,0]
    for j,i in enumerate(y.cpu().numpy()):
        true_correct[y.cpu().numpy()[j]]+=1
        if i==0:
            count0+=correct[j]
            total0+=1
        elif i==1:
            count1+=correct[j]
            total1+=1
        elif i==2:
            count2+=correct[j]
            total2+=1
        elif i==3:
            count3+=correct[j]
            total3+=1
        elif i==4:
            count4+=correct[j]
        else:
            total4+=1
    metric=torch.FloatTensor([count0/true_correct[0],count1/true_correct[1],count2/true_correct[2],count3/true_correct[3],count4/true_correct[4],f1_score(y.cpu().numpy(),predictions.cpu().numpy(),average='weighted')])
    metric[metric!=metric] = 0
    hi4 = correct.sum()
    hi5 = torch.FloatTensor([y.shape[0]]).to(device)
    hi1 =  hi4 / hi5
    hi2 = metric
    hi3 = confusion_matrix(y.cpu().numpy(),max_preds.cpu().numpy(),labels=[0,1,2,3,4])
    return hi1,hi2,hi3

def train(model, iterator, optimizer, criterion, i):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        if (i == 0):
            predictions =  model(batch.text).squeeze(1)
        else:
            predictions =  model(batch.text, batch_size = len(batch)).squeeze(1)
        loss = criterion(predictions, batch.category)
        acc,_,_ = categorical_accuracy(predictions, batch.category)
        loss.backward()
        clip_gradient(model, 1e-1)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, i):
    epoch_loss = 0
    epoch_acc = 0
    epoch_all_acc = torch.FloatTensor([0,0,0,0,0,0])
    confusion_mat = torch.zeros((5,5))
    confusion_mat_temp = torch.zeros((5,5))
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            if (i == 0):
                predictions = model(batch.text).squeeze(1)
            else:
                predictions = model(batch.text,batch_size=len(batch)).squeeze(1)   
            loss = criterion(predictions, batch.category)
            acc,all_acc,confusion_mat_temp = categorical_accuracy(predictions, batch.category)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_all_acc += all_acc
            confusion_mat+=confusion_mat_temp
    return epoch_loss / len(iterator), epoch_acc / len(iterator),epoch_all_acc/len(iterator),confusion_mat

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 60
best_f1 = [-1, -1]
t = trange(N_EPOCHS)

for epoch in t:
    t.set_description('EPOCH %i' % epoch)
    for i in range(2):
        print(model_names[i])
        start_time = time.time()
        train_loss, train_acc = train(models[i], train_iterator, optimizers[i], criterion, i)
        valid_loss, valid_acc,tot,conf = evaluate(models[i], valid_iterator, criterion, i)
        f1 = tot[5]
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if f1 > best_f1[i]:
            best_f1[i] = f1   
            path = model_save_names[i]
            print(path)
            torch.save(models[i].state_dict(), path)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print("Validation F1 : ", f1)
 
