import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import sklearn
import warnings
warnings.filterwarnings("ignore")
import re
import emoji
import nltk
from nltk.corpus import stopwords
# print(stopwords.words('english'))
cachedStopWords = stopwords.words("english")
new_stop_words = ["For","for","points","Points","this","name","identify","points","points "]
cachedStopWords.extend(new_stop_words)

def rem_stop_words(text):
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return text

def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def clean_str(string):
    string = string.strip().lower()
    string = rem_stop_words(string)
    string = decontracted(string)

    string = re.sub(r"<p>", " ", string)
    string = re.sub(r"</p>", " ", string)
    string = re.sub(r"\n", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def withoutduplicates(string):
    prev=-1
    chars=[]
    for c in range(len(string)):
        if (c==0):
            prev = c
            chars.append(string[c])
        else:
            if (string[prev] == string[c]):
                continue
            else:
                chars.append(string[c])
                prev = c
    return ''.join(chars)

def preprocess_sentences(tweets):
    for row_number, tweet in tweets.iterrows():
        twitt = clean_str(tweet['text'])
        twitt = re.sub('@USER', '', twitt)
        twitt = re.sub('HTTPURL', '', twitt)
        words = twitt.split()

        for word in words:
            if isinstance(word, str) and word.isnumeric():
                word1 = withoutduplicates(word)
                if(word1 != word):
                    twitt = re.sub(re.escape(word), lambda _: word1,twitt)
        tweets.at[row_number,'text']= emoji.demojize(twitt)


class StringConverter(dict):
    def __contains__(self, item):
        return True

    def __getitem__(self, item):
        return str

    def get(self, default=None):
        return str

df = pd.read_csv('toughness.txt', sep =  ':::')
df = df.dropna()
df = df[df.Label != None]
possible_labels = df.Label.unique()
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
df['label'] = df.Label.replace(label_dict)
df = df.groupby('Label').apply(lambda x: x.sample(n=5000)).reset_index(drop = True)
preprocess_sentences(df)
print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size=0.5, random_state=17,stratify=df.label.values)

df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'
print(df.groupby(['Label', 'label', 'data_type']).count())

import pickle
with open('label_dict.txt', 'wb') as fh:
   pickle.dump(label_dict, fh)

# !pip install transformers
from transformers import BertTokenizer, AlbertTokenizer, AlbertForSequenceClassification
from torch.utils.data import TensorDataset
# print(df[df.data_type=='train'].text.values[0:10])
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# print(type(df[df.data_type=='train'].text.values[0]),type("Raj"))
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=64,
    truncation=True,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=64,
    truncation=True,
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# from transformers import AlbertForSequenceClassification

model = AlbertForSequenceClassification.from_pretrained("albert-base-v2",num_labels=len(label_dict),output_attentions=False,output_hidden_states=False)
model.load_state_dict(torch.load('finetuned_ALBERT_epoch_2.model', map_location=torch.device('cpu')))
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
batch_size = 32

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

from transformers import AdamW, get_linear_schedule_with_warmup
optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  eps=1e-8)
epochs = 5
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)
import numpy as np
from sklearn.metrics import f1_score
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

import random
import numpy as np
seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

train_loss = []
validation_loss = []
#-------------------------------------------------------
for epoch in tqdm(range(1, epochs+1)):

    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


    torch.save(model.state_dict(), f'finetuned_ALBERT_epoch_{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')
    train_loss.append(loss_train_avg)
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    validation_loss.append(val_loss)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
# model.load_state_dict(torch.load('finetuned_BERT_epoch_3.model', map_location=torch.device('cpu')))
_, predictions, true_vals = evaluate(dataloader_validation)
accuracy_per_class(predictions, true_vals)
from sklearn.metrics import classification_report
preds_flat = np.argmax(predictions, axis=1).flatten()
labels_flat = true_vals.flatten()
print(classification_report(labels_flat, preds_flat))


import matplotlib.pyplot as plt
x_axis = list(range(epochs))
plt.plot(x_axis, train_loss,label = "train loss")
plt.plot(x_axis, validation_loss,label = "test loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.show()
