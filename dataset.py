from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
# import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
# import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
import time
class TokenDictionary:
    def __init__(self):
        self.words = ['<padding>','<unk>']
        self.tokens = {'<padding>':0, '<unk>':1}
        self.cnt = 2

    def __len__(self):
        return len(self.tokens)

    # @property
    # def pad_token(self):
    #     return len(self.tokens)

    def cnt_word(self, name):
        questions = json.load(open(f'/content/v2_OpenEnded_mscoco_{name}2014_questions.json', 'r'))['questions']
        cnt_dict = {}
        for sentence in questions:
            sentence = sentence['question']
            words = sentence.lower().replace(',', '').replace('?', '').replace('\'s', '').split()
            for w in words:
                if w not in cnt_dict:
                    cnt_dict[w] = 1
                elif w not in self.tokens:
                    self.tokens[w] = self.cnt
                    self.cnt += 1
                    self.words.append(w)

    def tokenize(self, sentence):
        words = sentence.lower().replace(',', '').replace('?', '').replace('\'s', '').split()
        tokens = []
        for word in words:
            if word not in self.tokens:
                tokens.append(self.tokens['<unk>'])
            else:
                tokens.append(self.tokens[word])
        return tokens

    # def save(self, path):
    #     torch.save((self.tokens, self.words), path)
    #
    # def load(self, path):
    #     self.tokens, self.words = torch.load(path)

def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary,max_question_length=14):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']
        self.dictionary = dictionary
        self.entries = []
        self.load_data(name, max_question_length)

    def load_data(self,name, max_question_length):
        questions = json.load(open(f'/content/v2_OpenEnded_mscoco_{name}2014_questions.json', 'r'))['questions']
        answers = json.load(open(f'/content/v2_mscoco_{name}2014_annotations.json', 'r'))['annotations']
        all_answers = pickle.load(open(f'data/cache/{name}_target_new.pkl', 'rb'))
        questions = sorted(questions, key=lambda q: q['question_id'])
        answers = sorted(answers, key=lambda a: a['question_id'])
        ans2labels = pickle.load(open('data/cache/trainval_ans2label.pkl','rb'))
        self.dictionary.cnt_word(name)
        for question, answer in zip(questions,answers):
            assert question['question_id'] == answer['question_id']
            assert question['image_id'] == answer['image_id']
            image_id = question['image_id']
            q_tensor = torch.tensor(self.dictionary.tokenize(question['question']))
            if len(q_tensor) < max_question_length:
                q_tensor = torch.cat((torch.zeros(max_question_length - len(q_tensor)) ,q_tensor),0).long()
            if len(q_tensor) > max_question_length: #TODO
                q_tensor = q_tensor[:14]
            ans = all_answers[(question['question_id'], question['image_id'])]
            a_token = torch.zeros(len(ans2labels))
            if not ans: #we ignore unkown answers
                continue
            for key, score in ans.items():
                a_token[key] = score #a_token[ans2labels[a]] = score
            self.entries.append((name, image_id, q_tensor,question['question_id'] ,a_token))

    def tokenize(self, max_length=14):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        name = self.entries[index][0]
        image_id = self.entries[index][1]
        for i in range(10):
            try:
                image = Image.open(f'/content/{name}2014/COCO_{name}2014_{str(image_id).zfill(12)}.jpg').convert('RGB')
                # Resize
                resize = transforms.Resize(size=(299, 299)) #TODO change?
                image = resize(image)
                # this also divides by 255 TODO we can normalize too change?!?!
                image_tensor = TF.to_tensor(image)
            except:
                print('except')
                time.sleep(3)
                image_tensor = [0]
                continue
            tmp = image_tensor, image_id, self.entries[index][2], self.entries[index][3], self.entries[index][4]
            #tmp = image_tensor, image_id, q_tensor, question['question_id'], a_token
            return tmp

    def __len__(self):
        return len(self.entries)

# class VQADataset(data.Dataset):
#     def __init__(self, dictionary, validation=False, max_question_length=14):
#         super(VQADataset, self).__init__()
#         self.entries = []
#         self.dictionary = dictionary
#         self.load(validation, max_question_length)
#
#     def load(self, validation, max_question_length):
#         name = 'val' if validation else 'train'
#         images = f'data/{name}2014'
#         questions = json.load(open(f'data/v2_OpenEnded_mscoco_{name}2014_questions.json','r'))['questions']
#         annotations = json.load(open(f'data/v2_mscoco_{name}2014_annotations.json','r'))['annotations']
#         questions = sorted(questions, key=lambda q: q['question_id'])
#         annotations = sorted(annotations, key=lambda a: a['question_id'])
#         for question, annotation in zip(questions, annotations):
#             assert question['question_id'] == annotation['question_id']
#             assert question['image_id'] == annotation['image_id']
#             image_id = question['image_id']
#             image = Image.open(f'data/images/{name}2014/COCO_{name}2014_{str(image_id).zfill(12)}.jpg')
#             q_tokens = self.dictionary.tokenize(question['question'])[:max_question_length]
#             a_tokens = self.dictionary.tokenize(annotation['multiple_choice_answer'])
#             if len(q_tokens) < max_question_length:
#                 q_tokens = [self.dictionary.pad_token] * (max_question_length - len(q_tokens)) + q_tokens
#             self.entries.append((image, q_tokens, a_tokens))

if __name__ == '__main__':
    dict = TokenDictionary()
    train_data = VQAFeatureDataset('train', dictionary=dict)
    val_data = VQAFeatureDataset('val', dictionary=dict)

    # VQA_itzik = VQADataset(dictionary=dict)
