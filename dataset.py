from __future__ import print_function
import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import time


class TokenDictionary:
    def __init__(self):
        self.words = ['<padding>','<unk>']
        self.tokens = {'<padding>':0, '<unk>':1}
        self.cnt = 2
        self.cnt_word()

    def __len__(self):
        return len(self.tokens)

    def cnt_word(self):
        questions = json.load(open(f'/content/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))['questions']
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


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary,create_pt = False,max_question_length=14):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']
        self.dictionary = dictionary
        self.entries = []
        ans2labels = pickle.load(open('/content/HW2-deep-learning/cache/trainval_ans2label.pkl', 'rb'))
        self.num_of_answers = len(ans2labels)
        del ans2labels
        if create_pt:
            self.create_pt(name)
        self.load_data(name, max_question_length)

    def create_pt(self,phase):
        self.imgs_ids = [s[13:-4] for s in os.listdir(os.path.join('/content', f'{phase}2014'))]
        for id in self.imgs_ids:
            path = os.path.join('/content',f'{phase}2014', f'COCO_{phase}2014_{id}.jpg')
            save_path = os.path.join('/content',f'{phase}2014', f'COCO_{phase}2014_{id}.pt')
            image = Image.open(path).convert('RGB')
            # Resize
            resize = transforms.Resize(size=(224, 224))
            image = resize(image)
            # this also divides by 255 TODO we can normalize too change?!?!
            image_tensor = TF.to_tensor(image)

            torch.save(image_tensor.to(dtype=torch.float16), save_path)
            os.remove(path)  # delete .jpg file # TODO

    def load_data(self,name, max_question_length):
        questions = json.load(open(f'/content/v2_OpenEnded_mscoco_{name}2014_questions.json', 'r'))['questions']
        answers = json.load(open(f'/content/v2_mscoco_{name}2014_annotations.json', 'r'))['annotations']
        all_answers = pickle.load(open(f'/content/HW2-deep-learning/cache/{name}_target_new.pkl', 'rb'))
        questions = sorted(questions, key=lambda q: q['question_id'])
        answers = sorted(answers, key=lambda a: a['question_id'])
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
            if not ans:  # we ignore unkown answers
                continue
            self.entries.append((name, image_id, q_tensor,question['question_id'] ,ans))

    def __getitem__(self, index):
        name = self.entries[index][0]
        image_id = self.entries[index][1]
        ans = self.entries[index][4]
        a_token = torch.zeros(self.num_of_answers)
        for key, score in ans.items():
            a_token[key] = score  # a_token[ans2labels[a]] = score
        for i in range(10):
            try:
                image = Image.open(f'/content/{name}2014/COCO_{name}2014_{str(image_id).zfill(12)}.jpg').convert('RGB')
                resize = transforms.Resize(size=(224, 224)) #TODO change?
                image = resize(image)
                image_tensor = TF.to_tensor(image)
                break
            except Exception as e:
                print(e)
                time.sleep(3)
                continue
        tmp = image_tensor, image_id, self.entries[index][2], self.entries[index][3], a_token
        return tmp

    def __len__(self):
        return len(self.entries)


if __name__ == '__main__':
    dict = TokenDictionary()
    train_data = VQAFeatureDataset('train', dictionary=dict)
    val_data = VQAFeatureDataset('val', dictionary=dict)
