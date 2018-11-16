import itertools
import re
from pathlib import Path
from copy import deepcopy
# from pprint import pprint as print

import numpy as np
from keras.utils import np_utils

from .Negotiation import Negotiation, User, Dialogue, Issue, Option

RE_LINE = re.compile(r'(\d+ \d+ \d+ \d+ \d+ \d+) (.*?) (item0=\d+ item1=\d+ item2=\d+|no agreement|disconnect) <eos> reward=(\d+|no agreement|disconnect) (.*?) (\d+ \d+ \d+ \d+ \d+ \d+)')

def preprocess(text):
    return text.replace('(', '( ').replace(')', ' )')

class FacebookUser(User):
    """docstring for FacebookUser."""
    def __init__(self, name):
        super(FacebookUser, self).__init__(name)
        self.option2value = {}
        self.issue2weight = {}
        
    def calc_value(self, options):
        return sum([self.option2value[option] for option in options])

class FacebookNegotiation(Negotiation):
    """docstring for FacebookNegotiation."""
    def __init__(self, spoken_words, choice, reward, agreement, item_counts, value, partner_value):
        super(FacebookNegotiation, self).__init__()        
        user_you = FacebookUser('YOU')
        user_them = FacebookUser('THEM')
        self.users = [
            user_you,
            user_them
        ]
        
        self.dialogues = []
        dialogue = None
        for word in spoken_words:
            if word == 'YOU:' or word == 'THEM:':
                if dialogue:
                    self.dialogues.append(dialogue)
                user = user_you if word == 'YOU:' else user_them
                dialogue = Dialogue(user, [])
            else:
                dialogue.words.append(word)
        if dialogue:
            self.dialogues.append(dialogue)

        self.issues = []
        for i, (item_name, item_count) in enumerate(zip(['book', 'hat', 'ball'], item_counts)):
            options = []
            for n_items_you_have in range(item_count + 1):
                option = Option('{}_{}'.format(item_name, n_items_you_have))
                user_you.option2value[option] = n_items_you_have * value[i]
                user_them.option2value[option] = (item_count - n_items_you_have) * partner_value[i]
                options.append(option)
                
            user_you.issue2weight[item_name] = value[i]
            user_them.issue2weight[item_name] = partner_value[i]
            issue = Issue(item_name, options)
            issue.item_count = item_count
            self.issues.append(issue)

        self.user_you = user_you
        self.user_them = user_them
        
        self.choice = choice
        self.reward = reward
        self.item_counts = item_counts
        self.agreement = agreement

    def __str__(self):
        return "".join([str(d) for d in self.dialogues])

    def get_drafts(self):
        options_combinations = list(itertools.product(*[issue.options for issue in self.issues]))
        drafts = [draft for draft in options_combinations]
        draft_values = [np.array([user.calc_value(draft) for user in self.users]).prod() for draft in options_combinations]
        return list(zip(drafts, draft_values))
        
    def get_drafts_user(self, user):
        options_combinations = list(itertools.product(*[issue.options for issue in self.issues]))
        drafts = [draft for draft in options_combinations]
        draft_values = [user.calc_value(draft) for draft in options_combinations]
        return list(zip(drafts, draft_values))

    def to_Xy(self, word2id, option2id):
        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        
        # drafts = self.get_drafts()
        drafts = self.get_drafts_user(self.users[0])
        
        str_dialogue = (" ".join([str(d) for d in self.dialogues])).split(" ")
        X_dialogue = [word2id[word] for word in str_dialogue]
        X_drafts = [[option2id[option.name] for option in draft] for draft, value in drafts]
        y = np.array([value for draft, value in drafts])
        y = y / y.sum()
        return X_dialogue, X_drafts, y

    @staticmethod
    def from_file(file_path):
        negos = []
        with open(file_path) as f:
            for line in f:
                m = RE_LINE.match(line)
                value = [int(_) for _ in m.group(1).split(' ')[1::2]]
                item_counts = [int(_) for _ in m.group(1).split(' ')[::2]]
                spoken_words = m.group(2).split(' ')
                choice = m.group(3)
                reward = m.group(4).split(' ')
                agreement = m.group(5)
                partner_value = [int(_) for _ in m.group(6).split(' ')[1::2]]
                nego = FacebookNegotiation(spoken_words, choice, reward, agreement, item_counts, value, partner_value)
                spoken_words_inv = ['YOU:' if word == 'THEM:' else 'THEM:' if word == 'YOU:' else word for word in spoken_words]
                nego.inverse = FacebookNegotiation(spoken_words_inv, None, None, None, item_counts, partner_value, value)
                negos.append(nego)
        return negos