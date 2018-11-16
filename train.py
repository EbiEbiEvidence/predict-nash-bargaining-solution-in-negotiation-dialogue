import random
import itertools
import logging
import operator
import argparse
from collections import Counter
from collections import namedtuple

import numpy as np
from scipy.stats import spearmanr
from keras.callbacks import Callback, ModelCheckpoint
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

from src.utils import timeit
from src.processor import *


@timeit
def read_negos(filename):
    from src.FacebookNegotiation import FacebookNegotiation
    return FacebookNegotiation.from_file(filename)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class FacebookGenerator(object):
    """docstring for FacebookGenerator."""
    def __init__(self, negos, word2id, option2id, processor):
        super(FacebookGenerator, self).__init__()
        self.negos = negos
        self.word2id = word2id
        self.option2id = option2id
        self.processor = processor

    def generator(self, batch_size=32, n_dialogues=256, n_drafts=40):
        Xs_dialogue = []
        ys = []
        while True:
            # TODO: generatorとgetの共通部分をまとめる
            # TODO: Xs_dialogueとyをそれぞれランダムに並び替える
            for n, nego in enumerate(self.negos):
                X_dialogue, y = self.processor.to_Xy(nego)
                Xs_dialogue.append(X_dialogue)
                ys.append(y)
                if len(Xs_dialogue) == batch_size or n == len(self.negos) - 1:
                    Xs_dialogue = np.array(Xs_dialogue)
                    ys = np.array(ys)
                    yield Xs_dialogue, ys
                    Xs_dialogue = []
                    ys = []
                    
    def get(self, n_dialogues=256, n_drafts=40):
        Xs_dialogue = []
        ys = []
        for n, nego in enumerate(self.negos):
            X_dialogue, y = self.processor.to_Xy(nego)
            Xs_dialogue.append(X_dialogue)
            ys.append(y)
            
        Xs_dialogue = np.array(Xs_dialogue)
        ys = np.array(ys)
        return Xs_dialogue, ys

           
class EvalAccuracyCallback(Callback):
    def __init__(self, X, y, negos, word2id):
        super(EvalAccuracyCallback, self).__init__()
        self.X = X
        self.y = y
        self.negos = negos
        self.word2id = word2id

    def on_epoch_end(self, epoch, logs={}):
        pred_ys = self.model.predict(self.X)
        spearmanr_ours = []
        for p, pred_y in enumerate(pred_ys):
            spearmanr_our = spearmanr(self.y[p][:3], pred_y[:3]).correlation
            if not np.isnan(spearmanr_our):
                spearmanr_ours.append(spearmanr_our)
        print("#{}:".format(epoch+1))
        print("Accuracy  : {}".format(accuracy_score(np.argmax(self.y, axis=1), np.argmax(pred_ys, axis=1))))
        print("Spearmanr : {}".format(sum(spearmanr_ours)/len(spearmanr_ours)))


def calc_user_score(weight, n_item):
    return sum([a * b for a, b in zip(weight, n_item)])


def calc_score(nego, n_item_you, weight_you, weight_them):
    n_item_them = [issue.item_count - n for issue, n in zip(nego.issues, n_item_you)]
    score_you = calc_user_score(weight_you, n_item_you)
    score_them = calc_user_score(weight_them, n_item_them)
    return score_you, score_them
        

def is_pareto(bid, bids):
    if not bid.n_item:
        return False
    for other_bid in bids:
        if other_bid.you_score >= bid.you_score and other_bid.them_score >= bid.them_score and not other_bid.n_item == tuple(bid.n_item):
            return False
    return True
    

class EvalNashCallback(Callback):
    def __init__(self, negos, processor):
        super(EvalNashCallback, self).__init__()
        self.negos = negos
        self.processor = processor

    def on_epoch_end(self, epoch, logs={}):
        negos_inversed = [nego.inverse for nego in self.negos]
        
        Xy_you = [self.processor.to_Xy(nego) for nego in self.negos]
        Xs_you = np.array([X for X, y in Xy_you])
        y_you = [y for X, y in Xy_you]
        ys_you = list(self.model.predict(Xs_you))
        
        Xy_them = [self.processor.to_Xy(nego) for nego in negos_inversed]
        Xs_them = np.array([X for X, y in Xy_them])
        y_them = [y for X, y in Xy_them]
        ys_them = list(self.model.predict(Xs_them))
        
        Bid = namedtuple('Bid', ('n_item', 'you_score', 'them_score'))
        
        nash_products_pred = []
        nash_products_human = []
        nash_products_true = []
        social_welfare_pred = []
        social_welfare_human = []
        social_welfare_true = []
        n_success_pred_pareto = 0
        n_success_pred_nash = 0
        n_success_human_pareto = 0
        n_success_human_nash = 0
        
        for n, nego in enumerate(self.negos):
            issue_book, issue_hat, issue_ball = nego.issues
            n_item_there = [issue.item_count for issue in nego.issues]
            weight_you_pred = ys_you[n]
            weight_you_pred *= 10 / calc_user_score(ys_you[n], n_item_there)
            weight_them_pred = ys_them[n]
            weight_them_pred *= 10 / calc_user_score(ys_them[n], n_item_there)
            weight_you_true = [nego.user_you.issue2weight[item] for item in ['book', 'hat', 'ball']]
            weight_them_true = [nego.user_them.issue2weight[item] for item in ['book', 'hat', 'ball']]

            y_pred = None
            y_pred_score = 0
            y_true = None
            y_true_score = 0
            
            bids = []
            for item_counts in itertools.product(*[list(range(issue.item_count + 1)) for issue in nego.issues]):
                n_you_have = item_counts
                score_you_true, score_them_true = calc_score(nego, n_you_have, weight_you_true, weight_them_true)
                bids.append(Bid(n_you_have, score_you_true, score_them_true))
            
            pareto_bids = []
            for bid in bids:
                if is_pareto(bid, bids):
                    pareto_bids.append(bid)

            y_human = None
            if "item" in nego.choice:
                y_human = [int(c.split("=")[1]) for c in nego.choice.split(" ")]              
            for item_counts in itertools.product(*[list(range(issue.item_count + 1)) for issue in nego.issues]):
                n_you_have = item_counts
                score_you_pred, score_them_pred = calc_score(nego, n_you_have, weight_you_pred, weight_them_pred)
                if y_pred_score < score_you_pred * score_them_pred:
                    y_pred_score = score_you_pred * score_them_pred
                    y_pred = n_you_have
                    
                score_you_true, score_them_true = calc_score(nego, n_you_have, weight_you_true, weight_them_true)
                if y_true_score < score_you_true * score_them_true:
                    y_true_score = score_you_true * score_them_true
                    y_true = n_you_have
            
            score_you_true, score_them_true = calc_score(nego, y_true, weight_you_true, weight_them_true)
            nash_products_true.append(score_you_true * score_them_true)
            social_welfare_true.append(score_you_true + score_them_true)
            
            score_you_human, score_them_human = 0, 0
            if y_human:
                score_you_human, score_them_human = calc_score(nego, y_human, weight_you_true, weight_them_true)
            human_bid = Bid(y_human, score_you_human, score_them_human)
            nash_products_human.append(score_you_human * score_them_human)
            social_welfare_human.append(score_you_human + score_them_human)
            if is_pareto(human_bid, bids):
                n_success_human_pareto += 1
            if score_you_human * score_them_human == y_true_score:
                n_success_human_nash += 1

            n_you_have = y_pred
            score_you_actual, score_them_actual = calc_score(nego, n_you_have, weight_you_true, weight_them_true)
            nash_products_pred.append(score_you_actual * score_them_actual)
            social_welfare_pred.append(score_you_actual + score_them_actual)
            pred_bid = Bid(n_you_have, score_you_actual, score_them_actual)
            if is_pareto(pred_bid, bids):
                n_success_pred_pareto += 1
            if score_you_actual * score_them_actual == y_true_score:
                n_success_pred_nash += 1
        
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            sum(nash_products_true) / len(nash_products_true),
            sum(social_welfare_true) / len(social_welfare_true),
            sum(nash_products_pred) / len(nash_products_pred),
            sum(social_welfare_pred) / len(social_welfare_pred),
            n_success_pred_pareto / len(self.negos),
            n_success_pred_nash / len(self.negos),
            sum(nash_products_human) / len(nash_products_human),
            sum(social_welfare_human) / len(social_welfare_human),
            n_success_human_pareto / len(self.negos),
            n_success_human_nash / len(self.negos)
        ))


def nego2str(nego):
    return "".join([str(dialogue) for dialogue in nego.dialogues])


@timeit
def main():
    negos = read_negos('dat/data.txt')
    nego_dialogues = {}
    negos_distinct = []
    for nego in negos:
        if nego2str(nego) in nego_dialogues:
            continue
        nego_dialogues[nego2str(nego.inverse)] = 0
        nego_dialogues[nego2str(nego)] = 0
        negos_distinct.append(nego)
    negos = negos_distinct
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int)
    args = parser.parse_args()
    word_counter = Counter()
    option_counter = Counter()
    for nego in negos:
        for dialogue in nego.dialogues:
            word_counter += Counter(str(dialogue).split(" "))
        for issue in nego.issues:
            option_counter += Counter([option.name for option in issue.options])
    
    word2id = {word:idx + 1 for idx, word in enumerate(word_counter.keys())}
    word2id['<UNK>'] = 0
    
    option2id = {name:idx + 1 for idx, name in enumerate(option_counter.keys())}
    option2id['<UNK>'] = 0
    
    processor = Processor(word2id, 256, 3)
    
    # K-fold cross-validation
    K = 10
    for k in range(K):
        model = processor.get_model()
        model.compile(optimizer='rmsprop',
                      loss='mse',
                      metrics=['accuracy'],)

        test_negos = negos[int(k * len(negos) * (1 / K)):int((k+1) * len(negos) * (1 / K))]
        train_negos = list(set(negos) - set(test_negos))
        train_negos += [nego.inverse for nego in train_negos]
        random.shuffle(train_negos)
        train_gen = FacebookGenerator(train_negos, word2id, option2id, processor)
        test_gen = FacebookGenerator(test_negos, word2id, option2id, processor)
        val_gen = FacebookGenerator(test_negos, word2id, option2id, processor)
        val_X, val_y = val_gen.get()
    
        BATCH_SIZE = args.batch_size
        if BATCH_SIZE is None:
            BATCH_SIZE = 32
    
        model.fit_generator(
            generator=train_gen.generator(BATCH_SIZE),
            steps_per_epoch=np.ceil(len(train_negos) / BATCH_SIZE),
            epochs=20,
            verbose=False,
            validation_data=test_gen.generator(BATCH_SIZE),
            validation_steps=np.ceil(len(train_negos) / BATCH_SIZE),
            callbacks=[
                ModelCheckpoint('models/weights.{epoch:02d}.hdf5'),
                EvalAccuracyCallback(val_X, val_y, negos, word2id),
                EvalNashCallback(test_negos, processor)
            ]
        )

if __name__ == '__main__':
    main()
