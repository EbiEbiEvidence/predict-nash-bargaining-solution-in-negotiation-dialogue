import numpy as np


class Negotiation(object):
    def __init__(self):
        super(Negotiation, self).__init__()
        self.issues = []
        self.dialogues = []
        self.users = []

    def option2id(self):
        return {option:o for issue in self.issues for o, option in enumerate(issue.options)}


class User(object):
    def __init__(self, name=None):
        super(User, self).__init__()
        self.name = name

    def calc_value(self, options):
        raise NotImplementedError()


class Dialogue(object):
    def __init__(self, user, words):
        super(Dialogue, self).__init__()
        self.user = user
        self.words = words

    def __str__(self):
        return "<{}> {}".format(self.user.name, ' '.join(self.words))


class Issue(object):
    def __init__(self, name=None, options=None):
        super(Issue, self).__init__()
        self.name = name
        self.options = options


class Option(object):
    def __init__(self, name):
        super(Option, self).__init__()
        self.name = name
