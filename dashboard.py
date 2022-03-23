import getpass
import os
import time

from Model.model import CoreBERT
from entre_token import evaluate_from_token
from utilities import one_shot_inf, batch_inf, retrieve_one_shot, retrieve_batch


enterprise_user = False
help_info = ' Get Problems by keywords(Enterprise)\n'


def construct_bert():#这里需要联网
    kw_model = CoreBERT()
    print('Transformer Loaded.')
    return kw_model


def launchpad(enterprise_user):
    bert_model = construct_bert()

    while(True):
        retrieve_batch(enterprise_user)
        if operation == 'clear':
            os.system('clear')


if __name__ == '__main__':
    enterprise_token = getpass.getpass('Enter Enterprise_psw, press enter to skip:')
    enterprise_user = evaluate_from_token(enterprise_token)
    
    launchpad(enterprise_user)
