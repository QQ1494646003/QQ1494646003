import getpass
import os
import time

from Model.model import CoreBERT
from entre_token import evaluate_from_token
from utilities import one_shot_inf, batch_inf, retrieve_one_shot, retrieve_batch


enterprise_user = False
help_info = 'HELP Notes:\n0: QUIT\n1: One-Shot Inference\n2: Batch Inference(Enterprise)\n3: Get Problems by keyword\n4: Get Problems by keywords(Enterprise)\n'


def construct_bert():
    #print('Loading Transformer Model..')
    kw_model = CoreBERT()
    print('Transformer Loaded.')
    return kw_model


def launchpad(enterprise_user):
    bert_model = construct_bert()

    while(True):
        #print(help_info)

        #if enterprise_user: operation = input('Enterprise: ')
        #else: operation = input('General: ')
        
        #if operation == '0':
        #    print('Exiting.')
        #    return
        #if operation == '1':
        #    one_shot_inf(bert_model)
        #if operation == '2':
        #    batch_inf(bert_model, enterprise_user)
        #if operation == '3':
        #    retrieve_one_shot()
        #if operation == '4':
        retrieve_batch(enterprise_user)
        #if operation == 'clear':
            #os.system('clear')


if __name__ == '__main__':
    enterprise_token = getpass.getpass('Enter Enterprise_psw, press enter to skip:')
    enterprise_user = evaluate_from_token(enterprise_token)
    
    launchpad(enterprise_user)
