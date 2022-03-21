from DataSpider import peek_dataset, peek_one_shot
from tqdm import tqdm
from glob import glob
import datetime

def one_shot_inf(bert_model):
    doc = input('Please enter doc content:')
    keywords = bert_model.extract_keywords(doc)
    print(f'Extract result: {keywords}')
    return


def batch_inf(bert_model, enterprise_user):
    if not enterprise_user:
        print('ACCESS DENIED')
        return
    dataset_dir = input('Enter Dataset dir')

    '''
    detail_mode = False
    
    if len(dataset_dir) != 0:
        question_data = peek_dataset(dataset_path=dataset_dir,detailed=detail_mode)
    else:
        question_data = peek_dataset(detailed=detail_mode)
    '''
    if len(dataset_dir) == 0:
        dataset_dir = '/home/wbc/wbc/TransformerKW/dataset/Cleaned/*.txt'
    datasets = glob(dataset_dir)
    

    for datasetindex, iter_dataset in enumerate(datasets):
        assigned_keyword = iter_dataset.split('/')[-1].split('.')[0] + ','
        print(f'Dataset {datasetindex}:{len(datasets)}', end='')
        question_data, dropped = peek_one_shot(iter_dataset, 100)

        with open('/home/wbc/wbc/TransformerKW/Output/question_DB.csv', 'a+', encoding="utf-8") as o:
            for dindex, question in enumerate(question_data):
                keyword_str = assigned_keyword
                #print(f'Inferencing {d}:{index} - {question}')
                print(f'\rInferencing {dindex}  ', end='')
                keywords = bert_model.extract_keywords(question, top_n=10)
                for keyword in keywords:
                    keyword_str = keyword_str + str(keyword[0]) + ','
                o.writelines(f'{question} /splitsign/{keyword_str}\n')
    
    print('\nDone.\n\n')
    return


def retrieve_one_shot():
    kw_q = []
    kww = input('Please enter keyword:')
    if ',' in kww:
        print('Batch Inference Requires Enterpise permit')
        retrieve_one_shot()
    kw_q.append(kww)

    strict_mode = input('Strict Mode? Y/N ')
    if strict_mode == 'Y' or strict_mode == 'y':
        strict_mode = True
    elif strict_mode == 'N' or strict_mode == 'n':
        strict_mode = False
    else:
        print('invalid input, set default to N.')
        strict_mode = False

    searchengine(kw_q=kw_q, strict_mode=strict_mode)

    return


def retrieve_batch(enterprise_user):
    if not enterprise_user:
        print('ACCESS DENIED')
        return
    kw_q = input('Please enter keywords, seperated by , . (eg. array, pointer):')
    kw_q = kw_q.split(',')

    strict_mode = input('Strict Mode? Y/N ')
    if strict_mode == 'Y' or strict_mode == 'y':
        strict_mode = True
    elif strict_mode == 'N' or strict_mode == 'n':
        strict_mode = False
    else:
        print('invalid input, set default to N.')
        strict_mode = False
    
    searchengine(kw_q=kw_q, strict_mode=strict_mode)
    return


def searchengine(kw_q, database='/home/wbc/wbc/TransformerKW/Output/question_DB.csv',strict_mode=False):
    print('searching..')
    questions_res = []
    with open(database, 'r', encoding="utf-8") as dbr:
        line = dbr.readline()
        while line:
            kw_a = line.split('/splitsign/')[-1].split(',')
            kw_a.pop()
            if strict_mode:
                kw_a.pop(0)
            hit = cross_val(kw_q, kw_a)
            if hit:
                questions_res.append(line.split('/splitsign/')[0])
            line = dbr.readline()
    
    print('\r Search Done, results:')
    
    for qindex, question_iter in enumerate(questions_res):
        print(f'\t[{qindex}: {question_iter[:80]}...]')
    
    ct = str(datetime.datetime.now())
    result_svnm = '/home/wbc/wbc/TransformerKW/Output/'+ct+'.search'#file route
    with open(result_svnm, 'a+') as rs:#additional mode ,will create that file 
        for qindex, question_iter in enumerate(questions_res):
            rs.writelines(f'{qindex}: {question_iter}\n')#write new on the end of the file

    return


def cross_val(kw_q, kw_a):
    for item in kw_q:
        for itemb in kw_a:
            item_l = item.lower()
            itemb_l = itemb.lower()
            if item_l in itemb_l or itemb_l in item_l:
                # print(f'cross_found: {item} -> {itemb}')
                return True
    return False
