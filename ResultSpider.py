import os
from glob import glob

def peek_dataset(dataset_path='/home/wbc/wbc/TransformerKW/dataset/Cleaned/*.txt', shortest_fix=100, detailed = False):
    print('CONSTRUCT PROBLEM DATASETS..')
    
    problem_count = 0
    datasets = glob(dataset_path)
    peek_results = []
    print(f'HIT {len(datasets)} DATASETS..')

    for index, dataset in enumerate(datasets):
        oneshot_peak, dropped = peek_one_shot(dataset, shortest_fix)
        peek_results.append(oneshot_peak)
        problem_count += len(oneshot_peak)
        print(f'\tDATASET id {index}\t-\t{dataset}\tHIT\t{len(oneshot_peak)} PROBLEMS, {dropped} DROPPED')
    
    if detailed:
        print(f'\n\nDATASET DETAILS-----------------')
        for index, peek_result in enumerate(peek_results):
            print(f'\nDATASET id {index} DETAILS\n----------------------------------------------------------')
            for qindex, details in enumerate(peek_result):
                print(f'Qid {qindex}: {details}')
            print('----------------------------------------------------------')
    
    print(f'\nDATASET CHECK COMPLETE. TOTAL {problem_count} Problems')
    return peek_results

def peek_one_shot(dataset, shortest_fix):
    dropped = 0
    questions = []

    with open(dataset) as f:
        line = f.readline()
        while line:
            if line != '\n':
                if len(line) < shortest_fix:
                    dropped += 1
                    line = f.readline()
                    continue
                questions.append(line.strip('\n'))
            line = f.readline()
    
    return questions, dropped


if __name__ == "__main__":
    peek_dataset(detailed=False)
