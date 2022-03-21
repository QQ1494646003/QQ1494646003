tokens = []

def construct_tokens():
    with open('/home/wbc/wbc/TransformerKW/enterprise_key.token', encoding="utf-8") as f:
        cachetokens = f.readlines()
    for t in cachetokens: tokens.append(t.strip('\n'))


def evaluate_from_token(enterprise_token):
    construct_tokens()
    if enterprise_token in tokens:
        print('Passed. \tEnterprise Mode')
        return 1#'Enterprise'
    else:
        print('General')
        return 0#'General Mode'
