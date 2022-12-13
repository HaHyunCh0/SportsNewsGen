import os


DATA_PATH = 'data/6/'
RAW_DATA = 'data/data_with_reference.tsv'
START_TOKENS = [
    "<HomeTeam>", "<AwayTeam>", "<FTHG>", "<FTAG>", "<HTHG>", "<HTAG>",
    "<FTR>", "<HTR>", "<Bias>", "<Date>"]
END_TOKENS = [
    "</HomeTeam>", "</AwayTeam>", "</FTHG>", "</FTAG>", "</HTHG>", "</HTAG>",
    "</FTR>", "</HTR>", "</Bias>", "</Date>"]
START_TOKENS_v2 = [
    "<HomeTeam>", "<AwayTeam>", "<FullTimeHomeGoals>", "<FullTimeAwayGoals>",
    "<HalfTimeHomeGoals>", "<HalfTimeAwayGoals>", "<FullTimeResult>",
    "<HalfTimeResult>", "<Bias>", "<Date>"]
END_TOKENS_v2 = [
    "</HomeTeam>", "</AwayTeam>", "</FullTimeHomeGoals>", "</FullTimeAwayGoals>",
    "</HalfTimeHomeGoals>", "</HalfTimeAwayGoals>", "</FullTimeResult>",
    "</HalfTimeResult>", "</Bias>", "</Date>"]


def create_source_target(s_tokens, e_tokens, data):
    s_t_pairs = list()
    for d in data:
        vs = d.split('\t')
        print(vs)
        s = f"<Date>{vs[11]}</Date>"
        for i in range(len(s_tokens)-4):
            s = s + f"{s_tokens[i]}{vs[i+2]}{e_tokens[i]}"
        t = d.split('"')[1]
        s_t_pairs.append('\t'.join([s,t]))
    return s_t_pairs


def create_source_target_with_r(s_tokens, e_tokens, data):
    s_t_pairs = list()
    for d in data:
        vs = d.split('\t')
        s = f"<Date>{vs[11]}</Date>"
        for i in range(len(s_tokens)-2):
            s = s + f"{s_tokens[i]}{vs[i+2]}{e_tokens[i]}"
        t = d.split('"')[1]
        s_t_pairs.append('\t'.join([s,t]))
    return s_t_pairs


def create_source_target_with_rb(s_tokens, e_tokens, data):
    s_t_pairs = list()
    for d in data:
        vs = d.split('\t')
        s = f"<Date>{vs[11]}</Date>"
        for i in range(len(s_tokens)-1):
            s = s + f"{s_tokens[i]}{vs[i+2]}{e_tokens[i]}"
        t = d.split('"')[1]
        s_t_pairs.append('\t'.join([s,t]))
    return s_t_pairs


if __name__ == '__main__':
    with open(RAW_DATA, 'r', encoding='utf8') as f:
        data = list()
        for line in f:
            if line[0] == '"':
                continue
            data.append(line.rstrip())
    data = data[1:]
    # s_t_pairs = create_source_target(START_TOKENS, END_TOKENS, data)
    # s_t_pairs = create_source_target_with_r(START_TOKENS, END_TOKENS, data)
    # s_t_pairs = create_source_target_with_rb(START_TOKENS, END_TOKENS, data)
    # s_t_pairs = create_source_target(START_TOKENS_v2, END_TOKENS_v2, data)
    # s_t_pairs = create_source_target_with_r(START_TOKENS_v2, END_TOKENS_v2, data)
    s_t_pairs = create_source_target_with_rb(START_TOKENS_v2, END_TOKENS_v2, data)
    pairs_len = len(s_t_pairs)
    train_limit = int(pairs_len * 0.8)
    val_limit = int(pairs_len * 0.9)

    train_d = list()
    val_d = list()
    test_d = list()

    for i, pair in enumerate(s_t_pairs):
        if i <= train_limit:
            train_d.append(pair)
        elif i <= val_limit:
            val_d.append(pair)
        else:
            test_d.append(pair)

    print(len(train_d))
    print(len(val_d))
    print(len(test_d))

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    with open(DATA_PATH + 'train.tsv', 'w') as t_f:
        t_f.write('\t'.join(['data', 'text']))
        t_f.write('\n'.join(train_d))
    with open(DATA_PATH + 'val.tsv', 'w') as v_f:
        v_f.write('\t'.join(['data', 'text']))
        v_f.write('\n'.join(val_d))
    with open(DATA_PATH + 'test.tsv', 'w') as te_f:
        te_f.write('\t'.join(['data', 'text']))
        te_f.write('\n'.join(test_d))

