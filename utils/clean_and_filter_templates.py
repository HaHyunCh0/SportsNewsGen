# ENTER filename in this list
FILE_LIST = [
]
CORRECT_TOKEN_LIST = ["[DateTime]", "[HomeTeam]", "[AwayTeam]", "[FTHG]", "[FTAG]", "[HTHG]", "[HTAG]"]
CORRUPTED_TOKEN_MAP = {
"[date]": "[DateTime]", "[datetime]": "[DateTime]", "[dates]": "[DateTime]", "[Date]": "[DateTime]", "[Datetime]": "[DateTime]",
"[Dates]": "[DateTime]", "[datatime]": "[DateTime]", "[Datatime]": "[DateTime]", "[data time]": "[DateTime]",
"Shelsea": "[AwayTeam]", "Chelse": "[AwayTeam]", "Chelsia": "[AwayTeam]", "Chelmea": "[AwayTeam]",
"Chelea": "[AwayTeam]", "Chelza": "[AwayTeam]", "Chellsea": "[AwayTeam]", "Chelsee": "[AwayTeam]",
"Chelesea": "[AwayTeam]", "Chel": "[AwayTeam]", "Chellea": "[AwayTeam]",
" she ": " they ", " he ": " they ", " his ": " their ", " her ": " their ",
" He ": " They ", " She ": " They ", " His ": " Their ", " Her ": " Their "
}


def clean_tokens_and_corefs(f_name):
    corrected_list = list()
    with open(f_name) as f:
        for line in f.readlines():
            line = line.strip()
            for token in CORRUPTED_TOKEN_MAP:
                if token in line:
                    line = line.replace(token, CORRUPTED_TOKEN_MAP[token])
            corrected_list.append(line)
    return corrected_list


def filter_templates(f_name, corrected_list):
    filtered_list = list()
    for line in corrected_list:
        token_in = 0
        for token in CORRECT_TOKEN_LIST:
            if token not in line:
                continue
            token_in += 1
        if token_in == len(CORRECT_TOKEN_LIST):
            filtered_list.append(line)
    dup_filtered_list = set(filtered_list)
    filtered = len(corrected_list) - len(dup_filtered_list)
    return filtered, dup_filtered_list


if __name__ == '__main__':
    for f in FILE_LIST:
        print(f"Clearing corrupted tokens and corefs in {f}")
        corrected_list = clean_tokens_and_corefs(f)
        print(f"Corrupted tokens and corefs in {f} all cleared")
        print(f"Filtering out templates that don't contain all the tokens in {f}")
        filtered, filtered_list = filter_templates(f, corrected_list)
        print(f"{filtered} templates have been filtered out... filtering {f} completed")
        fw_name = f"{f.split('_v')[0]}_filtered.txt"
        with open(fw_name, "w") as fw:
            fw.write('\n'.join(filtered_list))
        print(f"Saved into {fw_name}")

