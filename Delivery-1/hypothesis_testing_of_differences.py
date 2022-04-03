from collections import Counter
from decimal import DivisionByZero
from tqdm import tqdm
import math
import json
from collections import defaultdict


def calculate_t_values(bigrams,  n=20):
    collocations = {}
    v_w_dict = {}

    for bigram in bigrams.keys():
        tokens = bigram.split()
        v_w_dict[tokens[0]] = {tokens[1]: bigrams[bigram]}

    del bigrams

    # v1 => "a": {"ölünce": 1, "özel": 3, ...} -> len = 725
    # v2 => "aa": {"aynı": 1, "bağlıklı": 1, ...} -> len = 12

    for v1 in tqdm(v_w_dict.keys()):
        for v2 in v_w_dict.keys():
            if v1 == v2:
                continue
            else:
                if len(v_w_dict[v1]) < len(v_w_dict[v2]):
                    shorter_dict = v_w_dict[v1]
                    longer_dict = v_w_dict[v2]
                else:
                    shorter_dict = v_w_dict[v2]
                    longer_dict = v_w_dict[v1]
                # calculate the t values by C(v1, w) - C(v2, w) / sqrt(C(v1, w) + C(v2, w)),
                # where C(x) is times x occurs in the bigrams
                for w in shorter_dict.keys():
                    try:
                        t_value = (
                            shorter_dict[w] - longer_dict[w]) / math.sqrt(shorter_dict[w] + longer_dict[w])
                    except KeyError:
                        continue
                    except DivisionByZero:
                        t_value = 0
                    collocations[f"{v1} {w} - {v2} {w}"] = round(t_value, 2)

    with open('hypo_collocation.json', 'w', encoding='utf-8') as f:
        json.dump(collocations, f, ensure_ascii=False,
                  sort_keys=True, indent=4)

    sort_orders = sorted(collocations.items(),
                         key=lambda x: x[1], reverse=True)[:n]
    for item in sort_orders:
        print(str(item[0]) + ' ' + str(item[1]))
