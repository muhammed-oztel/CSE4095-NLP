from decimal import DivisionByZero
from tqdm import tqdm
import math
import json


class HypothesisTestingDiff:
    def __init__(self, bigrams):
        self.bigrams = bigrams

    # method to export the collocations by hypothesis testing diff
    def export_collocation_by_hypothesis_testing_diff(self, n=20):
        collocations = {}
        v_w_dict = {}

        for bigram in self.bigrams.keys():  # loop to get the vocabulary
            tokens = bigram.split()
            v_w_dict[tokens[0]] = {tokens[1]: self.bigrams[bigram]}

        del self.bigrams    # to free memory

        # v1 => "a": {"ölünce": 1, "özel": 3, ...} -> len = 725
        # v2 => "aa": {"aynı": 1, "bağlıklı": 1, ...} -> len = 12

        for v1 in tqdm(v_w_dict.keys()):  # v1 = "a"
            for v2 in v_w_dict.keys():  # v2 = "aa"
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
                        except KeyError:    # if w is not in longer_dict no t value needed
                            continue
                        except DivisionByZero:
                            t_value = 0
                        collocations[f"{v1} {w} - {v2} {w}"] = round(
                            t_value, 2)

        # export the collocations by hypothesis testing diff
        with open('data/hypothesis_testing_diff_collocation.json', 'w', encoding='utf-8') as f:
            json.dump(collocations, f, ensure_ascii=False,
                      sort_keys=True, indent=4)

        sort_orders = sorted(collocations.items(), key=lambda x: x[1], reverse=True)[
            :n]    # export the collocations by hypothesis testing diff
        with open(f'data/hypothesis_testing_diff_collocation_top_{n}.json', 'w', encoding='utf-8') as f:
            json.dump(sort_orders, f, ensure_ascii=False,
                      sort_keys=True, indent=4)
