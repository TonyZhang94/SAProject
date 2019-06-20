# -*- coding: utf-8 -*-


from SAProject.MFBM.lexicon.lexicon import GetLexicon
from SAProject.MFBM.read_label import read


class Match(object):
    def __init__(self, win_size):
        self.win_size = win_size
        lexicon = GetLexicon()
        lexicon.read_all()
        self.opinions = lexicon.get_opinions()

    def match(self):
        # data = [['实物没问题很好。', [[0, 2]]]]
        data = read()
        win_size = self.win_size
        result = list()

        for seq, comment in enumerate(data, start=1):
            # if seq > 100:
            #     break
            text, positions = comment
            opis = list()
            size = len(text)
            for pos in positions:
                start, end = pos
                # 到断句位置
                fw_start, fw_end = max(start - win_size, 0), start
                bw_start, bw_end = end, min(end + win_size, size)
                fw_text = text[fw_start: fw_end]
                bw_text = text[bw_start: bw_end]
                find = False
                opi = ""
                ori = ""
                for t in range(win_size):
                    turns = t + 1
                    range_size = win_size - t
                    for turn in range(turns):
                        # bw > fw
                        # bw
                        if range_size <= len(bw_text):
                            word = bw_text[turn: turn + range_size]
                            if word in self.opinions:
                                find = True
                                opi = word
                                ori = "bw"
                                break

                        # fw
                        if range_size <= len(fw_text):
                            word = fw_text[len(fw_text)-turn-range_size: len(fw_text)-turn]
                            if word in self.opinions:
                                find = True
                                opi = word
                                ori = "fw"
                                break

                    if find:
                        break

                opis.append([pos, opi, ori])
            result.append([text, opis])
            # if len(result) > 100:
            #     return result

        return result

    def make_inputs(self, data, s1=0.8, s2=0.1, s3=0.1):
        inputs = list()
        win_size = self.win_size
        for comment in data:
            text, positions = comment
            for pos in positions:
                p1, p2 = pos[0]
                try:
                    opi = self.opinions[pos[1]]
                    if opi not in [-1, 0, 1]:
                        continue
                except KeyError:
                    continue
                start = max(p1 - win_size, 0)
                end = min(p2 + win_size, len(text))
                if -1 == opi:
                    label = [1, 0, 0]
                elif 0 == opi:
                    label = [0, 1, 0]
                else:
                    label = [0, 0, 1]
                inputs.append([text[start: end], label])
        
        size = len(inputs)
        p1 = int(size*s1)
        p2 = int(size*s2) + p1
        p3 = int(size*s3) + p2

        train = inputs[: p1]
        test = inputs[p1: p2]
        dev = inputs[p2: p3]
        print("all data size", size)
        print("train size", len(train))
        print("test size", len(test))
        print("dev size", len(dev))

        return train, test, dev


if __name__ == '__main__':
    obj = Match(5)
    result = obj.match()
    result = result[: 100]
    inputs, _, _ = obj.make_inputs(result)
    for item in inputs:
        print(item)
