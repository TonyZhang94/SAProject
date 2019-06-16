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
                            word = bw_text[turn: range_size]
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

        return result


if __name__ == '__main__':
    # ['实物没问题很好。', [[[0, 2], '', '']]]
    obj = Match(5)
    result = obj.match()
    for item in result:
        print(item)
