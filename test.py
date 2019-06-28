# -*- coding: utf-8 -*-

import jieba.posseg as pseg

text = "料理机性能很好，外观精致！"
words = pseg.cut(text)
for w in words:
    print(w.word, w.flag)