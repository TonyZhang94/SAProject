# -*- coding: utf-8 -*-


def read():
    data = list()
    file = "../data/label_test"
    with open(file, mode="r", encoding="utf-8") as fp:
        lines = fp.readlines()
        chs, pos = list(), list()
        inx = 0
        for line in lines:
            if 1 == len(line):
                data.append([chs, pos])
                del chs
                del pos
                chs, pos = list(), list()
                inx = 0
                continue

            # _label 样本打标
            # label  模型打标
            hanz, _label, label = line.split()
            used_label = _label
            chs.append(hanz)

            if "B-T" == used_label:
                # 找到结束
                pass
            elif "I-T" == used_label:
                print("没有出现B-T")
                print(label)
            elif "0" == used_label:
                pass
            else:
                print("未知情况")
                print(label)

            inx += 1

    return data


if __name__ == '__main__':
    pass
