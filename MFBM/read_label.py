# -*- coding: utf-8 -*-


def read():
    data = list()
    file = "../data/label_test"
    with open(file, mode="r", encoding="utf-8") as fp:
        lines = fp.readlines()
        chs, pos = list(), list()
        comment = list()
        for line in lines:
            if 1 == len(line):
                inx = 0
                size = len(comment)
                while inx < size:
                    hanz, label = comment[inx]
                    if "B-T" == label:
                        start = inx
                        inx += 1
                        while inx < size and "I-T" == comment[inx][1]:
                            inx += 1
                        pos.append([start, inx])
                    elif "I-T" == label:
                        pass
                        print("没有出现B-T")
                        inx += 1
                    elif "0" == label:
                        inx += 1
                    else:
                        print("未知情况")
                        print(label)
                        inx += 1

                text = "".join(chs)
                data.append([text, pos])
                del chs
                del pos
                chs, pos = list(), list()
                del comment
                comment = list()
            else:
                # _label 样本打标
                # label  模型打标
                hanz, _label, label = line.split()
                used_label = _label
                comment.append([hanz, used_label])
                chs.append(hanz)

    return data


if __name__ == '__main__':
    data = read()
    print("len data:", len(data))
