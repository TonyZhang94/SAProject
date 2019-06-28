# -*- coding: utf-8 -*-

import copy


def make_input(data):
    """
    attr， opi 以出现的次序为key

    n_attr int, n_opi int
    mat R^2
    attrs [], opis []
    merge_info {pos_attr: [ori_words]}
    ind_attr set()
    """
    n_attr, n_opi = 0, 0
    mat = list()
    attrs, opis = list(), list()
    merge_info = dict()
    ind_attr = set()

    comment = data[0]

    attrpos2row = dict()
    opipos2col = dict()
    for inx in range(len(data)):
        if 0 == inx:
            continue
        attr = data[inx]["attr"]

        pos_attr = comment.find(attr)
        attrpos2row[pos_attr] = n_attr

        attrs.append(attr)
        if data[inx]["merge"] is True:
            merge_info[n_attr] = data[inx]["ori"]
        if 0 == len(data[inx]["opis"]):
            ind_attr.add(n_attr)

        n_attr += 1
        for opi in data[inx]["opis"]:
            pos = comment.find(opi)
            if pos not in opipos2col:
                opipos2col[pos] = n_opi
                opis.append(opi)
                n_opi += 1

    for y in range(n_attr):
        temp_vec = list()
        for x in range(n_opi):
            temp_vec.append(0)
        mat.append(temp_vec)
        del temp_vec

    for inx in range(len(data)):
        if 0 == inx:
            continue

        attr = data[inx]["attr"]
        pos_attr = comment.find(attr)
        for opi in data[inx]["opis"]:
            pos_opi = comment.find(opi)
            if pos_opi < pos_attr:
                pos = pos_opi + len(opi) - pos_attr - 1
            else:
                pos = pos_opi - pos_attr - len(attr) + 1
            mat[attrpos2row[pos_attr]][opipos2col[pos_opi]] = pos

    return comment, n_attr, n_opi, mat, attrs, opis, merge_info, ind_attr


def show_input(comment, n_attr, n_opi, mat, attrs, opis, merge_info):
    print("\n=============================")
    print("原评价：", comment)
    print("attrs", n_attr, attrs)
    print("opis", n_opi, opis)
    print("merge info", merge_info)
    print("mat:", mat)


def optimal_match(n_attr, n_opi, mat, ind_attr):
    records, records_hash_set = list(), set()
    status = list()
    # status.append([[], ind_attr, set()])
    status.append([[], set(), set()])
    while 0 != len(status):
        record, vis_attr, vis_opi = status.pop()
        if n_attr == len(vis_attr) or n_opi == len(vis_opi):
            if 0 == len(records):
                record.sort(key=lambda x: x[0])
                records.append(record)
                records_hash_set.add("=".join(["#".join(inx_pair for inx_pair in map(str, record))]))
            else:
                temp_max_match = len(records[0])
                if temp_max_match < len(record):
                    continue
                elif temp_max_match == len(record):
                    record.sort(key=lambda x: x[0])
                    hash_key = "=".join(["#".join(inx_pair for inx_pair in map(str, record))])
                    if hash_key not in records_hash_set:
                        records.append(record)
                        records_hash_set.add(hash_key)
                else:
                    record.sort(key=lambda x: x[0])
                    del records
                    records = [record]
                    del records_hash_set
                    records_hash_set.add("=".join(["#".join(inx_pair for inx_pair in map(str, record))]))
            continue
        for inx_attr in range(n_attr):
            if inx_attr in vis_attr:
                continue
            for inx_opi in range(n_opi):
                if inx_opi in vis_opi or 0 == mat[inx_attr][inx_opi]:
                    continue

                new_record = copy.copy(record)
                new_vis_attr, new_vis_opi = copy.copy(vis_attr), copy.copy(vis_opi)

                new_record.append([inx_attr, inx_opi])
                new_vis_attr.add(inx_attr)
                new_vis_opi.add(inx_opi)
                status.append([new_record, new_vis_attr, new_vis_opi])

    print("可能结果：")
    for record in records:
        print(sorted(record, key=lambda x: x[0]))
        cur_record_sum_dis, cur_record_bw_num = 0, 0
        for [inx_attr, inx_opi] in record:
            # inx_attr, inx_opi = inx_pair
            dis = mat[inx_attr][inx_opi]
            cur_record_sum_dis += abs(dis)
            if dis < 0:
                cur_record_bw_num += 1

        if 'result' not in locals().keys():
            result = record
            min_sum_dis = cur_record_sum_dis
            result_bw_num = cur_record_bw_num
        else:
            if cur_record_sum_dis < min_sum_dis:
                result = record
                min_sum_dis = cur_record_sum_dis
                result_bw_num = cur_record_bw_num
            elif cur_record_sum_dis == min_sum_dis:
                if cur_record_bw_num < result_bw_num:
                    result = record
                    result_bw_num = cur_record_bw_num

    print("最终结果：", result)
    return result


def show_result(result, attrs, opis, merge_info):
    result_inx = 0
    for inx_attr in range(len(attrs)):
        if result_inx >= len(result) or inx_attr != result[result_inx][0]:
            print(attrs[inx_attr], "无情感词")
            continue

        _, inx_opi = result[result_inx]
        result_inx += 1
        if inx_attr not in merge_info:
            print(attrs[inx_attr], opis[inx_opi])
        else:
            origin_attrs = merge_info[inx_attr]
            for attr in origin_attrs:
                print(attr, opis[inx_opi])


if __name__ == '__main__':
    datum = list()
    datum.append(["绞肉不错但是用的时候声音很大",
                 {"attr": "绞肉", "opis": ["不错"], "merge": False}, {"attr": "声音", "opis": ["很大"], "merge": False}])
    datum.append(["绞肉不错也可以使用它榨汁",
                 {"attr": "绞肉", "opis": ["不错"], "merge": False}, {"attr": "榨汁", "opis": [], "merge": False}])
    datum.append(["虽然声音非常大但绞肉绞的细腻",
                 {"attr": "声音", "opis": ["大"], "merge": False}, {"attr": "绞肉", "opis": ["大", "细腻"], "merge": False}])
    datum.append(["可以用来绞肉但声音很大",
                 {"attr": "绞肉", "opis": ["很大"], "merge": False}, {"attr": "声音", "opis": ["很大"], "merge": False}])
    datum.append(["声音很大很吵但是绞肉绞的细腻",
                 {"attr": "声音", "opis": ["很大", "很吵"], "merge": False}, {"attr":  "绞肉", "opis": ["很大", "很吵", "细腻"], "merge": False}])
    datum.append(["绞肉和榨汁全都很棒",
                 {"attr": "绞肉和榨汁", "opis": ["很棒"], "merge": True, "ori": ["绞肉", "榨汁"]}])

    for data in datum:
        comment, n_attr, n_opi, mat, attrs, opis, merge_info, ind_attr = make_input(data)
        show_input(comment, n_attr, n_opi, mat, attrs, opis, merge_info)
        result = optimal_match(n_attr, n_opi, mat, ind_attr)
        show_result(result, attrs, opis, merge_info)

"""
for if in 改大写
"""
""" Optimal Match: Inputs (LIST attrs, LIST opis, MATRIX mat, DICT mergeLog
    EMPTY SET records
    STACK status.push [EMPTY LIST, EMPTY SET, EMPTY SET]
    while !status.empty:
        LIST record, SET vis_attr, SET vis_opi = status.pop
        if attrs.size == vis_attr.size or n_opi == vis_opi.size:
            if 0 == records.size: records.add record.sort
            else: match_num = records.top.size
                  if match_num == record.size: records.add record.sort
                  if match_num < record.size: records.clear, records.add record.sort
            continue
        for INT inx_attr = 0 ~ mat.num_rows:
            if inx_attr in vis_attr: continue
            for INT inx_opi = 0 ~ mat.num_columns:
                if inx_opi in vis_opi or 0 == mat.inx_attr.inx_opi: continue
                status.push [record.copy.add [inx_attr, inx_opi], vis_attr.copy.add inx_attr, 
                             vis_opi.copy.add inx_opi]
    
    INT result_sum_dis, result_bw_num
    for record in records, INT sum_dis = 0, INT bw_num = 0:
        for [inx_attr, inx_opi] in record:
            sum_dis += abs(mat.inx_attr.inx_opi)
            if mat.inx_attr.inx_opi < 0: bw_num += 1
        if result is NULL or sum_dis < result_sum_dis: 
            result = record, result_sum_dis = sum_dis, result_bw_num = bw_num
        if cur_record_sum_dis == min_sum_dis and bw_num < result_bw_num:
            result = record, result_bw_num = bw_num
    
    Outputs result
"""
"""
    EMPTY LIST analysis, INT inx_result = 0
    for INT inx_attr = 0 ~ attrs.size:
        if result_inx >= result.size or inx_attr != result.inx_result.first:
            outputs.push [attrs.inx_attr, NULL]
        else:
            result_inx += 1, INT inx_opi = result.inx_result.second
            if inx_attr not in merge_info: outputs.push [attrs.inx_attr, opis.inx_opi]
            else: outputs.push [merge_info.inx_attr.first, opis.inx_opi] and  
                               [merge_info.inx_attr.second, opis.inx_opi] ……
    
    Outputs analysis
"""