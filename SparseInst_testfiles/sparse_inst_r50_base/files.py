def opentxt(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.readlines()
        contentnew = []
        for i in content:
            i1 = i.replace('\n','')
            contentnew.append(i1)
        return contentnew

def openjson(path):
    import json
    with open(path, "r", encoding='utf-8') as f:
        dic = json.load(f)
    return dic

def write_txtlist(path,content):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(content)

def write_json(path,content):
    import json
    with open(path, "w", encoding='utf-8') as f:
        json.dump(content, f, indent=4, sort_keys=True, ensure_ascii=False)

def write_folder(path):
    import os
    os.mkdir(path)

def find_file(search_path, include_str=None, filter_strs=None):
    """
    查找指定目录下所有的文件（不包含以__开头和结尾的文件）或指定格式的文件，若不同目录存在相同文件名，只返回第1个文件的路径
    :param search_path: 查找的目录路径
    :param include_str: 获取包含字符串的名称
    :param filter_strs: 过滤包含字符串的名称
    """
    import os
    if filter_strs is None:
        filter_strs = []

    files = []
    # 获取路径下所有文件
    names = os.listdir(search_path)
    for name in names:
        path = search_path+'/'+name
        if os.path.isfile(path):
            # 如果不包含指定字符串则
            if include_str is not None and include_str not in name:
                continue
            # 如果未break，说明不包含filter_strs中的字符
            for filter_str in filter_strs:
                if filter_str in name:
                    break
            else:
                files.append(path)
        else:
            files += find_file(path, include_str=include_str, filter_strs=filter_strs)
    return files