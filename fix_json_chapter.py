import json
import re

def fix_chapter_titles(file_path):
    """
    修复JSON文件中缺失的章节标题。

    Args:
        file_path (str): JSON文件的路径。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"读取或解析JSON文件时出错: {e}")
        return

    for item in data:
        chapter_title = item.get('chapter_title')
        if not chapter_title or not chapter_title.startswith('第'):
            context = item.get('context', '')
            # 优先匹配 "第X章 " 后面的部分
            # 匹配 "第X章 标题"
            match = re.search(r'(第\d+章\s+[^\s]+)', context)
            if match:
                item['chapter_title'] = match.group(1)
            else:
                # 如果没有章节号，尝试提取context开头的第一个词作为标题
                # 假设标题和正文之间有空格
                parts = context.split(' ', 1)
                if parts:
                    title_candidate = parts[0].strip()
                    if title_candidate:
                        item['chapter_title'] = title_candidate
            
            # 删除 status, context, 和 reason 字段
            
    


    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"文件 '{file_path}' 已成功修复并保存。")
    except IOError as e:
        print(f"写入文件时出错: {e}")
def change_chapter_titles(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"读取或解析JSON文件时出错: {e}")
        return
    try:
        for item in data:
            if 'status' in item:
                del item['status']
            if 'context' in item:
                del item['context']
            if 'reason' in item:
                del item['reason']
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"读取或解析JSON文件时出错: {e}")
        return
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"文件 '{file_path}' 已成功修复并保存。")
    except IOError as e:
        print(f"写入文件时出错: {e}")

if __name__ == '__main__':
    a=input("请输入1修复章节标题，请输入2删除多余内容")
    if a=='1':
        fix_chapter_titles('test.json')
    elif a=='2':
        change_chapter_titles('test.json')
