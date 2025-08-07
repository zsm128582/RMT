# def build_kmp_table(pattern):
#     """构建 KMP 算法的回退表"""
#     table = [0] * len(pattern)
#     j = 0
#     for i in range(1, len(pattern)):
#         while j > 0 and pattern[i] != pattern[j]:
#             j = table[j-1]
#         if pattern[i] == pattern[j]:
#             j += 1
#         table[i] = j
#     return table

# def count_sensitive_words(text, sensitive_words):
#     """计算字符串中所有敏感词的出现次数"""
#     total_count = 0
    
#     for pattern in sensitive_words:
#         # 初始化 KMP 回退表
#         kmp_table = build_kmp_table(pattern)
#         count = 0
#         j = 0  # 模式串指针
        
#         # 遍历文本
#         for i in range(len(text)):
#             while j > 0 and text[i] != pattern[j]:
#                 j = kmp_table[j-1]
#             if text[i] == pattern[j]:
#                 j += 1
#             if j == len(pattern):
#                 count += 1
#                 j = kmp_table[j-1]  # 继续查找可能的重叠匹配
        
#         total_count += count
    
#     return total_count


# # 输入
# n , m  = map(int,input().split())    # n 为字符串长度 , m为敏感词列表长度
# text = input().strip()  # 输入字符串
# sensitive_words = [input().strip() for _ in range(m)]  # 敏感词列表

# # 计算并输出结果
# result = count_sensitive_words(text, sensitive_words)
# print(result)

import re

def count_sensitive_words(text: str, sensitive_words: list[str]) -> int:
    """
    用正则表达式统计所有敏感词（含重叠）的总出现次数。
    正则语法 (?=pattern) 表示零宽度前瞻，不会消耗字符，
    因此能够在同一位置继续尝试后续匹配，从而数到重叠片段。
    """
    return sum(
        len(   re.findall(  fr'(?={re.escape(word)})'  , text  )  )
        for word in sensitive_words
    )

# ------------------ DEMO ------------------
if __name__ == "__main__":
    n, m = map(int, input().split())
    text = input().strip()
    sensitive_words = [input().strip() for _ in range(m)]

    print(count_sensitive_words(text, sensitive_words))
