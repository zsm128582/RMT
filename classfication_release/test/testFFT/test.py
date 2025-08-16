import sys
import ast

def wordBreak(s, wordDict):
    # 将列表转换为集合以提高查询效率
    wordSet = set(wordDict)
    n = len(s)
    # dp[i] 表示 s[0:i] 是否可以被拆分为字典中的单词
    dp = [False] * (n + 1)
    dp[0] = True  # 空串可以认为已经拆分成功

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in wordSet:
                dp[i] = True
                break
    return dp[n]

def main():
    # 从标准输入中读取所有内容
    data = sys.stdin.readline().strip()
    if not data:
        return
    # 假设输入格式为："字符串",["word1","word2",...]
    parts = data.split(',', 1)
    s_str = parts[0].strip()
    # 使用 ast.literal_eval 解析字符串（处理引号）
    s = ast.literal_eval(s_str)
    
    # 解析剩余部分为列表
    list_str = parts[1].strip()
    wordDict = ast.literal_eval(list_str)
    
    print(s , wordDict)
    result = wordBreak(s, wordDict)
    # 输出要求的格式 "true" 或 "false"
    print("true" if result else "false")

if __name__ == '__main__':
    main()
