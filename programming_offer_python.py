# -*-coding:utf-8-*-
import math

import numpy as np


# 二维数组中的查找
def two_dim_find(target, array):
    # write code here
    for i in range(len(array)):
        for j in range(len(array[i])):
            if array[i][j] == target:
                return True

    return False


# 替换空格
def replaceSpace(s):
    return s.replace(" ", "%20")


# 从尾到头打印链表
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
    def printListFromTailToHead(listNode):
        arraylist = []
        if listNode != None:
            arraylist.append(listNode.var)
            listNode = listNode.next()
        arraylist.append(listNode.var)
        return arraylist.reverse()


# 重建二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def reConstructBinaryTree(self, pre, tin):
    if not pre or not tin:
        return None
    root = TreeNode(pre.pop(0))
    index = tin.index(root.val)
    root.left = self.reConstructBinaryTree(pre, tin[:index])
    root.right = self.reConstructBinaryTree(pre, tin[index + 1:])
    return root


# 用两个栈实现队列
class Queue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, node):
        # write code here
        self.stack1.append(node)

    def pop(self):
        # return xx
        if len(self.stack2) > 0:
            return self.stack2.pop()
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        if len(self.stack2) > 0:
            return self.stack2.pop()


# 旋转数组的最小数字
def minNumberInRotateArray(rotateArray):
    # write code here
    # 数组为空的情况
    if len(rotateArray) == 0:
        return 0
    # 数组中只有一个元素的情况
    if len(rotateArray) == 1:
        return rotateArray[0]
    # 数组包含两个及其以上元素的情况，使用二分查找算法实现
    left = 0
    right = len(rotateArray) - 1
    # 利用好非递减排序的数组这个性质
    # 循环终止条件：区间长度为1，rotateArray[right]即为最小值
    while (right - left != 1):
        mid = (left + right) // 2
        if rotateArray[left] <= rotateArray[mid]:
            left = mid  # 不能是mid - 1，因为可能会跳过最小点
        if rotateArray[right] >= rotateArray[mid]:
            right = mid
    return rotateArray[right]


# 斐波那契数列-递归
def Fibonacci_dg(n):
    if n == 0 or n == 1:
        return n
    else:
        return Fibonacci_dg(n-1) + Fibonacci_dg(n - 2)


# 斐波那契数列-非递归
def Fibonacci_no_dg(n):
    f = [0, 1, 2]
    i = 3
    while i <= n:
        f.append(f[i-1] + f[i-2])
        i += 1
    return f[n]


# 跳台阶
class Solution_Floor:
    def jumpFloor(self, number):
        # write code here
        f = [0, 1, 2]
        i = 3
        while i <= number:
            f.append(f[i-1] + f[i-2])
            i += 1
        return f[number]


# 变态跳台阶
def jumpFloorII(number):
    # write code here
    return pow(2, number-1)


# 矩阵覆盖
class Solution_M:
    def rectCover(self, number):
        f = [0, 1, 2]
        i = 3
        while i <= number:
            f.append(f[i-1] + f[i-2])
            i += 1
        return f[number]


# 二进制中1的个数
def nums_1_of_binary(n):
    count = 0
    if n < 0:
        n = n & 0xffffffff
    while n:
        count += 1
        n = (n - 1) & n

    return count


# 数值的整数次方
def Power1(base, exponent):
    return pow(base, exponent)


def Power2(base, exponent):
        # write code here
        result = 1.0
        if exponent == 0:
            result = 1
        if exponent > 0:
            for i in range(1,exponent+1):
                result *= base
        else:
            for i in range(1,0-exponent+1):
                result *= base
            result = 1/result
        return result


# 调整数组顺序使奇数位于偶数前面
def reOrderArray(array):
    array1 = []
    array2 = []
    j = len(array)
    for i in range(j):
        if array[i] % 2 != 0:
            array1.append(array[i])
        else:
            array2.append(array[i])
    return array1 + array2


# 链表中倒数第k个结点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def FindKthToTail(head, k):
    # write code here
    ans = cur = head
    for i in range(0, k):
        if not cur:
            return None
        cur = cur.next
    while cur:
        cur = cur.next
        ans = ans.next
    return ans


# 反转链表
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution_Reverse:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if pHead==None or pHead.next==None:
            return pHead
        pre = None
        cur = pHead
        while cur!=None:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre


# 合并两个排序的链表
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
def Merge(pHead1, pHead2):
    # write code here
    dummy = ListNode(0)
    pHead = dummy

    while pHead1 and pHead2:
        if pHead1.val >= pHead2.val:
            dummy.next = pHead2
            pHead2 = pHead2.next
        else:
            dummy.next = pHead1
            pHead1 = pHead1.next

        dummy = dummy.next
    if pHead1:
        dummy.next = pHead1
    elif pHead2:
        dummy.next = pHead2

    return pHead.next


# 树的子结构
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if pRoot2 is None:
            return False
        if self.pretree2str(pRoot2) in self.pretree2str(pRoot1) and self.midtree2str(pRoot2) in self.midtree2str(pRoot2):
            return True
        else:
            return False
    def pretree2str(self,root):
        if not root:
            return ''
        re = ''
        re+=str(root.val)
        if root.left or  root.right:
            re += self.pretree2str(root.left) + self.pretree2str(root.right)
        return re
    def midtree2str(self,root):
        if not root:
            return ''
        re = ''
        re+=str(root.val)
        if root.left or  root.right:
            re = self.midtree2str(root.left) +re+ self.midtree2str(root.right)
        return re


# 二叉树的镜像
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution2:
    def Mirror(self, root):
        # write code here
        if not root:
            return root
        node = root.left
        root.left = root.right
        root.right = node
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root


# 顺时针打印矩阵
class Solution3:
    def printMatrix(self, matrix):
        # write code here
        s=[]
        while matrix:
            s+=matrix[0]
            del matrix[0]
            matrix=zip(*matrix)[::-1]
        return s


# 包含min函数的栈
class Solution4:
    def __init__(self):
        self.stack = []
        self.assist = []

    def push(self, node):
        min = self.min()
        if not min or node < min:
            self.assist.append(node)
        else:
            self.assist.append(min)
        self.stack.append(node)

    def pop(self):
        if self.stack:
            self.assist.pop()
            return self.stack.pop()

    def top(self):
        # write code here
        if self.stack:
            return self.stack[-1]

    def min(self):
        # write code here
        if self.assist:
            return self.assist[-1]


# 栈的压入、弹出序列
class Solution5:
    def IsPopOrder(self, pushV, popV):
        # write code here
        stack = []
        j = 0
        for i in pushV:
            stack.append(i)
            while len(stack)>0 and popV[j] == stack[len(stack)-1]:
                stack.pop()
                j+=1
        return len(stack)==0


# 从上往下打印二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution7:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return []
        queue = []
        result = []

        queue.append(root)
        while len(queue) > 0:
            node = queue.pop(0)
            result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result


# 二叉搜索树的后序遍历序列
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution8:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if len(sequence) == 0:
            return False

        root = sequence[-1]

        # 在二叉搜索中左子树的结点小于跟结点
        i = 0
        for node in sequence[:-1]:
            if node > root:
                break
            i += 1

        # 在二叉搜索中右子树的结点小于跟结点
        for node in sequence[i:-1]:
            if node < root:
                return False

        # 判断左子树是不是二叉搜索树
        left = True
        if i > 1:
            left = self.VerifySquenceOfBST(sequence[:i])

        right = True
        if i < len(sequence) - 2 and left:
            right = self.VerifySquenceOfBST(sequence[i + 1:-1])

        return left and right


# 二叉树中和为某一值的路径
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution_Path_Sum:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        res=[]
        treepath=self.dfs(root)
        for i in treepath:
            if sum(map(int,i.split('->')))==expectNumber:
                res.append(list(map(int,i.split('->'))))
        return res

    def dfs(self, root):
        if not root: return []
        if not root.left and not root.right:
            return [str(root.val)]
        treePath = [str(root.val) + "->" + path for path in self.dfs(root.left)]
        treePath += [str(root.val) + "->" + path for path in self.dfs(root.right)]
        return treePath


# 复杂链表的复制
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

class Solution43:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here

        head = pHead
        p_head = None
        new_head = None

        random_dic = {}
        old_new_dic = {}

        while head:
            node = RandomListNode(head.label)
            node.random = head.random
            old_new_dic[id(head)] = id(node)
            random_dic[id(node)] = node
            head = head.next

            if new_head:
                new_head.next = node
                new_head = new_head.next
            else:
                new_head = node
                p_head = node

        new_head = p_head
        while new_head:
            if new_head.random != None:
                new_head.random = random_dic[old_new_dic[id(new_head.random)]]
            new_head = new_head.next
        return p_head


# 二叉搜索树与双向链表
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution44:
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree: return
        self.arr = []
        self.midTraversal(pRootOfTree)
        for i, v in enumerate(self.arr[:-1]):
            v.right = self.arr[i + 1]
            self.arr[i + 1].left = v
        return self.arr[0]

    def midTraversal(self, root):
        if not root: return
        self.midTraversal(root.left)
        self.arr.append(root)
        self.midTraversal(root.right)


# 字符串的排列
class Solution45:
    def Permutation(self, ss):
        if len(ss) <= 1:
            return ss
        res = set()
        # 遍历字符串，固定第一个元素，第一个元素可以取a,b,c...，然后递归求解
        for i in range(len(ss)):
            for j in self.Permutation(ss[:i] + ss[i+1:]): # 依次固定了元素，其他的全排列（递归求解）
                res.add(ss[i] + j) # 集合添加元素的方法add(),集合添加去重（若存在重复字符，排列后会存在相同，如baa,baa）
        return sorted(res)         # sorted()能对可迭代对象进行排序,结果返回一个新的list


# 数组中出现次数超过一半的数字
def MoreThanHalfNum_Solution(numbers):
    dict = {}
    for no in numbers:
        if not dict.has_key(no):
            dict[no] = 1
        else:
            dict[no] = dict[no] + 1
        if dict[no] > len(numbers)/2:
            return no
    return 0


# 最小的K个数
def GetLeastNumbers_Solution(i, k):
    if k > len(i):
        return []
    return sorted(i)[0:k]


# 连续子数组的最大和
class Solution9:
    def FindGreatestSumOfSubArray(self, array):
        if not array:
            return 0
        res,cur=array[0],array[0]
        for i in array[1:]:
            cur+=i
            res=max(res,cur)
            if cur<0:
                cur=0
        return res


# 整数中1出现的次数（从1到n整数中1出现的次数）
class Solution10:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        str_n = ''
        for i in range(1, n + 1):
            str_n += str(i)
        res = str_n.count('1')
        return res


# 把数组排成最小的数
class Solution11:
    def PrintMinNumber(self, numbers):
        if numbers == []:
            return ""
        else:
            numbers = [str(i) for i in numbers]
            numbers.sort(cmp = lambda x,y : int(x+y)-int(y+x))
            return int(''.join(numbers))
        # write code here


# 丑数
def GetUglyNumber_Solution(self, index):
    res=[2**i*3**j*5**k for i in range(30) for j in range(20)  for k in range(15)]
    return sorted(res)[index-1] if index else 0


# 第一个只出现一次的字符位置
def FirstNotRepeatingChar(s):
    if s == "":
        return -1
    else:
        counts = {}
        for i in s:
            if i not in counts:
                counts[i] = 1
            else:
                counts[i] += 1
        for index, i in enumerate(s):
            if counts[i] == 1:
                return index


# 数组中的逆序对
count = 0
class Solution41:
    def InversePairs(self, data):
        global count
        def MergeSort(lists):
            global count
            if len(lists) <= 1:
                return lists
            num = int( len(lists)/2 )
            left = MergeSort(lists[:num])
            right = MergeSort(lists[num:])
            r, l=0, 0
            result=[]
            while l<len(left) and r<len(right):
                if left[l] < right[r]:
                    result.append(left[l])
                    l += 1
                else:
                    result.append(right[r])
                    r += 1
                    count += len(left)-l
            result += right[r:]
            result += left[l:]
            return result
        MergeSort(data)
        return count%1000000007


# 两个链表的第一个公共结点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution12:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if not pHead1 or not pHead2:
            return None

        stack1 = []
        stack2 = []

        while pHead1:
            stack1.append(pHead1)
            pHead1 = pHead1.next

        while pHead2:
            stack2.append(pHead2)
            pHead2 = pHead2.next

        first = None
        while stack1 and stack2:
            top1 = stack1.pop()
            top2 = stack2.pop()
            if top1 is top2:
                first = top1
            else:
                break
        return first


# 数字在排序数组中出现的次数
class Solution13:
    # 二分法找到k值的位置
    def BinarySearch(self, data, mlen, k):
        start = 0
        end = mlen - 1
        while start <= end:
            mid = (start + end) / 2
            if data[mid] < k:
                start = mid + 1
            elif data[mid] > k:
                end = mid - 1
            else:
                return mid
        return -1

    def GetNumberOfK(self, data, k):
        # write code here
        mlen = len(data)
        # 先使用二分法找到k值的位置
        index = self.BinarySearch(data, mlen, k)
        if index == -1:
            return 0
        # 分别向该位置的左右找
        count = 1
        for i in range(1, mlen):
            if index + i < mlen and data[index + i] == k:
                count += 1
            if index - i >= 0 and data[index - i] == k:
                count += 1
        return count


# 二叉树的深度
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution14:
    def TreeDepth(self, pRoot):
        # write code here
        # 使用层次遍历
        # 当树为空直接返回0
        if pRoot is None:
            return 0
        count = max(self.TreeDepth(pRoot.left), self.TreeDepth(pRoot.right)) + 1
        return count


# 平衡二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution15:
    def IsBalanced_Solution(self, root):
        if not root:
            return True
        if abs(self.maxDepth(root.left) - self.maxDepth(root.right)) > 1:
            return False
        return self.IsBalanced_Solution(root.left) and self.IsBalanced_Solution(root.right)

    def maxDepth(self, root):
        if not root: return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1


# 数组中只出现一次的数字
class Solution17:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        num_dict = {}
        for i in range(len(array)):
            if str(array[i]) not in num_dict:
                num_dict[str(array[i])] = 1
            else:
                del num_dict[str(array[i])]
        return num_dict.keys()


# 和为S的连续正数序列
class Solution18:
    def FindContinuousSequence(self, tsum):
        # write code here
        res=[]
        for i in range(1,tsum//2+1):
            sumRes=i
            for j in range(i+1,tsum//2+2):
                sumRes+=j
                if sumRes==tsum:
                    res.append(list(range(i,j+1)))
                    break
                elif sumRes>tsum:
                    break
        return res


# 和为s的两个正数
class Solution19:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        for i in array:
            if tsum-i in array:
                return [i,tsum-i]
        return []


# 左旋转字符串
def LeftRotateString(s, n):
    # write code here
    return s[n:len(s)] + s[0:n]


# 翻转单词顺序列
def ReverseSentence(s):
    s_list = s.split(" ")
    s_list.reverse()
    return " ".join(str(d) for d in s_list)


# 扑克牌顺子
class Solution20:
    def IsContinuous(self, numbers):
        # write code here
        if len(numbers):
            while min(numbers)==0:
                numbers.remove(0)
            if max(numbers) - min(numbers)<=4 and len(numbers)==len(set(numbers)):
                return True
        return False


# 孩子们的游戏(圆圈中最后剩下的数)
class Solution21:
    def LastRemaining_Solution(self, n, m):
        if n<=0 or m<=0:
            return -1
        fn =0 #f1=0
        for i in range(2,n+1):
            fn = (fn +m)% i
        return fn


# 求1+2+3+...+n
def Sum_Solution(self, n):
    # write code here
    return (1+n) * n /2


# 不用加减乘除做加法:异或
class Solution22:
    def Add(self, a, b):
        while(b):
           a, b = (a ^b) & 0xFFFFFFFF, ((a & b) << 1) & 0xFFFFFFFF
        return a if a <= 0x7FFFFFFF else ~(a ^0xFFFFFFFF)


# 把字符串转换成整数
class Solution23:
    def StrToInt(self, s):
        # write code here
        numlist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-']
        sum = 0
        label = 1  # 正负数标记
        if s == '':
            return 0
        for string in s:
            if string in numlist:  # 如果是合法字符
                if string == '+':
                    label = 1
                    continue
                if string == '-':
                    label = -1
                    continue
                else:
                    sum = sum * 10 + numlist.index(string)
            if string not in numlist:  # 非合法字符
                sum = 0
                break  # 跳出循环
        return sum*label


# 数组中重复的数字
class Solution24:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers,duplication):
        dic = {}
        for num in numbers:
            if not num in dic:
                dic[num] = 1
            else:
                dic[num] += 1
        for num in numbers:
            if dic[num] != 1:
                duplication[0] = num
                return True
        return False


# 构建乘积数组
def multiply(A):
    B = []
    for i in range(len(A)):
        sum = 1
        for j in range(len(A)):
            if i == j:
                sum = sum
            else:
                sum = sum * A[j]
        B.append(sum)
    return B


# 正则表达式的匹配
class Solution25:
    def match(self, s, pattern):
        if s == '' and pattern == '':  #两个字符串匹配到为空则为真
            return True
        if s != '' and pattern == '':  #s不为空，p为空则为假
            return False
        if len(pattern) < 2:     #p的长度为1时，为了下面不会超出下标抛出错误
            if s == pattern[0] or (s != '' and pattern[0] == '.'):  #匹配整个s，不匹配则为假
                return self.match(s[1:], pattern[1:])
            else:
                return False
        if pattern[1] != "*":
            if s == '':      #第二个不为*且s为空则为假
                return False
            if s[0] == pattern[0] or (s != '' and pattern[0] == '.'):  #当前字符为真，则递归查找剩余子串
                return self.match(s[1:], pattern[1:])
            else:
                return False
        else:             #第二个为*
            if s == '' and len(pattern) == 2:   #长度刚好为2，相当于匹配0个
                return True
            elif s == '' and len(pattern) > 2:  #大于2个就先减少前两个（匹配0个）
                return self.match(s, pattern[2:])
            elif s[0] == pattern[0] or (s != '' and pattern[0] == '.'): #当前字符匹配的话，pattern往左移两位或者s往左移一味
                return self.match(s, pattern[2:]) or self.match(s[1:], pattern)
            else:
                return self.match(s, pattern[2:])   #不匹配则把这个*当做匹配0个字符


# 表示数值的字符串
import re
class Solution26:
    def isNumeric(self, s):
        return re.match(r"^[\+\-]?[0-9]*(\.[0-9]*)?([eE][\+\-]?[0-9]+)?$",s)


# 字符流中第一个不重复的字符
class Solution27:

    def __init__(self):
        self.s = ""

    def FirstAppearingOnce(self):
        res = list(filter(lambda c: self.s.count(c) == 1, self.s))
        return res[0] if res else "#"

    def Insert(self, char):
        self.s += char


# 链表中环的入口结点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution28:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        linkls = []
        while pHead:
            if pHead in linkls:
                return pHead
            linkls.append(pHead)
            pHead = pHead.next
        return None


# 删除链表中的重复节点
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Solution29:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead == None or pHead.next == None:
            return pHead
        new_head = ListNode(-1)
        new_head.next = pHead
        pre = new_head
        p = pHead
        nex = None
        while p != None and p.next != None:
            nex = p.next
            if p.val == nex.val:
                while nex != None and nex.val == p.val:
                    nex = nex.next
                pre.next = nex
                p = nex
            else:
                pre = p
                p = p.next
        return new_head.next


# 二叉树的下一个节点
class TreeLinkNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.next = None
class Solution30:
    def GetNext(self, pNode):
        # write code here
        if pNode.right:#有右子树
            p=pNode.right
            while p.left:
                p=p.left
            return p
        while pNode.next:#无右子树，则找第一个当前节点是父节点左孩子的节点
            if(pNode.next.left==pNode):
                return pNode.next
            pNode = pNode.next#沿着父节点向上遍历
        return  #到了根节点仍没找到，则返回空


# 对称的二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution31:
    def isSymmetrical(self, pRoot):
        if not pRoot:
            return True
        return self.compare(pRoot.left, pRoot.right)

    def compare(self, pRoot1, pRoot2):
        if not pRoot1 and not pRoot2:
            return True
        if not pRoot1 or not pRoot2:
            return False
        if pRoot1.val == pRoot2.val:
            if self.compare(pRoot1.left, pRoot2.right) and self.compare(pRoot1.right, pRoot2.left):
                return True
        return False


# 按照之字形打印二叉树
'''
解法：
利用一个标志变量flag来标记从左往右还是从右往走
如果从左往右，那就从头到尾遍历当前层的节点current_nodes，然后将左孩子和右孩子分别append到一个list new_nodes中
如果从右往前，那就从尾到头遍历当前层的节点current_nodes，然后将右孩子和左孩子分别insert到一个list new_nodes中
这样得到的new_nodes还是从左到右有序的
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution32:
    def Print(self, pRoot):
        # write code here
        if pRoot == None:
            return []
        falg = 0 # 0表示从左往右，1表示从右往左
        node_list = [[pRoot]]
        result = []
        while node_list:
            current_nodes = node_list[0] # 当前层的节点
            node_list = node_list[1:]
            new_nodes = [] # 下一层的节点，按照从左往右的顺序存储
            res = [] # 当前层得到的输出
            while len(current_nodes) > 0:
                # 从左往右
                if falg == 0:
                    res.append(current_nodes[0].val)
                    if current_nodes[0].left != None:
                        new_nodes.append(current_nodes[0].left)
                    if current_nodes[0].right != None:
                        new_nodes.append(current_nodes[0].right)
                    current_nodes = current_nodes[1:]
                # 从右往左
                else:
                    res.append(current_nodes[-1].val)
                    if current_nodes[-1].right != None:
                        new_nodes.insert(0, current_nodes[-1].right)
                    if current_nodes[-1].left != None:
                        new_nodes.insert(0, current_nodes[-1].left)
                    current_nodes = current_nodes[:-1]
            result.append(res)
            falg = 1 - falg
            if new_nodes:
                node_list.append(new_nodes)
        return result


# 把二叉树打印成多行
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution33:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        if not pRoot:
            return []
        nodeStack = [pRoot]
        result = []
        while nodeStack:
            res = []
            nextStack = []
            for i in nodeStack:
                res.append(i.val)
                if i.left:
                    nextStack.append(i.left)
                if i.right:
                    nextStack.append(i.right)
            nodeStack = nextStack
            result.append(res)
        return result


# 序列化二叉树
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution34:
    def Serialize(self, root):
        # write code here
        return root
    def Deserialize(self, s):
        # write code here
        return s


# 二叉搜索树的第k个节点
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution35:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        self.res = []
        self.dfs(pRoot)
        return self.res[k-1] if 0 < k <= len(self.res) else None
    def dfs(self,root):
        if not root: return
        self.dfs(root.left)
        self.res.append(root)
        self.dfs(root.right)


# 数据流中的中位数
class Solution36:
    def __init__(self):
        self.data = []
    def Insert(self, num):
        # write code here
        self.data.append(num)
        self.data.sort()
    def GetMedian(self,data):
        # write code here
        length = len(self.data)
        if length % 2 == 0:
            return (self.data[length // 2] + self.data[length // 2-1]) / 2.0
        else:
            return self.data[int(length // 2)]


# 滑动窗口里的最大值
class Solution37:
    def maxInWindows(self, num, size):
        # write code here
        if size <= 0:
            return []
        m = []
        for i in xrange(len(num) - size + 1):
            s = max(num[i: i+size])
            m.append(s)

        return m


# 矩阵中的路径
# Python Solution
# 构造board二维数组来存储原有一维数组
# 利用dict列表来存储已经经过的路径
# 利用word来存储，还需遍历的字符
# 构造4个方向的递归，设置截至条件 word == ''
class Solution38:
    def hasPath(self, board, row, col, word):
            self.col, self.row = col, row
            board = [list(board[col * i:col * i + col]) for i in range(row)]
            for i in range(row):
                for j in range(col):
                    if board[i][j] == word[0]:
                        self.b = False
                        self.search(board, word[1:], [(i, j)], i, j)
                        if self.b:
                            return True
            return False
    def search(self, board, word, dict, i, j):
            if word == "":
                self.b = True
                return
            if j != 0 and (i, j - 1) not in dict and board[i][j - 1] == word[0]:
                self.search(board, word[1:], dict + [(i, j - 1)], i, j - 1)
            if i != 0 and (i - 1, j) not in dict and board[i - 1][j] == word[0]:
                self.search(board, word[1:], dict + [(i - 1, j)], i - 1, j)
            if j != self.col - 1 and (i, j + 1) not in dict and board[i][j + 1] == word[0]:
                self.search(board, word[1:], dict + [(i, j + 1)], i, j + 1)
            if i != self.row - 1 and (i + 1, j) not in dict and board[i + 1][j] == word[0]:
                self.search(board, word[1:], dict + [(i + 1, j)], i + 1, j)


# 机器人的运动范围
'''
思路：将地图全部置1，遍历能够到达的点，将遍历的点置0并令计数+1.这个思路在找前后左右相连的点很有用，
比如leetcode中的海岛个数问题/最大海岛问题都可以用这种方法来求解。
'''
class Solution39:
    def __init__(self):
        self.count = 0

    def movingCount(self, threshold, rows, cols):
        # write code here
        arr = [[1 for i in range(cols)] for j in range(rows)]
        self.findway(arr, 0, 0, threshold)
        return self.count

    def findway(self, arr, i, j, k):
        if i < 0 or j < 0 or i >= len(arr) or j >= len(arr[0]):
            return
        tmpi = list(map(int, list(str(i))))
        tmpj = list(map(int, list(str(j))))
        if sum(tmpi) + sum(tmpj) > k or arr[i][j] != 1:
            return
        arr[i][j] = 0
        self.count += 1
        self.findway(arr, i + 1, j, k)
        self.findway(arr, i - 1, j, k)
        self.findway(arr, i, j + 1, k)
        self.findway(arr, i, j - 1, k)


# 剪绳子
class Solution40:
    def cutRope(self, n):
        # write code here
        if n < 4:return n-1
        i = n//2
        a, b = n//i, n % i
        maxM = ((a+1)**b) * (a**(i-b))
        return maxM


if __name__ == "__main__":
    pass


