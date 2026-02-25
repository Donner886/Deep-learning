## Git Merge 与 Rebase 完全指南

*分支实际上是什么，分支只是一个指向提交的指针* 
cat .git/refs/heads/master
# 输出：a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0 
# 这就是一个 commit SHA 值

分支 = 指向某个提交的可移动指针, 
HEAD = 指向当前所在分支的指针
提交 = 包含代码快照、父提交引用、作者信息等的对象

```bash
# 查看所有分支及其指向的提交
git branch -v
# 输出：
# * master  a1b2c3d Fix login bug
#   feature b2c3d4e Add new API
#   develop c3d4e5f Update docs

# 查看详细信息
git log --oneline --graph --all --decorate
Administrator@Madong MINGW64 /d/project-ddma/Deep-learning (learngitmerge)
$ git log --oneline --graph --all --decorate
*   1b5cfbd (HEAD -> learngitmerge, origin/master, origin/HEAD, master) Merge pull request #1 from Donn
er886/chapter4/objection-location
### 当提交过一次之后
$ git log --oneline --graph
* f174c3d (HEAD -> learngitmerge) git merge study-p1
*   1b5cfbd (origin/master, origin/HEAD, master) Merge pull request #1 from Donner886/chapter4/objectio
n-location

```

#### 2.1 Merge 的方向原理
```bash
git merge <branch-name>
意思是：将 <branch-name> 合并到当前分支（HEAD 所在分支）
方向：<branch-name> → 当前分支

实例： 经过多次操作提交后， 假设有如下提交图：
        A (a1b2c3d)
        ↓
        B (b2c3d4e)
       / \
      /   \
     C     E (e5f6g7h)
     ↓     ↓
     D     master (HEAD)
     ↓
   feature

分支儲存在 heads 目錄中
每個分支檔案包含一個 40 字元的 commit SHA 值
分支只是指向某個提交的引用，不佔用額外空間
```
#### 2.3 执行 Merge：详细过程   
场景 A：将 feature 合并到 master
```bash
# 1. 确认当前在 master 分支
git branch
# * master
#   feature

# 2. 查看即将合并的内容
git log --oneline master..feature
# d4e5f6g D: Add feature 2
# c3d4e5f C: Add feature 1
# 这显示了 feature 有但 master 没有的提交
# 3. 查看反向（master 有但 feature 没有的）
git log --oneline feature..master
# e5f6g7h E: Update on master

# 4. 查看合并预览
git diff master...feature
# 显示 feature 相对于共同祖先的所有改动

# 5. 执行合并
git merge feature
这里git merge会将 feature 分支的改动合并到 当前分支master分支上面
```
Git 内部过程：
步骤 1：找到共同祖先
- master 的提交：A → B → E
- feature 的提交：A → B → C → D
- 共同祖先：B（最近的共同提交）

步骤 2：三方比较
- 基准版本（共同祖先 B）：file.txt 内容
  Initial
  Second

- 当前分支版本（master 的 E）：file.txt 内容
  Initial
  Second
  Master update

- 目标分支版本（feature 的 D）：file.txt 内容
  Initial
  Second
  Feature 1
  Feature 2

步骤 3：自动合并
Git 分析：
- B → E 的改动：添加了 "Master update"
- B → D 的改动：添加了 "Feature 1" 和 "Feature 2"
- 两者没有冲突（修改的是不同部分）

步骤 4：创建合并提交
合并后的 file.txt：
  Initial
  Second
  Feature 1
  Feature 2
  Master update

图解合并后的提交历史：
        A
        ↓
        B ← 共同祖先
       / \
      C   E
      ↓   ↓
      D   ↓
       \ /
        M (合并提交，有两个父提交：D 和 E)
        ↓
      master (HEAD)

2.4 Merge 方向的记忆技巧
```bash
# 公式：
当前分支 ← git merge ← 目标分支

# 实例：
git checkout master     # 切换到 master（当前分支）
git merge feature       # 将 feature 合并进来

结果：master 分支包含了 feature 的所有更改

# 或者理解为：
git merge <把这个分支的内容拿过来>
```

###  三、如何找到共同祖先（Merge Base）
定义： 共同祖先（merge base）是两个分支最近的共同提交，是它们历史的分叉点。
A ← B ← C ← D (master)
    ↑
    ├─ E ← F (feature)
    
共同祖先是 B
3.2 手动查找共同祖先
```bash
# 查找两个分支的共同祖先
git merge-base master feature

# 输出：b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0
# 这就是共同祖先的 commit SHA

# 查看这个提交的详细信息
git show b2c3d4e

# 或者直接：
git show $(git merge-base master feature)

``` 


