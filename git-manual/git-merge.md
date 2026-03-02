### Use "git add <file>...." to include in what will be committed.


## Git Merge 与 Rebase 完全指南

*分支实际上是什么，分支只是一个指向提交的指针* 
cat .git/refs/heads/master
# 输出：a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0 
# 这就是一个 commit SHA 值

分支 = 指向某个提交的可移动指针,  
HEAD = 指向当前所在分支的指针, HEAD指的是当前分支的最新提交
提交 = 包含代码快照、父提交引用、作者信息等的对象

A branch (e.g. master) is a named, movable pointer to a commit, stored under refs/heads/....
HEAD is your current position pointer: usually it points to a branch name (symbolic ref), not directly to a commit.
So the usual chain is: HEAD → master → latest commit.
When you git commit on a normal branch, Git moves that branch forward, and HEAD follows because it points to that branch.
When you git switch <branch>, HEAD is repointed to another branch.
In detached HEAD state (git switch --detach <commit>), HEAD points directly to a commit; no branch moves with new commits unless you create one.


|git branch的常见用法，使用频率最高的几个用法|说明|
|-|-|
|git branch 主要用来 查看， 创建， 删除， 重命名分支|
|查看| git branch: 查看本地分支, 会列出本地分支，当前分支前面会加*  |
|查看| git branch -a: 查看本地分支 +  远程分支, 会列出本地分支，当前分支前面会加*  |
|查看| git branch -r: 只查看远程分支，当前分支前面会加*  |
|查看| *git branch -vv: 查看本地分支及其指向的最新提交，显示分支的最新提交信息（短SHA + 提交信息）*  |
|创建|git branch <branch> 创建分支不切换，更常见的工作流其实是“创建并切换”，一般大家会用 git switch -c <branch>; 仅仅切换为 git switch <branch>|
|删除| 删除本地分支（已合并的安全删法 git branch -d <branch> |
|删除| 强制删除（确认不要了才用） git branch -D <branch> |
|重命名| 重命名当前分支 git branch -m new-name  |
|重命名| 重命名指定分支 git branch -m old-name new-name|



# 查看git log详情
|git log 用来查看提交历史| 说明 |
|-|-|
|1） 查看最近提交 | git log: 查看当前分支提交历史。 从当前 HEAD 往回看详细历史（作者、日期、提交说明、diff 摘要等）|
|2）单行精简显示 | git log --oneline: 每个提交只显示一行，包含简短的 SHA 和提交信息。 常搭配限制条数：-n 20 避免刷屏|
|3）带分支/标签信息 | git log --oneline --decorate: 显示提交历史，并在提交旁边显示分支和标签信息, 确认“我现在在哪个提交点、和远端/标签关系是什么”。| 
|4）图形化显示分叉合并 | git log --oneline --graph --decorate --all: 以 ASCII 图形方式显示分叉和合并关系，配合 --decorate 显示分支标签信息，清晰展示分支结构。|
|5) 对比“我落后/领先远端多少提交”（日常同步很常用）| git log --oneline --decorate --left-right origin/master...HEAD: 显示当前分支（HEAD）和远端 master 分支的差异，左侧显示远端有但本地没有的提交（<），右侧显示本地有但远端没有的提交（>）。 ... 两边不共有的提交（对称差集），常用于比较分支差异|
|6) 对比“我落后/领先远端多少提交”（日常同步很常用）| git log --oneline --decorate --left-right origin/master..HEAD: 显示当前分支（HEAD）和远端 master 分支的差异，左侧显示远端有但本地没有的提交（<），右侧显示本地有但远端没有的提交（>）。 .. 只显示本地有但远端没有的提交（右侧 >），常用于查看本地新增提交|



## Merge 的方向原理
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
或者： 
  $ git log --oneline --graph --decorate --all
  * 3c72c63 (HEAD -> feature) add line 4 to feature branch
  | * e75b83a (master) add 3rd comment
  |/
  * ab43cfd  master branch - verion 1



分支儲存在 .git/ref/heads 目錄中
每個分支檔案包含一個 40 字元的 commit SHA 值
分支只是指向某個提交的引用，不佔用額外空間
```
## 2.3 执行 Merge：详细过程   
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
### ... 是对称差集，显示两个分支不共有的提交
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

步骤 3：自动合并 (merge)
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

##  三、如何找到共同祖先（Merge Base）
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


#### git merge 与 rebase 的区别
git rebase 的字面意思是 “重新定义基点”。它的作用是把一个分支的修改，在另一个分支的最新提交之后“重新放一遍”。
他的核心目的是保持提交历史的线性，避免分叉。 通过将当前分支的提交一个个地复制到新的基点上，rebase 可以让历史看起来像是没有分叉过一样。
1. 原理图解 (ASCII)
假设你从 main 分支的 B 点分出了 feature 分支。在你开发 D, E 的同时，同事往 main 提交了 C。
        A
        ↓
        B ← main
       / \
      D   C
      ↓
      E
      feature 

执行 git rebase main 后，feature 分支会被“重放”在 C 之后，形成新的提交 D' 和 E'。
        A
        ↓
        B ← main
         \
          C
           \
            D' ← feature
             \
              E'  
2. 主要区别
|特性|git merge|git rebase| 
|-|-|-|
|历史结构|保留分叉，显示完整的分支和合并历史|线性历史，重写提交历史，丢失分叉信息|
|合并提交|会生成一个新的合并提交，包含两个父提交|不会生成合并提交，直接将提交重放到新基点上|
|冲突处理|在合并时处理冲突，可能需要解决一次|在重放每个提交时处理冲突，可能需要多次解决|
|适用场景|需要保留完整历史，展示分叉和合并关系|希望保持历史清晰，避免分叉，适合个人分支的更新|

3. 日常最常见的 3 种用法
用法 A：同步远程主干 (最常用)
当你在自己的分支开发时，主干（main）更新了，你想把主干的最新改动同步过来，同时保持你的提交在最前面。
```bash
git fetch origin
git rebase origin/main
```
用法 C：git pull --rebase
这是 git pull 的一个变体。普通 pull 是 fetch + merge（容易产生没意义的 merge commit）；--rebase 是 fetch + rebase。 场景：本地有提交还没 push，远端也有新提交，用这个命令能让你的本地提交永远在远端之后，避免历史交错。

**Golden Rule: 不要对已经推送到公共分支（且别人也在用的）提交进行 rebase！**