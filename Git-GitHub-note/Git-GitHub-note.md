# Git学习笔记  

`@Author PommesPeter`  
`@Date 20200112`  
[Git Installation](#1) | [Git config](#2) | [Git WorkProcess](#3)| [Git Clone下载](#4) | [Git创建仓库](#5) | [创建或获取项目](#6) | [版本控制](#7) |[Git分支管理](#8) | [Git Remote](#9)

## 前言

#### 什么是版本控制系统？

**版本控制**是一种记录一个或若干个文件内容变化，以便将来查阅特定版本修订情况得系统。

1. 系统具体功能
2. 记录文件的所有历史变化
3. 随时可恢复到任何一个历史状态
4. 多人协作开发或修改
5. 错误恢复

#### Github和Git是什么关系

Git是版本控制软件
Github是项目代码托管的平台，借助git来管理项目代码

# <a id='1'>Git Installation</a>

---

[Click here](http://git-scm.com/downloads)

**以上为各平台的安装包下载地址**
### Linux installing Command(Ubuntu)
```
$ apt-get install libcurl4-gnutls-dev libexpat1-dev gettext \
  libz-dev libssl-dev

$ apt-get install git

$ git --version
git version 1.8.1.2
```
### Windows installation
直接点击安装包即可

# <a id='2'>Git config</a>
----
 ```
 /etc/gitconfig 文件：系统中对所有用户都普遍适用的配置。若使用 git config 时用 --system 选项，读写的就是这个文件。
 
 ~/.gitconfig 文件：用户目录下的配置文件只适用于该用户。若使用 git config 时用 --global 选项，读写的就是这个文件。
 ```
### 配置username以及eamil adress
##### 配置所用用户
```
git config --system [选项]
配置文件位置: /etc/gitconfig
```
##### 配置当前用户
```
$ git config --global user.name "runoob"
$ git config --global user.email test@runoob.com
配置文件位置: ~/.gitconfig
```
##### 配置当前项目
```
$ git config [选项]
配置文件位置: project/.git/config
```
如果用了 --global 选项，那么更改的配置文件就是位于你用户主目录下的那个，以后你所有的项目都会默认使用这里配置的用户信息。

如果要在某个特定的项目中使用其他名字或者电邮，只要去掉 --global 选项重新配置即可，新的设定保存在当前项目的 .git/config 文件里。
### 配置编译器
```
$ git config core.editor <complier>
```
### 查看配置信息
要检查已有的配置信息，可以使用 git config --list 命令：
```
$ git config --list
```

# <a id='3'>Git工作流程</a>

---
### 一般工作流程如下
- 克隆 Git 资源作为工作目录。
- 在克隆的资源上添加或修改文件。
- 如果其他人修改了，你可以更新资源。
- 在提交前查看修改。
- 提交修改。
- 在修改完成后，如果发现错误，可以撤回提交并再次修改并提交。
![](https://www.runoob.com/wp-content/uploads/2015/02/git-process.png)

##### Basic Concept
- 工作区：电脑里的目录
- 暂存区: 英文叫stage, 或index。一般存放在 ".git目录下" 下的index文件（.git/index）中，所以我们把暂存区有时也叫作索引（index）。
- 版本库：工作区有一个隐藏目录.git，这个不算工作区，而是Git的版本库。
- 仓库区: 用于备份工作区的内容
- 远程仓库: 远程主机上的GIT仓库
![](https://www.runoob.com/wp-content/uploads/2015/02/1352126739_7909.jpg)
1. 图中左侧为工作区，右侧为版本库。在版本库中标记为 "index" 的区域是暂存区（stage, index），标记为 "master" 的是 master 分支所代表的目录树。
2. 图中我们可以看出此时 "HEAD" 实际是指向 master 分支的一个"游标"。所以图示的命令中出现 HEAD 的地方可以用 master 来替换。
3. 图中的 objects 标识的区域为 Git 的对象库，实际位于 ".git/objects" 目录下，里面包含了创建的各种对象及内容。
4. 当对工作区修改（或新增）的文件执行 "git add" 命令时，暂存区的目录树被更新，同时工作区修改（或新增）的文件内容被写入到对象库中的一个新的对象中，而该对象的ID被记录在暂存区的文件索引中。
5. 当执行提交操作（git commit）时，暂存区的目录树写到版本库（对象库）中，master 分支会做相应的更新。即 master 指向的目录树就是提交时暂存区的目录树。
6. 当执行 "git reset HEAD" 命令时，暂存区的目录树会被重写，被 master 分支指向的目录树所替换，但是工作区不受影响。
7. 当执行 "git rm --cached <file>" 命令时，会直接从暂存区删除文件，工作区则不做出改变。
8. 当执行 "git checkout ." 或者 "git checkout -- <file>" 命令时，会用暂存区全部或指定的文件替换工作区的文件。这个操作很危险，会清除工作区中未添加到暂存区的改动。
9. 当执行 "git checkout HEAD ." 或者 "git checkout HEAD <file>" 命令时，会用 HEAD 指向的 master 分支中的全部或者部分文件替换暂存区和以及工作区中的文件。这个命令也是极具危险性的，因为不但会清除工作区中未提交的改动，也会清除暂存区中未提交的改动。
**Tpis:这里的HEAD类似于指针，指向哪个branch就对应哪个branch**

# <a id ='4'>Git Create repository</a>
---

### First-Step
```
git init    //init a repository and create a file named .git
            // .git file direct the repository
git init newrepo    //init in a folder called newrepo and create a file named .git
```
### Second-Step
```
git add <file>.<any>    //To add the file to the repository 
git add <file>
git commit -m 'init the project version'
```

# <a id = '5'>Git Clone</a>
---
```
git clone <repo>/<url>
git clone <repo>/<url> <directory>
```
repo:Git repository **supported url**

// eg. http://github.com/CosmosHua/locate new

directory:Local directory

# <a id = '6'>Create/Get project</a>
![](http://111.229.52.254/static/git.png)
---
### Create
```
$ mkdir test    //create a folder named test
$ cd test/      //diect to the test
$ git init      //init a repository
Initialized empty Git rephttp://111.229.52.254/static/git.pngository in /Users/tianqixin/www/runoob/.git/
# 在 /www/test/.git/ 目录初始化空 Git 仓库完毕。
```

### 添加readme文件

git add 命令可将该文件添加到缓存，如我们添加以下两个文件：
```
$ touch README
$ ls
README       
$ git status -s
?? README
$ git add README
```
// git status 命令用于查看项目的当前状态。

##### Add file:

```
touch README
git add README 
```

新项目中，添加所有文件很普遍，我们可以使用 git add . 命令来添加当前项目的所有文件。

##### Modify
```
$ vim README
```
保存退出
```
Press 'i'(edit) and enter ':wq'(save)
git add .
```
再次查看状态
```
git status -s
```
##### Add to repository
使用 git add 只是保存到了缓存区，
执行 git commit 添加到仓库中
因为每一个提交都提交名字和电子邮箱地址，记得配置
```
git commit -a跳过提交缓存 直接提交
git commit -m -m表示添加一些同步信息
commit 作用可以将文件同步
```

**说明: -m表示添加一些同步信息，表达同步内容**

##### 删除缓存和文件
```
git reset HEAD  
//用于取消已缓存的内容（相当于从缓存之中删除）
//执行 git reset HEAD 以取消之前 git add 添加，但不希望包含在下一提交快照中的缓存。
```
```
git rm <file>   //删除文件
git rm --cached //把文件从缓存区域中删除
git rm -f       //强制删除
```

##### 重命名移动文件
git mv 命令用于移动或重命名一个文件、目录、软连接
```
git mv 
```
**注意: 这两个操作会修改工作区内容，同时将操作记录提交到暂存区。**
###### 关于vim在控制台的使用

```
一、vim 有两种工作模式：

1.命令模式：接受、执行 vim操作命令的模式，打开文件后的默认模式；

2.编辑模式：对打开的文件内容进行 增、删、改 操作的模式；

3.在编辑模式下按下ESC键，回退到命令模式；在命令模式下按i，进入编辑模式
```
```
二、创建、打开文件：

1.输入 touch 文件名 ，可创建文件。
2.使用 vim 加文件路径（或文件名）的模式打开文件，如果文件存在则打开现有文件，如果文件不存在则新建文件。
3.键盘输入字母i进入插入编辑模式。
```
```
三、保存文件： 

1.在编辑模式下编辑文件 

2.按下ESC键，退出编辑模式，切换到命令模式。 

3.在命令模式下键入"ZZ"或者":wq"保存修改并且退出 vim。 

4.如果只想保存文件，则键入":w"，回车后底行会提示写入操作结果，并保持停留在命令模式。
```
```
四、放弃所有文件修改： 
1.放弃所有文件修改：按下ESC键进入命令模式，键入":q!"回车后放弃修改并退出vim。 

2.放弃所有文件修改，但不退出 vi，即回退到文件打开后最后一次保存操作的状态，继续进行文件操作：按下ESC键进入命令模式，键入":e!"，回车后回到命令模式。
```
```
五、查看文件内容：
在git窗口，输入命令：cat 文件名
```
```
六、创建文件夹
在git窗口，输入命令：touch 文件夹名
```

##### 杂项
**查看日志**
    ```
    git log --pretty = online
    ```

**比较工作区与仓库文件差异**
    ```
    git diff <file>
    ```

**将缓存区或者commit的点文件恢复到工作区**
    ```
    git checkout <commit> --<file>
    ```


# <a id = '7'>版本控制</a>
---
**版本作用：工作的时候可以设置不同的版本也可以每个人在原有代码（分支）的基础上建立自己的工作环境，单独开发，互不干扰。完成开发工作后再进行分支统一合并，有不同的版本就可以避免改代码改不回去的情况**

### 指令
- 退回上一个commit节点
```
git reset --hard HEAD^
Tips:一个^表示退回上一个版本。
```
- 退回指定的commit_id节点
```
git reset --hard <commit_id>
```
- 查看所有操作记录
```
git relog 
Tips:注意:最上面的为最新记录，可以利用commit_id去往任何操作位置
```
### 标签
Tag：想特别把一个版本标记出来记录 可以打上标签
```
git tag -a <tagname> [e.g (v1.0)]
git log --decorate              // search the tags
git tag -a <tagname> -m "Tag"   // 指定Tag信息
git tag -s <tagname> -m "Tag"   // PGP Sign
```
- -a 意思为"创建一个带注解的标签"，如果不用-a就不会记录时谁打的，什么时候打的，也不会让你添加一个带有注解的标签
- **Tpis：执行后会打开vim编辑器来让你写注解**
### e.g 
    
    $ git tag -a v0.9 85fc7e7
    $ git log --oneline --decorate --graph
    *   d5e9fc2 (HEAD -> master) Merge branch 'change_site'
    |\  
    | * 7774248 (change_site) changed the runoob.php
    * | c68142b 修改代码
    |/  
    * c1501a2 removed test.txt、add runoob.php
    * 3e92c19 add test.txt
    * 3b58100 (tag: v0.9) 第一次版本提交
    Tips:每行前面的表示版本号 对应了不同的版本号


# <a id = '8'>Git分支管理</a>
---
Branch：几乎每一种版本控制系统都以某种形式支持分支。使用分支意味着你可以从**开发主线**上分离开来，然后在不影响主线的同时继续工作。

有人把 Git 的分支模型称为必杀技特性，而正是因为它，将 Git 从版本控制系统家族里区分出来。
### Branch Management
1. Create Branch  
    ```
    git branch <branchname>
    git checkout -b <branchname>    //Instantly create and switch to this branch
    ```
2. Switch Branch
    ```
    git checkout <branchname>
    ```
    Tips:当你切换分支的时候，Git 会用该分支的最后提交的快照替换你的工作目录的内容， 所以多个分支不需要多个目录。
3. Merge Branch
    ```
    git merge <brachname>
    ```
    Tips:你可以多次合并到统一分支， 也可以选择在合并之后直接删除被并入的分支。
e.g
```
$ mkdir gitdemo
$ cd gitdemo/
$ git init
Initialized empty Git repository...
$ touch README
$ git add README
$ git commit -m 'v1'
[master (root-commit) 3b58100] v1
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 README
```
4. Show Branch
    ```
    git branch
    ```
5. Delete Branch
    ```
    git branch -d <branchname>
    ```
e.g
```
$ git branch testing
$ git branch
* master
  testing
$ ls
README
$ echo 'runoob.com' > test.txt
$ git add .
$ git commit -m 'add test.txt'
[master 3e92c19] add test.txt
 1 file changed, 1 insertion(+)
 create mode 100644 test.txt
$ ls
README        test.txt
$ git checkout testing
Switched to branch 'testing'
$ ls
README
$ git checkout master
Switched to branch 'master'
$ ls
README        test.txt
```

# <a id ='9'>Commitment History</a>
### History Record
```
git log
git log --oneline   //simple versiong (developed history)
git log --graph     //show in graph
git log --reverse   //reversing display all log
git log --author=<username>     //search someone's committment
```

### Anthor Command
```
--since / --before  // search log by time
--until / --after   // search log by time
-- no-merges        // Hidden merge committment
```

### e.g 
```
$ git log --reverse --oneline
$ git log --author=Linus --oneline -5
$ git log --oneline --before={3.weeks.ago} --after={2010-04-18} --no-merges
```
**Tpis:We can use all the --<command> flexibly**

# <a id = '9'>Remote Repository</a>
原文链接
[Click here](https://www.runoob.com/git/git-remote-repo.html)
---
### Create Remote Repository
```
git remote add <repositoryname> [url]
```
由于你的本地 Git 仓库和 GitHub 仓库之间的传输是通过SSH加密的，所以我们需要配置验证信息：

使用以下命令生成 SSH Key：
```
$ ssh-keygen -t rsa -C "youremail@example.com"
```
在github设置中添加ssh key之后输入
```
$ ssh -T git@github.com
Hi tianqixin! You've successfully authenticated, but GitHub does not provide shell access.
```
来检验是否验证成功，下面输出的结果即为验证成功

创建好新的仓库之后显示如下
![](https://www.runoob.com/wp-content/uploads/2015/03/1BCB4379-1A25-4C77-BB82-92B3E7185435.jpg)
根据提示我们就可以将本地仓库push进远程仓库里0（上传）
```
git remote add origin + ssh/https
git push -u origin master
```

### Remote Command
```
git remote
git fetch <alias>   //从远程仓库下载新分支和数据
git merge <alisa/branch>  //从远程仓库提取数据并尝试合并到当前分支
git push <alisa> <branch>   //将你的 [branch] 分支推送成为 [alias] 远程仓库上的 [branch] 分支
git remote rm <name>    //删除远程仓库
```
###### e.g
```
git fetch origin
git merge master/origin
```

Git命令
```
全局配置

git config –global user.name crperlin #git的用户名

git config –global user.email crper@outlook.com #git的登录账号

git config –global core.editor vim #设置默认编辑器

git config –global merge.tool vimdiff #设置默认的对比文件差异工具

git config –global color.status auto #显示颜色信息

git config –global color.branch auto #显示颜色信息

git config –global color.interactive auto #显示颜色信息

git config –global color.diff auto #显示颜色信息

git config –global push.default simple #简化提交

git config –list #查看配置的信息

git help config #获取帮助信息

登录git

如果上面的操作没有获取到用户配置，则只能拉取代码，不能修改;因为要想使用git，就要告诉git是谁在使用;

查看git的版本信息
git –version

获取当前登录的用户
git config –global user.name

获取当前登录用户的邮箱
git config –global user.email

设置登录用户名
git config –global user.name ‘userName’//你的github的账号名称(非邮箱)
设置登录邮箱
git config –global user.email ‘email’//你的github的账号绑定的邮箱(xxx@xxx.com)

分支管理

列出本地分支
git branch

列出远端分支
git branch -r

列出所有分支
git branch -a

查看各个分支最后一个提交对象的信息
git branch -v

查看已经合并到当前分支的分支
git branch –merge

查看未合并到当前分支的分支
git branch –no-merge

创建分支
git branch branch_name

切换分支
git checkout branch_name

创建分支并切换分支
git checkout -b branch_name

删除分支
git branch -d branch_name

在分支上提交新的版本
git commit -a -m ‘备注信息’

合并到当前分支
git merge branch_name

分支的合并后显示log
git log –oneline –graph –decorate

分支数据推送更新

origin为默认的远程仓库名,可以使用:git remote add xxx “remote_repository_URL”来添加远程仓库,所以下面命令中的origin不唯一;
master为主分支名称;

获取远端上指定分支
git fetch origin remote_branch_name

合并远端上指定分支
git merge origin remote_branch_name

推送到远端上指定分支
git push origin remote_branch_name

推送到远端上指定分支
git push origin local_branch_name:remote_branch_name

基于远端dev新建test分支
git checkout -b test origin/dev

删除远端分支[推送空分支，目前等同于删除]
git push origin :remote_branch_name

添加远程仓库
git push origin master

连接远程仓库
git remote add xxx ‘remote_repository_URL’

查看远程仓库
git remote -v

删除远程仓库
git remote rm

远程仓库相关命令

检出仓库：git clone git://github.com/jquery/jquery.git

查看远程仓库：git remote -v

添加远程仓库：git remote add
remoteRespositoryName为远程仓库标识,名称任意;

删除远程仓库：git remote rm

修改远程仓库：git remote set-url –push

拉取远程仓库：git pull remoteName] [localBranchName]

推送远程仓库：git push

如:
git push
*如果想把本地的某个分支test提交到远程仓库，并作为远程仓库的master分支，或者作为另外一个名叫test的分支，如下：

git push origin test:master //提交本地test分支作为远程的master分支

git push origin test:test //提交本地test分支作为远程的test分支

分支(branch)操作相关命令

查看本地分支：git branch

查看远程分支：git branch -r

创建本地分支：git branch [name] —-注意新分支创建后不会自动切换为当前分支

切换分支：git checkout [name]

创建新分支并立即切换到新分支：git checkout -b [name]

删除分支：git branch -d [name] —- -d选项只能删除已经参与了合并的分支，对于未有合并的分支是无法删除的。如果想强制删除一个分支，可以使用-D选项

合并分支：git merge [name] —-将名称为[name]的分支与当前分支合并

创建远程分支(本地分支push到远程)：git push origin [name]

删除远程分支：git push origin :heads/[name]或gitpush origin :[name]

创建空的分支：(执行命令之前记得先提交你当前分支的修改，否则会被强制删干净没得后悔)
git symbolic-ref HEAD refs/heads/[name]
rm .git/index
git clean -fdx

版本(tag)操作相关命令

查看版本：git tag

创建版本：git tag [name]

删除版本：git tag -d [name]

查看远程版本：git tag -r

创建远程版本(本地版本push到远程)：git push origin [name]

删除远程版本：git push origin :refs/tags/[name]

合并远程仓库的tag到本地：git pull origin –tags

上传本地tag到远程仓库：git push origin –tags

创建带注释的tag：git tag -a [name] -m ‘yourMessage‘

子模块(submodule)相关操作命令

添加子模块：git submodule add [url] [path]
如：git submodule add git://github.com/soberh/ui-libs.git src/main/webapp/ui-libs

初始化子模块：$ git submodule init —-只在首次检出仓库时运行一次就行

更新子模块：$ git submodule update —-每次更新或切换分支后都需要运行一下

删除子模块：（分4步走）

1)git rm –cached [path]
2)编辑“.gitmodules”文件，将子模块的相关配置节点删除掉
3)编辑“.git/config”文件，将子模块的相关配置节点删除掉
4)手动删除子模块残留的目录

忽略一些文件、文件夹不提交

在仓库根目录下创建名称为“.gitignore”的文件，写入不需要的文件夹名或文件，每个元素占一行即可，如
target
bin
*.db

Git 常用命令
git branch 查看本地所有分支
git status 查看当前状态
git commit 提交
git branch -a 查看所有的分支
git branch -r 查看本地所有分支
git commit -am “init” 提交并且加注释
git remote add origin git@xxx.xxx.xxx 添加远程仓库
git push origin master 将文件给推到服务器上
git remote show origin 显示远程库origin里的资源
git push origin master:develop
git push origin master:hb-dev 将本地库与服务器上的库进行关
git checkout –track origin/dev 切换到远程dev分支
git branch -D master develop 删除本地库develop
git checkout -b dev 建立一个新的本地分支dev
git merge origin/dev 将分支dev与当前分支进行合并
git checkout dev 切换到本地dev分支
git remote show 查看远程库
git add . 添加当前目录的所有文件到暂存区
git rm 文件名(包括路径) 从git中删除指定文件
git clone xxxxxxx.git 从服务器上将代码给拉下来
git config –list 看所有用户
git ls-files 看已经被提交的
git rm [file name] 删除一个文件
git commit -a 提交当前repos的所有的改变
git add [file name] 添加一个文件到git index
git commit -v 当你用－v参数的时候可以看commit的差异
git commit -m “This is the message describing the commit” 添加commit信息
git commit -a -a是代表add，把所有的change加到git index里然后再commit
git commit -a -v 一般提交命令
git log 看你commit的日志
git diff 查看尚未暂存的更新
git rm a.a 移除文件(从暂存区和工作区中删除)
git rm –cached a.a 移除文件(只从暂存区中删除)
git commit -m “remove” 移除文件(从Git中删除)
git rm -f a.a 强行移除修改后文件(从暂存区和工作区中删除)
git diff –cached 或 $ git diff –staged 查看尚未提交的更新
git stash push 将文件给push到一个临时空间中
git stash pop 将文件从临时空间pop下来
git remote add origin xxxxxx.git
git push origin master 将本地项目给提交到服务器中
git pull 本地与服务器端同步
git push (远程仓库名) (分支名) 将本地分支推送到服务器上去。
git push origin serverfix:awesomebranch
git fetch 相当于是从远程获取最新版本到本地，不会自动merge
git commit -a -m “log_message” (-a是提交所有改动，-m是加入log信息) 本地修改同步至服务器端
git branch branch_0.1 master 从主分支master创建branch_0.1分支
git branch -m branch_0.1 branch_1.0 将branch_0.1重命名为branch_1.0
git checkout branch_1.0/master 切换到branch_1.0/master分支
```

# Github学习笔记

[Basic Concept](#10) | [Git config](#) | [Git WorkProcess](#)| [Git Clone下载](#) | [Git创建仓库](#) | [创建或获取项目](#) | [版本控制](#) |[Git分支管理](#) | [Git Remote](#)

---

# <a id = '10'>基本概念</a>

1. **仓库（Repository）**
 仓库的意思，即你的项目，你想在 GitHub 上开源一个项目，那就必须要新建一个 **Repository** ，如果你开源的项目多了，你就拥有了多个 **Repositories** 。
仓库用来存放项目代码，每个项目对应一个仓库，多个开源项目则有多个仓库。

2. **收藏（Star）**
收藏项目，方便下次查看。
仓库主页**star**按钮，意思为收藏项目的人数
3. **复制克隆项目（Fork）**
你开源了一个项目，别人想在你这个项目的基础上做些改进，然后应用到自己的项目中，
这个时候他就可以 **Fork** 你的项目（打开项目主页点击右上角的fork按钮即可），然后他的 **GitHub** 主页上就多了一个项目，只不过这个项目是基于你的项目基础（本质上是在原有项目的基础上新建了一个分支），他就可以随心所欲的去改进，但是丝毫不会影响原有项目的代码与结构。
4. **发起请求（Pull Request）**
简单来说就是B**fork**了A的项目之后，发现A的项目可以有改进，然后B想改了之后告诉A，B就可以**pull request**，A收到请求之后并测试修改的地方没有问题之后可以接受**pull request**，那么B的改动就会应用到A原来的项目上。
5. **关注**
关注项目，当项目更新可以接收到通知。
这个就是观察，如果你 **Watch** 了某个项目，那么以后只要这个项目有任何更新，你都会第一时间收到关于这个项目的通知提醒。
6. **事务卡片（Issue）**
 发现**代码BUG**，但是目前没有成型代码，需要**讨论**时用；
问题的意思，举个例子，就是你开源了一个项目，别人发现你的项目中有bug，或者哪些地方做的不够好，他就可以给你提个 **Issue** ，即问题，提的问题多了，也就是 **Issues** ，然后你看到了这些问题就可以去逐个修复，修复ok了就可以一个个的 **Close** 掉。
7. **Github主页**
账号创建成功或点击网址导航栏github图标都可进入github主页：该页**左侧**主要显示**用户动态以及关注用户或关注仓库的动态**；右侧显示所有的git库
8. **仓库主页**
仓库主页主要显示项目的信息，如：**项目代码，版本，收藏/关注/fork情况**等
9. **个人主页**
个人信息：头像，个人简介，关注我的人，我关注的人，我关注的git库，我的开源项目，我贡献的开源项目等信息
