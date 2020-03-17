Welcome to my CodeWorld

# 个人常用git bash命令
初始配置
git config --global user.name author

git config --global user.email author@xxx.com

git clone xxxxx //拷贝远程仓库到本地

git add -A//添加所有变化到暂存区

git commit -m“message”//备注message信息

git push (origin master)//推到远程仓库

git pull //拉取同步远处仓库

git status//查看暂存区历史信息

1. 当我们在github版本库中发现一个问题后，你在github上对它进行了在线的修改；或者你直接在github上的某个库中添加readme文件或者其他什么文件，但是没有对本地库进行同步。这个时候当你再次有commit想要从本地库提交到远程的github库中时就会出现push失败的问题。
2. 当github中的README.md文件不在本地代码目录中。
也就是说我们需要先将远程代码库中的任何文件先pull到本地代码库中，才能push新的代码到github代码库中。
git pull --rebase origin master
git push -u origin master
