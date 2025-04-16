---
layout: default
title: 学学CUDA和ML吧孩子们
---
## 导航

{% include toc.html %}

本项目的github pages托管静态网站也可以访问了：
https://archycillp.github.io/HPC-ML-HUB/Readme.html



## 使用Obsidian在本地编辑和同步本项目

#### 快速开始
- 把该项目git clone到本地。
- 确认获得了本项目编辑权限，并可以SSH到github。
	- 你可以在命令行执行`ssh -T git@github.com`，如果返回内容包含你的账户名则成功；否则可以根据这里的[教程](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?platform=windows) 来将你操作设备的SSH公钥添加到你github账户中的信任列表。
- 安装[Obsidian](https://obsidian.md/download)，用Obsidian打开本项目的根目录。
- 在偏好设置->第三方插件 中，寻找安装GitHub Sync插件。
- 在偏好中GitHub Sync插件的设置中，设置Remote URL为本项目的SSH地址。
	- `git@github.com:ArchyCillp/HPC-ML-HUB.git`
![](accessories/Pasted%20image%2020250227204331.png)
- 至此，已经安装完成，点击主界面左边的github图标对本项目进行同步（包括把同步github到本地，然后把本地修改同步到github）。
- 为了避免出现文件修改冲突，请尽可能修改自己创建的页面，如果修改了和别人合作的页面，请及时同步。
![](accessories/Pasted%20image%2020250227191223.png)
- 如果你出现“Vault is not a GIt repo or git binary cannot be found”或其他错误，说明可能是以下几个原因之一：
	- 你没有配置插件的Remote URL地址
	- 你的系统PATH环境里没有git（检测方法是打开你的命令行，输入git，看看是否已经安装了（~~某个神人竟然在windows系统上在wsl上装了git然后疑问为什么Obsidian找不到git~~））
	- 你不知怎么的把本项目下的.git文件夹删了
	- 你Obsidian打开的不是本项目的根目录（也就是本文件Readme在最外层的情况）
	- 你`ssh -T git@github.com`不能成功，ssh key没配置对

#### 注意事项
![](accessories/Pasted%20image%2020250227192004.png)
- 为了方便放置附图到accessories文件夹，并使用符合markdown标准格式的语法，请修改以上的Obsidian设置。

#### GitHub Sync插件优化
- 因为GitHub Sync插件对commit对命名很无趣，所以可以改成让deepseek帮忙给commit起名字
- 详见[[Github Sync插件自动生成commit名](HPC-ML-HUB/Obsidian相关/Github%20Sync插件自动生成commit名.html)]
- 效果图![](accessories/Pasted%20image%2020250302173910.png)
