## 使用Obsidian在本地编辑和同步本项目

#### 快速开始
- 把该项目git clone到本地。
- 确认获得了本项目编辑权限，并可以SSH到github。
	- 你可以在命令行执行`ssh -T git@github.com`，如果返回内容包含你的账户名则成功；否则可以根据这里的[教程](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account?platform=windows) 来将你操作设备的SSH公钥添加到你github账户中的信任列表。
- 安装[Obsidian](https://obsidian.md/download)。
- 在偏好设置->第三方插件 中，寻找安装GitHub Sync插件。
- 在偏好中GitHub Sync插件的设置中，设置Remote URL为本项目的SSH地址。
	- `git@github.com:ArchyCillp/HPC-ML-HUB.git`
- 至此，已经安装完成，点击主界面左边的github图标对本项目进行同步（包括把同步github到本地，然后把本地修改同步到github）。
- 为了避免出现文件修改冲突，请尽可能修改自己创建的页面，如果修改了和别人合作的页面，请及时同步。
![](accessories/Pasted%20image%2020250227191223.png)

#### 注意事项
![](accessories/Pasted%20image%2020250227192004.png)
- 为了方便放置附图到accessories文件夹，并使用符合markdown标准格式的语法，请修改以上的Obsidian设置。