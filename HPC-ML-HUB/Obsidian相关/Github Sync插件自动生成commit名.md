---
title: Github Sync插件自动生成commit名
---

- 因为GitHub Sync插件对commit对命名很无趣，所以可以改成让deepseek帮忙给commit起名字
- 如果你想要这个功能，只需要用plugins/github-sync/main.js文件 替换 本项目下.obsidian/plugins/github-sync/main.js文件
- 因为涉及到硅基流动上deepseek的API，所以main.js中第4735行的`'Authorization': 'Bearer <deepseek的密钥>'`要进行替换，请使用自己的硅基流动密钥或者联系项目管理员。