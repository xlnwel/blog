# GitHub Pages 部署指南

本指南将帮助你将博客部署到 GitHub Pages。

## 前置条件

1. 确保你的代码已经推送到 GitHub 仓库
2. 确保你的仓库是公开的（或者你有 GitHub Pro 账户以支持私有仓库的 GitHub Pages）

## 部署步骤

### 方法一：使用 GitHub Actions（推荐）

这是自动部署方法，每次推送到 `main` 分支时都会自动构建和部署。

#### 1. 启用 GitHub Pages

1. 进入你的 GitHub 仓库
2. 点击 **Settings**（设置）
3. 在左侧菜单中找到 **Pages**（页面）
4. 在 **Source**（源）部分，选择：
   - **Source**: `GitHub Actions`
5. 保存设置

#### 2. 推送代码

将代码推送到 `main` 分支：

```bash
git add .
git commit -m "配置 GitHub Pages 部署"
git push origin main
```

#### 3. 查看部署状态

1. 在仓库页面，点击 **Actions** 标签
2. 查看 "Deploy to GitHub Pages" 工作流的运行状态
3. 等待部署完成（通常需要 1-2 分钟）

#### 4. 访问你的网站

部署完成后，你的网站将在以下地址可用：

- 如果仓库名是 `username.github.io`：`https://username.github.io`
- 如果仓库名是其他名称（如 `blog`）：`https://username.github.io/blog/`

### 方法二：手动部署

如果你想手动部署，可以按照以下步骤：

1. 构建项目：
   ```bash
   npm run build
   ```

2. 如果仓库名不是 `username.github.io`，需要设置 base 路径：
   ```bash
   VITE_BASE_PATH=/仓库名/ npm run build
   ```

3. 将 `dist` 目录的内容推送到 `gh-pages` 分支

## 配置说明

### Base 路径

项目已经配置为自动检测仓库名并设置正确的 base 路径：

- **仓库名为 `username.github.io`**：base 路径为 `/`
- **其他仓库名**：base 路径为 `/仓库名/`

如果你需要手动设置 base 路径，可以在构建时设置环境变量：

```bash
VITE_BASE_PATH=/你的路径/ npm run build
```

### 自定义域名

如果你想使用自定义域名：

1. 在仓库的 `Settings` > `Pages` 中设置自定义域名
2. 在项目的 `public` 目录下创建 `CNAME` 文件，内容为你的域名
3. 配置 DNS 记录指向 GitHub Pages

## 故障排除

### 部署后页面空白

- 检查 base 路径是否正确设置
- 查看浏览器控制台的错误信息
- 确保所有资源路径都是相对路径

### 404 错误

- 确保 GitHub Pages 已正确启用
- 检查仓库设置中的 Pages 配置
- 等待几分钟让 DNS 生效

### 样式或资源加载失败

- 检查 `vite.config.ts` 中的 `base` 配置
- 确保所有资源使用相对路径
- 清除浏览器缓存后重试

## 更新内容

每次更新博客内容后：

1. 提交更改：
   ```bash
   git add .
   git commit -m "更新博客内容"
   git push origin main
   ```

2. GitHub Actions 会自动构建和部署新版本
3. 等待几分钟后访问网站查看更新

## 本地预览构建结果

在部署前，你可以本地预览构建结果：

```bash
npm run build
npm run preview
```

这将启动一个本地服务器预览构建后的网站，帮助你检查是否有问题。

