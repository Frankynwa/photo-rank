---
kind: frontend_style
name: PhotoRank 前端样式系统：内联 CSS + 原生 HTML/JS 单页应用
category: frontend_style
scope:
    - '**'
source_files:
    - templates/index.html
---

## 1. 使用的系统与方案
- 纯静态单页应用（SPA）：所有前端代码集中在 `templates/index.html` 一个文件中，包含 HTML 结构、CSS 样式与 JavaScript 逻辑。
- 无外部 CSS 框架或预处理器：未使用 Bootstrap、Tailwind、SCSS/Less 等，全部采用原生 CSS 与内联 `<style>` 标签。
- 无组件库或设计系统：没有引入 React/Vue/AntD/Material 等 UI 库，也没有设计令牌（design tokens）文件。
- 响应式策略：通过 CSS Grid（`grid-template-columns: repeat(auto-fill, minmax(250px, 1fr))`）和 `flex-wrap` 实现自适应布局，配合 `viewport` meta 标签适配移动端。

## 2. 核心文件与位置
- `templates/index.html`：唯一的前端入口，包含完整的页面结构、样式与交互逻辑。
- 无独立的 `.css`、`.scss`、`.js` 文件；所有资源均内联在 HTML 中。

## 3. 架构与约定
- 样式组织方式：按功能模块划分 CSS 类名，如 `.header`、`.upload-zone`、`.photo-grid`、`.photo-card`、`.modal`、`.status`、`.stats` 等，每个类对应一个 UI 区块。
- 命名约定：采用 BEM-like 的扁平类名风格（如 `btn-primary`、`btn-secondary`、`platform-btn`、`stat-card`），未使用嵌套或模块化方案。
- 主题色体系：以紫色渐变（`#667eea` → `#764ba2`）为主色调，用于按钮、链接、排名徽章等强调元素；背景使用浅灰（`#f5f5f5`）与白色卡片对比。
- 字体栈：使用系统字体栈 `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`，确保跨平台一致性。
- 交互模式：通过 DOM 操作（`document.getElementById`、`createElement`、`innerHTML`）动态渲染照片网格与结果列表，结合 WebSocket 实时推送状态更新。

## 4. 约定与约束
- **样式必须内联**：当前项目所有样式均写在 `<style>` 标签内，未见外部 CSS 文件引用。
- **无构建步骤**：HTML 文件直接由 FastAPI 模板引擎渲染，无需编译或打包。
- **响应式优先**：使用 CSS Grid 自动填充布局，未定义媒体查询断点，依赖浏览器默认行为适应不同屏幕尺寸。
- **状态类命名**：UI 状态通过类名切换（如 `.show`、`.active`、`.dragover`、`.success`、`.error`、`.info`），而非 CSS-in-JS 或状态管理库。
- **无障碍支持有限**：仅设置了 `lang="zh-CN"` 和基础语义标签，未使用 ARIA 属性或键盘导航增强。

该方案适合小型工具型 Web 界面，结构简单直观，但缺乏样式复用机制与设计系统支撑，扩展性受限。