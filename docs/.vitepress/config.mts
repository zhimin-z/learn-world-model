/// <reference types="node" />
import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

const docsBase = "/learn-world-model/";
const brandLogo = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23D95C41" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12.1" y1="11.9" x2="18.9" y2="8.2" /><line x1="12.1" y1="12.1" x2="20.3" y2="12.9" /><line x1="12.2" y1="12.4" x2="16.6" y2="19.1" /><line x1="11.8" y1="12.4" x2="7.3" y2="19.2" /><line x1="11.9" y1="12.1" x2="3.7" y2="13.3" /><line x1="11.8" y1="11.7" x2="7.8" y2="4.4" /></svg>';
const githubRepoTreeLink = "https://github.com/zhimin-z/learn-world-model/tree/main";

const zhWorldModelItems = [
  { text: "欢迎", link: "/zh/world-model/" },
  {
    text: "第一讲：内部仿真与历史脉络",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-01-internal-simulation/" },
      { text: "思想基石", link: "/zh/lectures/lecture-01-internal-simulation/01-foundations" },
      { text: "四个时代", link: "/zh/lectures/lecture-01-internal-simulation/02-four-eras" },
      { text: "为什么是现在", link: "/zh/lectures/lecture-01-internal-simulation/03-why-now" },
    ]
  },
  {
    text: "第二讲：观测编码与潜在动力学",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-02-encode-and-dynamics/" },
      { text: "Part A：观测编码", link: "/zh/lectures/lecture-02-encode-and-dynamics/01-encoding" },
      { text: "Part B：潜在动力学", link: "/zh/lectures/lecture-02-encode-and-dynamics/02-dynamics" },
    ]
  },
  {
    text: "第三讲：架构模式、学习范式与规划",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-03-architecture-patterns/" },
      { text: "Part A：六大架构族", link: "/zh/lectures/lecture-03-architecture-patterns/01-architectures" },
      { text: "Part B：规划机制", link: "/zh/lectures/lecture-03-architecture-patterns/02-planning" },
    ]
  },
  {
    text: "第四讲：按模型评估",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-04-evaluation-by-model/" },
      { text: "Dreamer、MuZero 与 TD-MPC", link: "/zh/lectures/lecture-04-evaluation-by-model/01-model-metrics" },
      { text: "STORM、扩散世界模型与漂移", link: "/zh/lectures/lecture-04-evaluation-by-model/02-storm-diffusion-drift" },
      { text: "真实部署与课程总结", link: "/zh/lectures/lecture-04-evaluation-by-model/03-deployment" },
    ]
  },
  {
    text: "第五讲：前沿思辨",
    collapsed: false,
    items: [
      { text: "导读", link: "/zh/lectures/lecture-05-frontier-debates/" },
      { text: "语言是鸦片与 Bitter Lesson", link: "/zh/lectures/lecture-05-frontier-debates/01-language-and-bitter-lesson" },
      { text: "AGI 与两条路的收敛", link: "/zh/lectures/lecture-05-frontier-debates/02-agi-and-convergence" },
      { text: "数据从哪里来与结语", link: "/zh/lectures/lecture-05-frontier-debates/03-data-and-future" },
    ]
  },
];

const zhProjectItems = [
  { text: "欢迎", link: "/zh/projects/" },
  { text: "P01：训练 VAE 编码器", link: "/zh/projects/project-01-vae-encoder/" },
  { text: "P02：构建潜在动力学模型", link: "/zh/projects/project-02-latent-dynamics/" },
  { text: "P03：完整 Dreamer 流水线", link: "/zh/projects/project-03-dreamer-pipeline/" },
  { text: "P04：实现 TD-MPC 规划", link: "/zh/projects/project-04-td-mpc/" },
  { text: "P05：STORM + 三模型评估仪表盘", link: "/zh/projects/project-05-storm-dashboard/" },
];

const enWorldModelItems = [
  { text: "Welcome", link: "/en/world-model/" },
  {
    text: "Lecture 01: Internal Simulation",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-01-internal-simulation/" },
      { text: "Foundations", link: "/en/lectures/lecture-01-internal-simulation/01-foundations" },
      { text: "Four Eras", link: "/en/lectures/lecture-01-internal-simulation/02-four-eras" },
      { text: "Why Now", link: "/en/lectures/lecture-01-internal-simulation/03-why-now" },
    ]
  },
  {
    text: "Lecture 02: Encoding & Latent Dynamics",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-02-encode-and-dynamics/" },
      { text: "Part A: Observation Encoding", link: "/en/lectures/lecture-02-encode-and-dynamics/01-encoding" },
      { text: "Part B: Latent Dynamics", link: "/en/lectures/lecture-02-encode-and-dynamics/02-dynamics" },
    ]
  },
  {
    text: "Lecture 03: Architecture, Paradigms & Planning",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-03-architecture-patterns/" },
      { text: "Part A: Six Architecture Families", link: "/en/lectures/lecture-03-architecture-patterns/01-architectures" },
      { text: "Part B: Planning Mechanisms", link: "/en/lectures/lecture-03-architecture-patterns/02-planning" },
    ]
  },
  {
    text: "Lecture 04: Evaluation by Model",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-04-evaluation-by-model/" },
      { text: "Dreamer, MuZero & TD-MPC", link: "/en/lectures/lecture-04-evaluation-by-model/01-model-metrics" },
      { text: "STORM, Diffusion & Drift", link: "/en/lectures/lecture-04-evaluation-by-model/02-storm-diffusion-drift" },
      { text: "Deployment & Summary", link: "/en/lectures/lecture-04-evaluation-by-model/03-deployment" },
    ]
  },
  {
    text: "Lecture 05: Frontier Debates",
    collapsed: false,
    items: [
      { text: "Overview", link: "/en/lectures/lecture-05-frontier-debates/" },
      { text: "Language as Opium & Bitter Lesson", link: "/en/lectures/lecture-05-frontier-debates/01-language-and-bitter-lesson" },
      { text: "AGI & Convergence", link: "/en/lectures/lecture-05-frontier-debates/02-agi-and-convergence" },
      { text: "Data & Future", link: "/en/lectures/lecture-05-frontier-debates/03-data-and-future" },
    ]
  },
];

const enProjectItems = [
  { text: "Welcome", link: "/en/projects/" },
  { text: "P01: Train a VAE Encoder", link: "/en/projects/project-01-vae-encoder/" },
  { text: "P02: Build a Latent Dynamics Model", link: "/en/projects/project-02-latent-dynamics/" },
  { text: "P03: Full Dreamer Pipeline", link: "/en/projects/project-03-dreamer-pipeline/" },
  { text: "P04: Implement TD-MPC Planning", link: "/en/projects/project-04-td-mpc/" },
  { text: "P05: STORM + Three-Model Dashboard", link: "/en/projects/project-05-storm-dashboard/" },
];

export default withMermaid(
  defineConfig({
    base: docsBase,
    title: "Learn World Models",
    description:
      "A project-based curriculum on world models — from VAE encoders and latent dynamics to Dreamer, TD-MPC, STORM, and frontier debates on language vs physical grounding.",
    cleanUrls: true,
    srcExclude: ["temp/**"],
    ignoreDeadLinks: true,
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: brandLogo }]
    ],
    themeConfig: {
      logo: brandLogo,
      search: {
        provider: "local"
      },
      socialLinks: [{ icon: "github", link: githubRepoTreeLink }]
    },
    markdown: {
      theme: {
        light: 'github-light',
        dark: 'github-dark'
      },
      math: true
    },
    mermaid: {
      theme: 'base',
      themeVariables: {
        primaryColor: '#F4F3EE',
        primaryBorderColor: '#D1D1D1',
        primaryTextColor: '#1A1A1A',
        lineColor: '#B3B3B3',
        fontFamily: 'Inter, sans-serif',
        fontSize: '18px'
      },
      flowchart: {
        nodeSpacing: 40,
        rankSpacing: 56,
        padding: 12
      }
    },
    locales: {
      root: {
        label: "简体中文",
        lang: "zh-CN",
        link: "/zh/"
      },
      en: {
        label: "English",
        lang: "en",
        link: "/en/",
        themeConfig: {
          nav: [
            { text: "Lectures", link: "/en/world-model/", activeMatch: '^/en/(lectures/|world-model/)' },
            { text: "Projects", link: enProjectItems[0].link, activeMatch: '^/en/projects/' },
          ],
          sidebar: {
            '/en/projects/': [{ text: "Projects", items: enProjectItems }],
            '/en/world-model/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-01-internal-simulation/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-02-encode-and-dynamics/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-03-architecture-patterns/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-04-evaluation-by-model/': [{ text: "World Models", items: enWorldModelItems }],
            '/en/lectures/lecture-05-frontier-debates/': [{ text: "World Models", items: enWorldModelItems }],
          },
          socialLinks: [{ icon: "github", link: githubRepoTreeLink }]
        }
      },
      zh: {
        label: "简体中文",
        lang: "zh-CN",
        link: "/zh/",
        themeConfig: {
          nav: [
            { text: "讲义", link: "/zh/world-model/", activeMatch: '^/zh/(lectures/|world-model/)' },
            { text: "项目", link: zhProjectItems[0].link, activeMatch: '^/zh/projects/' },
          ],
          sidebar: {
            '/zh/projects/': [{ text: "项目", items: zhProjectItems }],
            '/zh/world-model/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-01-internal-simulation/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-02-encode-and-dynamics/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-03-architecture-patterns/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-04-evaluation-by-model/': [{ text: "世界模型", items: zhWorldModelItems }],
            '/zh/lectures/lecture-05-frontier-debates/': [{ text: "世界模型", items: zhWorldModelItems }],
          },
          outline: {
            level: [2, 3]
          },
          docFooter: {
            prev: "上一篇",
            next: "下一篇"
          },
          lastUpdated: {
            text: "最后更新于"
          },
          returnToTopLabel: "回到顶部",
          sidebarMenuLabel: "菜单",
          darkModeSwitchLabel: "主题",
          lightModeSwitchTitle: "切换到浅色模式",
          darkModeSwitchTitle: "切换到深色模式",
          socialLinks: [{ icon: "github", link: githubRepoTreeLink }]
        }
      }
    }
}));
