<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="./static/mobilerun-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/mobilerun.png">
  <img src="./static/mobilerun.png"  width="full">
</picture>


<div align="center">

[![Docs](https://img.shields.io/badge/Docs-📕-0D9373?style=for-the-badge)](https://docs.mobilerun.ai)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-0D9373?style=for-the-badge)](https://cloud.mobilerun.ai/sign-in?waitlist=true)


[![GitHub stars](https://img.shields.io/github/stars/droidrun/mobilerun?style=social)](https://github.com/droidrun/mobilerun/stargazers)
[![mobilerun.ai](https://img.shields.io/badge/mobilerun.ai-white)](https://mobilerun.ai)
[![Twitter Follow](https://img.shields.io/twitter/follow/mobilerun_ai?style=social)](https://x.com/mobilerun_ai)
[![Discord](https://img.shields.io/discord/1360219330318696488?color=white&label=Discord&logo=discord&logoColor=white)](https://discord.gg/ZZbKEZZkwK)
[![Benchmark](https://img.shields.io/badge/Benchmark-91.4﹪-white)](https://mobilerun.ai/benchmark)



<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=dark&period=daily&t=1753948032207">
  <source media="(prefers-color-scheme: light)" srcset="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=neutral&period=daily&t=1753948125523">
  <a href="https://www.producthunt.com/products/mobilerun-framework-for-mobile-agent?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_source=badge-mobilerun" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=983810&theme=neutral&period=daily&t=1753948125523" alt="Mobilerun - Give&#0032;AI&#0032;native&#0032;control&#0032;of&#0032;physical&#0032;&#0038;&#0032;virtual&#0032;phones&#0046; | Product Hunt" style="width: 200px; height: 54px;" width="200" height="54" /></a>
</picture>


[Deutsch](https://zdoc.app/de/droidrun/mobilerun) | 
[Español](https://zdoc.app/es/droidrun/mobilerun) | 
[français](https://zdoc.app/fr/droidrun/mobilerun) | 
[日本語](https://zdoc.app/ja/droidrun/mobilerun) | 
[한국어](https://zdoc.app/ko/droidrun/mobilerun) | 
[Português](https://zdoc.app/pt/droidrun/mobilerun) | 
[Русский](https://zdoc.app/ru/droidrun/mobilerun) | 
[中文](https://zdoc.app/zh/droidrun/mobilerun)

</div>



Mobilerun is a cloud solution powered by Mobilerun a powerful framework for controlling Android and iOS devices through LLM agents. It allows you to automate device interactions using natural language commands. [Checkout our benchmark results](https://mobilerun.ai/benchmark)


- 🤖 Control Android and iOS devices with natural language commands
- 🔀 Supports multiple LLM providers (OpenAI, Anthropic, Gemini, Ollama, DeepSeek)
- 🧠 Planning capabilities for complex multi-step tasks
- 💻 Easy to use CLI with enhanced debugging features
- 🐍 Extendable Python API for custom automations
- 📸 Screenshot analysis for visual understanding of the device
- 🫆 Execution tracing with Arize Phoenix

## 📦 Installation

> **Note:** Python 3.14 is not currently supported. Please use Python 3.11 – 3.13.

```bash
pip install mobilerun
```

## 🚀 Quickstart

### 1. Install the portal on your device
```bash
mobilerun setup
```

### 2. Configure your LLM provider
```bash
mobilerun configure
```
This walks you through choosing a provider (Gemini, OpenAI, Anthropic, etc.), auth method (API key or OAuth), and model.

### 3. Run a command
```bash
mobilerun run "open settings and turn on dark mode"
```

Read the full guide in [our docs](https://docs.mobilerun.ai/v3/quickstart)!

[![Quickstart Video](https://img.youtube.com/vi/4WT7FXJah2I/0.jpg)](https://www.youtube.com/watch?v=4WT7FXJah2I)

## 🎬 Demo Videos

1. **Accommodation booking**: Let Mobilerun search for an apartment for you

   [![Mobilerun Accommodation Booking Demo](https://img.youtube.com/vi/VUpCyq1PSXw/0.jpg)](https://youtu.be/VUpCyq1PSXw)

<br>

2. **Trend Hunter**: Let Mobilerun hunt down trending posts

   [![Mobilerun Trend Hunter Demo](https://img.youtube.com/vi/7V8S2f8PnkQ/0.jpg)](https://youtu.be/7V8S2f8PnkQ)

<br>

3. **Streak Saver**: Let Mobilerun save your streak on your favorite language learning app

   [![Mobilerun Streak Saver Demo](https://img.youtube.com/vi/B5q2B467HKw/0.jpg)](https://youtu.be/B5q2B467HKw)


## 💡 Example Use Cases

- Automated UI testing of mobile applications
- Creating guided workflows for non-technical users
- Automating repetitive tasks on mobile devices
- Remote assistance for less technical users
- Exploring mobile UI with natural language commands

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 

## Security Checks

To ensure the security of the codebase, we have integrated security checks using `bandit` and `safety`. These tools help identify potential security issues in the code and dependencies.

### Running Security Checks

Before submitting any code, please run the following security checks:

1. **Bandit**: A tool to find common security issues in Python code.
   ```bash
   bandit -r mobilerun
   ```

2. **Safety**: A tool to check your installed dependencies for known security vulnerabilities.
   ```bash
   safety scan
   ```
