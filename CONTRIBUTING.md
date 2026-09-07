# Contributing to EasyWAM

[中文](#中文)

EasyWAM is built with the community. Contributions that fix bugs, improve documentation, add model or benchmark support, or make World Action Model research more accessible are welcome.

## Report a bug

Open an [issue](https://github.com/OpenMOSS/EasyWAM/issues) and include:

- A concise description of the problem and expected behavior.
- Steps and commands that reproduce it.
- The task configuration and any relevant overrides.
- Environment details, logs, and the complete error traceback.

Do not include private datasets, credentials, access tokens, or other sensitive information.

## Propose a feature

Open an issue before starting a substantial change. Describe the problem, the proposed behavior, affected models or benchmarks, and any compatibility considerations. This gives maintainers and contributors a chance to agree on scope before implementation begins.

## Submit a pull request

- Keep the change focused and avoid unrelated cleanup.
- Follow the existing code and configuration conventions.
- Add or update tests for behavior changes when practical.
- Update the relevant English and Chinese documentation when user-facing behavior changes.
- Describe the change, compatibility impact, and verification commands in the pull request.
- Do not commit datasets, checkpoints, generated outputs, credentials, or machine-specific paths.

Before submitting, run the tests relevant to your change and check formatting with:

```bash
git diff --check
```

If you would like to contribute or discuss a change with the team, contact [siyinwang20@fudan.edu.cn](mailto:siyinwang20@fudan.edu.cn).

## 中文

EasyWAM 由社区共同建设。我们欢迎修复 bug、改进文档、增加模型或 benchmark 支持，以及帮助降低 World Action Model 研究门槛的贡献。

### 报告问题

请创建 [Issue](https://github.com/OpenMOSS/EasyWAM/issues)，并提供：

- 问题与预期行为的简要说明。
- 可以复现问题的步骤和命令。
- 使用的 task 配置及相关 overrides。
- 环境信息、日志和完整的错误堆栈。

请勿提交私有数据集、凭据、访问令牌或其他敏感信息。

### 提出功能建议

对于较大的改动，请先创建 Issue，说明待解决的问题、预期行为、受影响的模型或 benchmark，以及兼容性方面的考虑，以便在开始实现前确认范围。

### 提交 Pull Request

- 保持改动聚焦，避免混入无关清理。
- 遵循现有代码与配置约定。
- 行为发生变化时，尽可能增加或更新测试。
- 面向用户的行为发生变化时，同步更新相关中英文文档。
- 在 Pull Request 中说明改动内容、兼容性影响和验证命令。
- 不要提交数据集、checkpoint、生成结果、凭据或机器相关的路径。

提交前请运行与改动相关的测试，并检查格式：

```bash
git diff --check
```

如果希望参与贡献或与团队讨论改动，请联系 [siyinwang20@fudan.edu.cn](mailto:siyinwang20@fudan.edu.cn)。
