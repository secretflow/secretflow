# Versioning

SecretFlow artifacts belongs to one of the following categories:

- "Releases", i.e. artifact versions follow the **x.y.z** format, where **x** is called "major version", **y** is called "minor version", **z** is called "patch version". We follow [Semantic Versioning 2.0.0](https://semver.org/) to update versions. e.g.
```
1.2.3
```

- "Pre-releases", i.e. artifact versions follow the **x.y.zb0** format. **Pre-release segment** is fixed to **b0**.  e.g.
```
0.8.7b0 # Pre-release of 0.8.7
```

- "Developmental releases", i.e. artifact versions follow the **x.y.z.devYYMMDD** or **x.y.z.devYYMMDD** format, where **YY** is the last two digits of year(00-99), **MM** is the zero-padded digits of month(01-12) and **DD** is zero-padded digits of day of the month(01-31). **YY**, **MM** and **DD** indicates the date when a Developmental release is published.  e.g.
```
1.1.0.dev230820 # Developmental release of 1.1.0 published at 23/08/20
1.3.0.dev231115 # Developmental release of 1.3.0 published at 23/11/15
```

- "Post-releases", i.e. artifact versions follow the **x.y.zb0.postN** or **x.y.z.postN** format. **x.y.zb0.postN** is a Developmental release of a Pre-release while **x.y.z.postN** is a Developmental release of a Release.  e.g.
```
1.1.0.post0 # The 1st Post-release of 1.1.0
1.3.0b0.post9 # The 10th Post-release of 1.3.0b0
```

## Releases

Release **x.y.z** meets the following conditions:
- Pre-release **x.y.zb0** has been published before, and
- Pre-release **x.y.zb0** has passed through evaluation.

## Pre-releases

Pre-release **x.y.zb0** is to support testing by external users prior to release **x.y.z**.

## Developmental releases

Developmental releases are to provide the preview of new features. They are considered unstable.

## Post-releases

Post-releases are to address minor errors of Releases and Pre-releases. They are considered unstable.

# 版本控制

SecretFlow 的 artifact 属于以下几种类别之一：

- "Releases"，即 artifact 版本遵循 **x.y.z** 格式，其中 **x** 称为 "主版本"， **y** 称为 "次版本"， **z** 称为 "修复版本"。我们遵循 [Semantic Versioning 2.0.0](https://semver.org/) 更新版本。例如：
```
1.2.3
```

- "Pre-releases"，即 artifact 版本遵循 **x.y.zb0** 格式。预发布字段固定为 **b0**。例如：
```
0.8.7b0 # 0.8.7的Pre-release
```

- "Developmental releases"，即 artifact 版本遵循 **x.y.z.devYYMMDD** 或 **x.y.z.devYYMMDD** 格式，其中 **YY** 是年份的最后两位（00-99）， **MM** 是月份的零填充数字（01-12）， **DD** 是当月日期的零填充数字（01-31）。**YY**， **MM** 和 **DD** 用以表示发布开发版的日期。例如：
```
1.1.0.dev230820 # 于23/08/20发布的1.1.0开发版
1.3.0.dev231115 # 于23/11/15发布的1.3.0开发版
```


- "Post-releases"，即 artifact 版本遵循 **x.y.zb0.postN** 或 **x.y.z.postN** 格式。 **x.y.zb0.postN** 是 Pre-releases 的 Post-releases，而 x.y.z.postN 是 Releases 的Post-releases 。例如：
```
1.1.0.post0 # 1.1.0的第1个Post-release
1.3.0b0.post9 # 1.3.0b0的第10个Post-release
```

## Releases

Release **x.y.z** 满足以下条件：

- Pre-release **x.y.zb0** 已经发布，并且
- Pre-release **x.y.zb0** 已经经过测试评估。

## Pre-releases

Pre-release **x.y.zb0** 是对外提供的 Release **x.y.z** 的测试版本。

## Developmental releases

Developmental releases 旨在提供新功能的预览。它们被认为是不稳定的。

## Post-releases

Post-releases 用于解决 Releases 和 Pre-releases 的轻微错误。它们被认为是不稳定的。
