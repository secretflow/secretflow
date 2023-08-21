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
- Pre-release **x.y.zb0** has been externally inspected.

## Pre-releases

Pre-release **x.y.zb0** meets the following conditions:
- At least one Developmental release **x.y.z.devYYMMDD** has been published before, and
- The last published Developmental release has been internally inspected.

## Developmental releases

Developmental releases are considered unstable and not inspected.

## Post-releases

Post-releases are to address minor errors of Releases and Pre-releases. They are considered unstable and not inspected.

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

- 先发布了Pre-release **x.y.zb0**，并且
- Pre-release **x.y.zb0** 已经经过外部检查。

## Pre-releases

Pre-release **x.y.zb0** 满足以下条件：

- 至少发布了一个开发版 **x.y.z.devYYMMDD**，并且
- 最后发布的开发版已经经过内部检查。

## Developmental releases

Developmental releases被认为是不稳定的，未经过检查。

## Post-releases

Post-releases用于解决Releases和Pre-releases的轻微错误。它们被认为是不稳定的，未经过检查。
