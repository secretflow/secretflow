# Build and Upload SecretFlow Docker Image

1. Build docker image only.

```bash
bash build.sh -v <version> -r <docker reg> -l
```

NOTE:  
- -v <docker reg> is optional, which is default to be 'secretflow'.
- -l is optional, which marks the image as latest as well.

2. Build and upload

```bash
bash build.sh -v <version> -r <docker reg> -l -u
```

You should login the docker reg ahead.
