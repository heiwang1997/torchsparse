- Build
```bash
TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6+PTX" python setup.py bdist_wheel
```

- Upload to S3
```bash
awsm cp dist/torchsparse-2.0.0b0-cp310-cp310-linux_x86_64.whl s3://nksr/dev-whls/ $AWSGA
```
