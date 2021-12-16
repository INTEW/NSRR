# Neural Supersampling for Real-time Rendering with Pytorch

Create super-resolution images from low-resolution in real time. Non-official implementation of the paper [NSRR](https://research.fb.com/wp-content/uploads/2020/06/Neural-Supersampling-for-Real-time-Rendering.pdf) by Facebook Reality Labs in 2020. A [blog post](https://research.fb.com/blog/2020/07/introducing-neural-supersampling-for-real-time-rendering/) is available with more details.

**This is a work-in-progress, the report about our advances is [available here](doc/report-v1.pdf)**

## Getting started

### Requirements

You need Python at least 3.5 (3.6 recommended).

To install other dependencies, you can use pip with :

```bash
pip install -r requirements.txt
```

### Usage

#### Dataset

You can generate your own dataset from [this Unity 2019 project](https://gitlab.com/piptouque/unity_ml_dataset) it will export the view, the depth buffer and the motion vector of the game camera in any resolution you want. We've setup a few animated scenes already, check the repo for more details.

**Pre-created dataset will be downloadable soon**

In order to be loaded using `NSRRDataLoader`, the dataset should be structured like so:

```
[root_dir]
│
└───View
│   │   img_1.png
│   │   img_2.png
│    ...
│   
└───Depth
│   │   img_1.png
│   │   img_2.png
│    ...
│   
└───Motion
│   │   img_1.png
│   │   img_2.png
│    ...
```

Where `root_dir` is the `data_dir` in `config.json` of `NSRRDataLoader`

**Note that corresponding tuples of (view, depth, motion) images files should share the same name, as they cannot be grouped together otherwise.**

#### Unit testing

You can remove `-d 1` if you do not have a CUDA-capable GPU.

```bash
python3 debug.py -c 'config.json' -d 1
```

### Miscellaneous information

Using :

* Pytorch project template at:
  https://github.com/victoresque/pytorch-template

* Pytorch implementation of SSIM:
  https://github.com/Po-Hsun-Su/pytorch-ssim

* Pytorch implementation of colour-space conversions:
  https://github.com/jorge-pessoa/pytorch-colors
