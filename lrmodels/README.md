# IR models
place your optimized ir models here

One folder corresponds to one precision format 

eg. my lrmodels
```bash
$ tree -a
.
├── FP16
│   ├── frozen_inference_graph.bin
│   ├── frozen_inference_graph.mapping
│   └── frozen_inference_graph.xml
├── FP32
│   ├── frozen_inference_graph.bin
│   ├── frozen_inference_graph.mapping
│   └── frozen_inference_graph.xml
└── INT8
    ├── frozen_inference_graph_i8.bin
    └── frozen_inference_graph_i8.xml
```
