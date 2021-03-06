# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.
## Model used

Tensorflow model for single shot detection at http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
Conversion is performed in modelConversion.ipynb

## Explaining Custom Layers

There might be reasons to handle custom layers either of your own operations or any that wasn't supported by the model converter. You can either manually create your own extension or simply use your operator with output of the network that was converted using the model converter. Custom are layers not directly supported by the model converter and the need to convert custom layers is due to operators not directly recognized or supported by the model converter. You seldomly need to work with custom layers.

## Comparing Model Performance

Performance is compared in a scientific manner using the coco dataset in MetricCalculationForInference.ipynb. The result for IOU and inference speed is shown at the end. The average IOU value on the coco dataset with the tensorflow model without conversion is 0.4086891604267041 while average inference time is 1.2551904726028442 while using the converted model with openvino toolkit have average IOU of 0.1362183 with threshhold of 0.5 and inference time on average being slightly under 0.07. The model lost some accuracy after conversion but is fast enough for edge computation.

Tabular summary
|        |IOU               |Inference time    |
|Vanilla |0.4086891604267041|1.2551904726028442|
|Optmized|0.1362183         |0.06908           |

## Assess Model Use Cases

1. Estimate people in a building
2. Estimate visitor count

The first would be useful for enforcing policy with respect to capacity IE elevator load and with recent pandemic situation store capacity. The second case is useful for helping with understanding potential needs and demands for things like office supply/groceries etc.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
The model for mobilenet SSD is likely trained primarily in high lighting situations where low lighting may effect model accuracy. The method used for tracking works well with a stationary camera as the bounding box is tracked per person with extension to the bounding box. If the camera is assumed to move then new methods of tracking will be needed. Potentially we can add more configuration flags for different tracking methods using mixin and dependency injections along with TDD to implement such need.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
