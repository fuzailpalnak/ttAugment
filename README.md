# TTAugment
![GitHub](https://img.shields.io/github/license/cypherics/TTAugment)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

Perform Augmentation during Inference and aggregate the results of all the applied augmentation to create a
final output

## Installation

    pip install git+https://github.com/cypherics/ttAugment.git#egg=ttAugment


## Supported Augmentation
Library supports all [color](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_color.html), 
[blur](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html) and [contrast](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_contrast.html)
transformation provided by [imgaug](https://imgaug.readthedocs.io/en/latest/) along with custom Geometric Transformation.

1. Mirror : Crop an image to `transform_dimension` and mirror pixels to match the size of `network_dimension`
2. CropScale : Crop an image to `transform_dimension` and rescale the image to match the size of `network_dimension`
3. NoAugment : Keep the input unchanged
4. Crop : Crop an image to `transform_dimension`
5. Rot : Rotate an Image
6. FlipHorizontal
7. FlipVertical 

## Parameters

How to use when test image is much *bigger* than what my models needs as input, Don't worry the library has it covered
it will generate fragments according to the specified dimension, so you can run inference on the desired dimension, 
and get the output as per the original test image.

- image_dimension - What is the size of my input image
        
        image_dimension=(2, 1500, 1500, 3) 

- transformer - list of augmentations to perform `transform_dimension` - _specifies what dimension is the network
expecting the input to be in_, if less than image_dimension, the library will generate smaller fragment of size `transform_dimension`
for inference and apply transformation over it
    
        transformers=[
        {
            "name": "CLAHE",
            "transform_dimension": (2, 1000, 1000, 3),
        },
        ],
        
    - Dealing with parameters during Scaling transformation, two transformation perform scaling on the test images
    For Scaling transformation `transform_dimension` and `network_dimension` are mandatory parameters
    
            transformers=[
            {
            "name": "Mirror",
            "transform_dimension": (2, 800, 800, 3),
            "network_dimension": (2, 1000, 1000, 3)
            },
            
            {
            "name": "ScaleCrop",
            "transform_dimension": (2, 800, 800, 3),
            "network_dimension": (2, 1000, 1000, 3)
            }
            ],
            
        The `network_dimension` parameter informs the library to override the network input
        and crop the image to `transform_dimension` and rescale it to `network_dimension` to get it as per network
        requirement
        
        And again the library will merge all the fragments to form the final output of `image_dimension`
    
    - For using `Rot` - Rotate add `"param": angle` as an additional argument 
    
If the test image has the same dimension to what the network expects, in that case just remove the `transform_dimension` param.

## Using is during Inference

##### Define tta object
```python

tta = Segmentation.populate_color(
        image_dimension=(2, 1500, 1500, 3),
         transformers=transformers) # transfromer as defined in parameters
```
 
##### Calling the generator
Input image is required to be a 4d numpy array of shape `(batch, height, width, channels)` 
```python

for image list(loop over all the images): 
    for transformation in tta.run():
        # Apply forward transfromation
        forward_image = tta.forward(transformation, image=image)

        # Apply normalization
        # Convert input to framework specific type
        # Perform inference
        inferred_image = model.predict(forward_image)

        # make sure to convert the inferred_image to 4d numpy array [batch, height, width, classes]
        reversed_image = tta.reverse(transformation, inferred_image)

        # Add the reversed image to transformation
        tta.update(transformation, reversed_image)
    
    # Access the output
    output = tta.transformations.output

```


    

