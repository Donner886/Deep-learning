# Tensorflow basics
This guide provides a quick overview of TensorFlow basics. Each section of this doc is an overview of a larger topic-you can find links to full guides at the end of each section.

Tensorflow is an end-to-end platform for machine learning. It supports the following:
- Multidimensional-array based numeric computation (similar to NumPy)
- GPU and distributed processing
- Automatic differentiation 
- Model construction, training, and export
- And more

### Tensor
Tensorflow operates on multidimensional arrays or tensors represented by the `tf.Tensor` object. Here is a two-dimensional tensor:

```python