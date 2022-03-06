from jax import vmap, pmap, jit, grad, value_and_grad
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision
import jaxlib
from jax.scipy.special import logsumexp
from typing import Tuple, List

def main():
    #init mlp
    seed = 42
    def init_mlp(layer_params: List, key: jaxlib.xla_extension.DeviceArray, scale: float = 0.01) -> List:
        """Creates an MLP and returns its weights and biases."""
        params = list()
        keys = jax.random.split(key, num = len(layer_params)-1)
        for in_channels, out_channels, layer_key in zip(layer_params[:-1], layer_params[1:], keys):
            weight_key, bias_key = jax.random.split(layer_key)
            params.append(
                [
                    scale * jax.random.normal(weight_key, shape = (out_channels, in_channels)), #weights
                    scale * jax.random.normal(bias_key, shape = (out_channels,)) #biases
                ]
            )
        return params
    key = jax.random.PRNGKey(seed=seed)
    mlp_params = init_mlp([784, 512, 256, 10], key = key)
    print(f"mlp structure: \n {jax.tree_map(lambda x: x.shape, mlp_params)}")
    #define forward function
    def mlp_forward(params, x):
        """predicting on mlp"""
        hidden_layers = params[:-1]
        activation = x
        for weight, bias in hidden_layers:
            activation = jax.nn.relu(jnp.dot(weight, activation) + bias)

        weight_logit, bias_logit = params[-1]
        logits = jnp.dot(weight_logit, activation) + bias_logit

        return logits - logsumexp(logits)

    img_size = (28, 28)
    batched_mlp_forward = vmap(mlp_forward, in_axes=(None, 0))

    #loading data
    def custom_transform(image: torch.Tensor) -> jnp.ndarray:
        """flatten a given image and convert it to jnp array"""
        return jnp.reshape(image.cpu().numpy(), (-1, )).astype(jnp.float32)
    def custom_collate_fn(batch):
        transposed_data = list(zip(*batch))
        return np.array(transposed_data[0]), np.array(transposed_data[1])

    batch_size_train = batch_size_test = batch_size = 20
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,)), 
                               custom_transform                             
                             ])), batch_size=batch_size_train, shuffle=True,
                             collate_fn = custom_collate_fn)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                               custom_transform
                             ])), batch_size=batch_size_test, shuffle=True,
                             collate_fn = custom_collate_fn)
#train loop fn
    def loss_fn(params, batch_x, batch_y):
        predictions = batched_mlp_forward(params, batch_x)
        return -jnp.mean(predictions * batch_y.reshape(-1, 1))
    @jit            
    def update(params, batch_x, batch_y, lr = 1e-3):
        """backpropagate through"""
        loss, grads = value_and_grad(loss_fn)(params, batch_x, batch_y)
        return loss, jax.tree_multimap(
                                lambda p, g: p - lr*g, params, grads
                            )
                            
    def accuracy(params, dataset_imgs, dataset_lbls):
        pred_classes = jnp.argmax(batched_mlp_forward(params, dataset_imgs), axis = 1)
        return jnp.mean(dataset_lbls == pred_classes)

    n_epochs = 100
    for epoch in range(n_epochs):
        for idx, (imgs, lbls) in  enumerate(train_loader):
            gt_labels = jax.nn.one_hot(lbls, num_classes = len(torchvision.datasets.MNIST.classes))
            loss, mlp_params = update(mlp_params, imgs, lbls)
            if idx %100: print(loss)
if __name__ == "__main__": main()

