"""Spaces.Product for Theano."""
from garage.misc import ext
from garage.spaces import Product as GarageProduct


class Product(GarageProduct):
    """Theano extension of garage.Product."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in Theano.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        return ext.new_tensor(
            name=name,
            ndim=extra_dims + 1,
            dtype=self._common_dtype,
        )
