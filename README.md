# Evolve a neural network with a genetic algorithm
## Modified by [@mekomlusa](https://github.com/mekomlusa)

This is an example of how we can use a genetic algorithm in an attempt to find the optimal network parameters for classification tasks. It's adapted from [harvitronix](https://github.com/harvitronix/neural-network-genetic-algorithm)'s work, with the intention of realizing [Hinz et al.'s paper](http://jmlr.csail.mit.edu/manudb/autoreg/reviewer/eFOdi7rU0d4NIc9kqErL/getfile2/8636/manuscript/JMLR-17-098-1.pdf).

It's currently limited to only simple CNNs and uses the Keras library to build, train and validate.

Test results will be reported soon. Right now, the algorithm is able to achieve 67.88% accuracy on the CIFAR10 dataset, with 5 iterations and 10 networks each.

For more, see this blog post:
https://medium.com/@harvitronix/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164

For a more robust implementation that you can use in your projects, take a look at [Jan Liphardt's implementation, DeepEvolve](https://github.com/jliphard/DeepEvolve).

## To run

To run the genetic algorithm:

```python3 main.py```

You can set your network parameter choices by editing each of those files first. You can also choose whether to use the MNIST or CIFAR10 datasets. Simply set `dataset` to either `mnist` or `cifar10`.

## Credits

The genetic algorithm code is based on the code from this excellent blog post: https://lethain.com/genetic-algorithms-cool-name-damn-simple/

## Contributing

Have an optimization, idea, suggestion, bug report? Pull requests greatly appreciated!

## License

MIT
