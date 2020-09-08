import mnist.train
import amazon.train
import amazon.retrain
import mit.train


if __name__ == "__main__":
    mnist.train.main()
    mit.train.main()
    amazon.train.main()
    amazon.retrain.main()