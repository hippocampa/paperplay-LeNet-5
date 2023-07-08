import fire


def train(model, batch_size, learning_rate, epochs):
    print(model, batch_size, learning_rate, epochs)


if __name__ == "__main__":
    fire.Fire(train)
