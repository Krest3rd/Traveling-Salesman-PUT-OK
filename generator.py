from random import randint

def geterate_instance(count):
    with open('instance.txt', 'w') as f:
        f.write(f"{count}\n")
        instance = []
        for i in range(count):
            x = randint(0, 1000)
            y = randint(0, 1000)
            if (x, y) not in instance:
                f.write(f"{i+1} {x} {y}\n")


if __name__ == "__main__":
    geterate_instance(20)