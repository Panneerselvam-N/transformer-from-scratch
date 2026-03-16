from train import causal_mask


def main():
    for size in [1, 2, 5]:
        mask = causal_mask(size)
        print(f"causal_mask({size}):")
        print(mask)
        print()


if __name__ == "__main__":
    main()
